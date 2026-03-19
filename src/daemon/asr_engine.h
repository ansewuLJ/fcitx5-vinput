#pragma once

#include "common/core_config.h"
#include "common/model_manager.h"
#include "daemon/qwen_asr_client.h"
#include "vad_trimmer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct SherpaOnnxOfflineRecognizer;

struct AsrConfig {
  std::string language;
  std::string hotwords_file;
  int thread_num = 4;
  bool normalize_audio = true;
  bool vad_enabled = true;
  std::string vad_model_path;
};

// Abstract ASR backend interface (renamed to avoid conflict with config struct)
class AsrBackendBase {
public:
  virtual ~AsrBackendBase() = default;
  virtual bool Init(const AsrConfig &config) = 0;
  virtual std::string Infer(const std::vector<int16_t> &pcm_data) = 0;
  virtual void Shutdown() = 0;
  virtual bool IsInitialized() const = 0;
  virtual std::string Name() const = 0;
};

// Sherpa-ONNX local backend
class SherpaOnnxBackend : public AsrBackendBase {
public:
  SherpaOnnxBackend(const ModelInfo &model_info);
  ~SherpaOnnxBackend() override;

  bool Init(const AsrConfig &config) override;
  std::string Infer(const std::vector<int16_t> &pcm_data) override;
  void Shutdown() override;
  bool IsInitialized() const override { return initialized_; }
  std::string Name() const override { return "sherpa-onnx"; }

private:
  ModelInfo model_info_;
  const SherpaOnnxOfflineRecognizer *recognizer_ = nullptr;
  bool initialized_ = false;
  bool normalize_audio_ = true;
  VadTrimmer vad_;
};

// Qwen3-ASR HTTP backend
class QwenHttpBackend : public AsrBackendBase {
public:
  QwenHttpBackend(const AsrBackend::QwenHttp &config);
  ~QwenHttpBackend() override = default;

  bool Init(const AsrConfig &config) override;
  std::string Infer(const std::vector<int16_t> &pcm_data) override;
  void Shutdown() override;
  bool IsInitialized() const override { return initialized_; }
  std::string Name() const override { return "qwen-http"; }

  // Load hotwords to server
  bool LoadHotwords(const std::map<std::string, std::map<std::string, float>> &hotwords);

private:
  AsrBackend::QwenHttp http_config_;
  std::unique_ptr<qwen_asr::Client> client_;
  std::string language_;
  bool initialized_ = false;
};

// Main ASR engine - wrapper around backends
class AsrEngine {
public:
  static constexpr std::size_t kMinSamplesForInference = 8000; // 0.5 s @ 16 kHz

  AsrEngine();
  ~AsrEngine();

  // Initialize with local sherpa-onnx backend
  bool Init(const ModelInfo &info, const AsrConfig &asr_config);

  // Initialize with Qwen HTTP backend
  bool Init(const AsrBackend::QwenHttp &http_config, const AsrConfig &asr_config);

  // Initialize from CoreConfig (auto-select backend)
  bool Init(const CoreConfig &core_config, const AsrConfig &asr_config);

  std::string Infer(const std::vector<int16_t> &pcm_data);
  void Shutdown();
  bool IsInitialized() const;
  std::string BackendName() const;

  // Get backend for advanced operations (e.g., hotwords)
  AsrBackendBase* GetBackend() { return backend_.get(); }

private:
  std::unique_ptr<AsrBackendBase> backend_;
  bool initialized_ = false;
};
