#pragma once

#include "model_manager.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct SherpaOnnxOfflineRecognizer;

class AsrEngine {
public:
  static constexpr std::size_t kMinSamplesForInference = 8000; // 0.5 s @ 16 kHz

  AsrEngine();
  ~AsrEngine();

  bool Init(const ModelInfo &info, int thread_num = 4);
  std::string Infer(const std::vector<int16_t> &pcm_data);
  void Shutdown();
  bool IsInitialized() const;

private:
  const SherpaOnnxOfflineRecognizer *recognizer_ = nullptr;
  bool initialized_ = false;
};
