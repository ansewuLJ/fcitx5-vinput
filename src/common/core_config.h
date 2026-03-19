#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "common/postprocess_scene.h"

struct LlmProvider {
  std::string name;
  std::string base_url;
  std::string api_key;
};

// ASR Backend type: "local" (sherpa-onnx) or "qwen-http" (Qwen3-ASR HTTP API)
struct AsrBackend {
  std::string type{"local"};  // "local" or "qwen-http"

  // Qwen3-ASR HTTP backend config
  struct QwenHttp {
    std::string url{"http://127.0.0.1:8000"};
    std::string model{"Qwen/Qwen3-ASR-0.6B"};
    int timeoutMs{30000};
    bool streaming{false};
  } qwenHttp;

  bool IsLocal() const { return type == "local"; }
  bool IsQwenHttp() const { return type == "qwen-http"; }
};

struct CoreConfig {
  std::string captureDevice{"default"};
  std::string activeModel{"paraformer-zh"};
  std::string modelBaseDir;
  std::string registryUrl{"https://raw.githubusercontent.com/xifan2333/vinput-models/main/registry.json"};

  std::string defaultLanguage{"zh"};

  std::string hotwordsFile;

  // Hotwords in JSON format (for Qwen3-ASR HTTP backend)
  // Format: {"category": {"word": weight, ...}, ...}
  std::map<std::string, std::map<std::string, float>> hotwordsJson;

  struct Llm {
    std::vector<LlmProvider> providers;
  } llm;

  struct Asr {
    bool normalizeAudio{true};
    struct Vad {
      bool enabled{true};
    } vad;
  } asr;

  struct Scenes {
    std::string activeScene{"default"};
    std::vector<vinput::scene::Definition> definitions;
  } scenes;

  // ASR backend configuration
  AsrBackend asrBackend;
};

// API Functions
CoreConfig LoadCoreConfig();
bool SaveCoreConfig(const CoreConfig &config);
std::string GetCoreConfigPath();

void NormalizeCoreConfig(CoreConfig *config);
const LlmProvider *ResolveLlmProvider(const CoreConfig &config,
                                      const std::string &provider_id);
const vinput::scene::Definition *FindCommandScene(const CoreConfig &config);
std::filesystem::path ResolveModelBaseDir(const CoreConfig &config);

constexpr std::string_view kCommandSceneId = "__command__";
