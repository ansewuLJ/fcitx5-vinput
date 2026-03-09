#include "asr_engine.h"

#include <sherpa-onnx/c-api/c-api.h>

#include <cstdio>

AsrEngine::AsrEngine() = default;

AsrEngine::~AsrEngine() { Shutdown(); }

bool AsrEngine::Init(const ModelInfo &info, int thread_num) {
  if (initialized_) {
    return true;
  }

  SherpaOnnxOfflineRecognizerConfig config = {};
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  // Default decoding configs
  config.decoding_method = "greedy_search";
  config.max_active_paths = 4;
  config.hotwords_score = 1.5f;
  config.blank_penalty = 0.0f;

  config.model_config.tokens = info.tokens.c_str();
  config.model_config.num_threads = thread_num;
  config.model_config.provider = "cpu";

  if (info.model_type == "paraformer") {
    config.model_config.paraformer.model = info.model.c_str();
    config.model_config.model_type = "paraformer";
    config.model_config.modeling_unit =
        info.modeling_unit.empty() ? "cjkchar" : info.modeling_unit.c_str();
  } else if (info.model_type == "sense_voice") {
    config.model_config.sense_voice.model = info.model.c_str();
    config.model_config.sense_voice.language = info.language.c_str();
    config.model_config.sense_voice.use_itn = info.use_itn ? 1 : 0;
    config.model_config.model_type = "sense_voice";
  } else if (info.model_type == "whisper") {
    // Just as an example, whisper usually requires encoder and decoder
    // but since we only have one 'model' path right now, we can log an error or
    // try to map. Assuming user puts encoder in "model" for now, but whisper
    // needs both. For actual whisper support, ModelInfo needs encoder and
    // decoder fields. We'll leave it out or handle basic for now.
    fprintf(
        stderr,
        "vinput: whisper model is partially configured, requires more args\n");
    return false;
  } else {
    fprintf(stderr, "vinput: unsupported model type '%s'\n",
            info.model_type.c_str());
    return false;
  }

  recognizer_ = SherpaOnnxCreateOfflineRecognizer(&config);
  if (!recognizer_) {
    fprintf(stderr,
            "vinput: failed to create sherpa-onnx recognizer for type '%s'\n",
            info.model_type.c_str());
    return false;
  }

  initialized_ = true;
  fprintf(
      stderr,
      "vinput: sherpa-onnx ASR initialized successfully (type: %s, lang: %s)\n",
      info.model_type.c_str(), info.language.c_str());
  return true;
}

std::string AsrEngine::Infer(const std::vector<int16_t> &pcm_data) {
  if (!initialized_ || pcm_data.empty()) {
    return "";
  }

  if (pcm_data.size() < kMinSamplesForInference) {
    fprintf(stderr,
            "vinput: skipping ASR for short audio: %zu samples (%.1f ms)\n",
            pcm_data.size(),
            static_cast<double>(pcm_data.size()) * 1000.0 / 16000.0);
    return "";
  }

  // sherpa-onnx expects float samples in [-1, 1]
  std::vector<float> samples(pcm_data.size());
  for (size_t i = 0; i < pcm_data.size(); ++i) {
    samples[i] = static_cast<float>(pcm_data[i]) / 32768.0f;
  }

  const SherpaOnnxOfflineStream *stream =
      SherpaOnnxCreateOfflineStream(recognizer_);
  if (!stream) {
    fprintf(stderr, "vinput: failed to create sherpa-onnx stream\n");
    return "";
  }

  SherpaOnnxAcceptWaveformOffline(stream, 16000, samples.data(),
                                  static_cast<int32_t>(samples.size()));
  SherpaOnnxDecodeOfflineStream(recognizer_, stream);

  const SherpaOnnxOfflineRecognizerResult *result =
      SherpaOnnxGetOfflineStreamResult(stream);
  std::string text;
  if (result && result->text) {
    text = result->text;
  }

  if (result) {
    SherpaOnnxDestroyOfflineRecognizerResult(result);
  }
  SherpaOnnxDestroyOfflineStream(stream);

  return text;
}

void AsrEngine::Shutdown() {
  if (initialized_) {
    SherpaOnnxDestroyOfflineRecognizer(recognizer_);
    recognizer_ = nullptr;
    initialized_ = false;
  }
}

bool AsrEngine::IsInitialized() const { return initialized_; }
