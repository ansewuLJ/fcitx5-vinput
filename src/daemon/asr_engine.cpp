#include "asr_engine.h"
#include "audio_utils.h"

#include <sherpa-onnx/c-api/c-api.h>

#include <cstdio>
#include <cstdlib>

namespace {

float SafeStof(const std::string &s, float default_val) {
  try { return std::stof(s); } catch (...) { return default_val; }
}

int SafeStoi(const std::string &s, int default_val) {
  try { return std::stoi(s); } catch (...) { return default_val; }
}

} // namespace

// =============================================================================
// SherpaOnnxBackend Implementation
// =============================================================================

SherpaOnnxBackend::SherpaOnnxBackend(const ModelInfo &model_info)
    : model_info_(model_info) {}

SherpaOnnxBackend::~SherpaOnnxBackend() { Shutdown(); }

bool SherpaOnnxBackend::Init(const AsrConfig &asr_config) {
  if (initialized_) {
    return true;
  }

  SherpaOnnxOfflineRecognizerConfig config = {};
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;

  // Decoding configs from model params (repo can preset optimal values)
  const std::string p_decoding_method =
      model_info_.Param("decoding_method", "greedy_search");
  config.decoding_method = p_decoding_method.c_str();
  config.max_active_paths =
      SafeStoi(model_info_.Param("max_active_paths", "4"), 4);
  config.blank_penalty =
      SafeStof(model_info_.Param("blank_penalty", "0.0"), 0.0f);

  // Stash file paths to keep c_str() pointers alive through recognizer creation
  const std::string tokens_path = model_info_.File("tokens");
  const std::string f_model = model_info_.File("model");
  const std::string f_encoder = model_info_.File("encoder");
  const std::string f_decoder = model_info_.File("decoder");
  const std::string f_joiner = model_info_.File("joiner");
  const std::string f_preprocessor = model_info_.File("preprocessor");
  const std::string f_uncached_decoder = model_info_.File("uncached_decoder");
  const std::string f_cached_decoder = model_info_.File("cached_decoder");
  const std::string f_merged_decoder = model_info_.File("merged_decoder");
  const std::string f_encoder_adaptor = model_info_.File("encoder_adaptor");
  const std::string f_llm = model_info_.File("llm");
  const std::string f_embedding = model_info_.File("embedding");
  const std::string f_tokenizer = model_info_.File("tokenizer");
  const std::string f_lm = model_info_.File("lm");
  const std::string f_hotwords_file = model_info_.File("hotwords_file");
  const std::string f_bpe_vocab = model_info_.File("bpe_vocab");
  const std::string f_rule_fsts = model_info_.File("rule_fsts");
  const std::string f_rule_fars = model_info_.File("rule_fars");
  const std::string p_language = asr_config.language;
  const std::string p_modeling_unit = model_info_.Param("modeling_unit", "cjkchar");
  const std::string p_tgt_lang = model_info_.Param("tgt_lang", asr_config.language);
  const std::string p_system_prompt = model_info_.Param("system_prompt");
  const std::string p_user_prompt = model_info_.Param("user_prompt");
  const std::string p_hotwords = model_info_.Param("hotwords");

  config.model_config.tokens = tokens_path.c_str();
  config.model_config.num_threads = asr_config.thread_num;
  config.model_config.provider = "cpu";

  // Optional general model config fields from params
  if (!f_bpe_vocab.empty()) {
    config.model_config.bpe_vocab = f_bpe_vocab.c_str();
  }

  // Optional LM config
  if (!f_lm.empty()) {
    config.lm_config.model = f_lm.c_str();
    config.lm_config.scale =
        SafeStof(model_info_.Param("lm_scale", "0.5"), 0.5f);
  }
  const auto &type = model_info_.model_type;

  // Only zipformer_transducer supports hotwords_file with modified_beam_search
  const bool type_supports_hotwords = (type == "zipformer_transducer");
  if (type_supports_hotwords) {
    if (!asr_config.hotwords_file.empty()) {
      config.hotwords_file = asr_config.hotwords_file.c_str();
      config.decoding_method = "modified_beam_search";
    } else if (!f_hotwords_file.empty()) {
      config.hotwords_file = f_hotwords_file.c_str();
      config.decoding_method = "modified_beam_search";
    }
  }

  // Optional rule FSTs/FARs
  if (!f_rule_fsts.empty()) {
    config.rule_fsts = f_rule_fsts.c_str();
  }
  if (!f_rule_fars.empty()) {
    config.rule_fars = f_rule_fars.c_str();
  }

  if (type == "paraformer") {
    config.model_config.paraformer.model = f_model.c_str();
    config.model_config.model_type = "paraformer";
    config.model_config.modeling_unit = p_modeling_unit.c_str();

  } else if (type == "sense_voice") {
    config.model_config.sense_voice.model = f_model.c_str();
    config.model_config.sense_voice.language = p_language.c_str();
    config.model_config.sense_voice.use_itn =
        model_info_.ParamBool("use_itn") ? 1 : 0;
    config.model_config.model_type = "sense_voice";

  } else if (type == "whisper") {
    config.model_config.whisper.encoder = f_encoder.c_str();
    config.model_config.whisper.decoder = f_decoder.c_str();
    config.model_config.whisper.language = p_language.c_str();
    config.model_config.whisper.task = "transcribe";
    config.model_config.whisper.tail_paddings =
        SafeStoi(model_info_.Param("tail_paddings", "-1"), -1);
    config.model_config.whisper.enable_token_timestamps =
        model_info_.ParamBool("enable_token_timestamps") ? 1 : 0;
    config.model_config.whisper.enable_segment_timestamps =
        model_info_.ParamBool("enable_segment_timestamps") ? 1 : 0;
    config.model_config.model_type = "whisper";

  } else if (type == "moonshine") {
    config.model_config.moonshine.preprocessor = f_preprocessor.c_str();
    config.model_config.moonshine.encoder = f_encoder.c_str();
    config.model_config.moonshine.uncached_decoder =
        f_uncached_decoder.c_str();
    config.model_config.moonshine.cached_decoder = f_cached_decoder.c_str();
    if (!f_merged_decoder.empty()) {
      config.model_config.moonshine.merged_decoder = f_merged_decoder.c_str();
    }
    config.model_config.model_type = "moonshine";

  } else if (type == "zipformer_transducer") {
    config.model_config.transducer.encoder = f_encoder.c_str();
    config.model_config.transducer.decoder = f_decoder.c_str();
    config.model_config.transducer.joiner = f_joiner.c_str();
    config.model_config.model_type = "transducer";

  } else if (type == "zipformer_ctc") {
    config.model_config.zipformer_ctc.model = f_model.c_str();
    config.model_config.model_type = "zipformer_ctc";

  } else if (type == "fire_red_asr") {
    if (!f_encoder.empty()) {
      config.model_config.fire_red_asr.encoder = f_encoder.c_str();
      config.model_config.fire_red_asr.decoder = f_decoder.c_str();
      config.model_config.model_type = "fire_red_asr";
    } else {
      config.model_config.fire_red_asr_ctc.model = f_model.c_str();
      config.model_config.model_type = "fire_red_asr_ctc";
    }

  } else if (type == "dolphin") {
    config.model_config.dolphin.model = f_model.c_str();
    config.model_config.model_type = "dolphin";

  } else if (type == "nemo_ctc") {
    config.model_config.nemo_ctc.model = f_model.c_str();
    config.model_config.model_type = "nemo_ctc";

  } else if (type == "wenet_ctc") {
    config.model_config.wenet_ctc.model = f_model.c_str();
    config.model_config.model_type = "wenet_ctc";

  } else if (type == "tdnn") {
    config.model_config.tdnn.model = f_model.c_str();
    config.model_config.model_type = "tdnn";

  } else if (type == "telespeech_ctc") {
    config.model_config.telespeech_ctc = f_model.c_str();
    config.model_config.model_type = "telespeech_ctc";

  } else if (type == "omnilingual") {
    config.model_config.omnilingual.model = f_model.c_str();
    config.model_config.model_type = "omnilingual";

  } else if (type == "medasr") {
    config.model_config.medasr.model = f_model.c_str();
    config.model_config.model_type = "medasr";

  } else if (type == "canary") {
    config.model_config.canary.encoder = f_encoder.c_str();
    config.model_config.canary.decoder = f_decoder.c_str();
    config.model_config.canary.src_lang = p_language.c_str();
    config.model_config.canary.tgt_lang = p_tgt_lang.c_str();
    config.model_config.canary.use_pnc = model_info_.ParamBool("use_pnc") ? 1 : 0;
    config.model_config.model_type = "canary";

  } else if (type == "funasr_nano") {
    config.model_config.funasr_nano.encoder_adaptor =
        f_encoder_adaptor.c_str();
    config.model_config.funasr_nano.llm = f_llm.c_str();
    config.model_config.funasr_nano.embedding = f_embedding.c_str();
    config.model_config.funasr_nano.tokenizer = f_tokenizer.c_str();
    config.model_config.funasr_nano.language = p_language.c_str();
    config.model_config.funasr_nano.itn =
        model_info_.ParamBool("use_itn") ? 1 : 0;
    if (!p_system_prompt.empty()) {
      config.model_config.funasr_nano.system_prompt =
          p_system_prompt.c_str();
    }
    if (!p_user_prompt.empty()) {
      config.model_config.funasr_nano.user_prompt = p_user_prompt.c_str();
    }
    if (!p_hotwords.empty()) {
      config.model_config.funasr_nano.hotwords = p_hotwords.c_str();
    }
    config.model_config.funasr_nano.max_new_tokens =
        SafeStoi(model_info_.Param("max_new_tokens", "1024"), 1024);
    config.model_config.funasr_nano.temperature =
        SafeStof(model_info_.Param("temperature", "1.0"), 1.0f);
    config.model_config.funasr_nano.top_p =
        SafeStof(model_info_.Param("top_p", "0.9"), 0.9f);
    config.model_config.funasr_nano.seed =
        SafeStoi(model_info_.Param("seed", "0"), 0);
    config.model_config.model_type = "funasr_nano";

  } else {
    fprintf(stderr, "vinput: unsupported model type '%s'\n", type.c_str());
    return false;
  }

  recognizer_ = SherpaOnnxCreateOfflineRecognizer(&config);
  if (!recognizer_) {
    fprintf(stderr,
            "vinput: failed to create sherpa-onnx recognizer for type '%s'\n",
            type.c_str());
    return false;
  }

  initialized_ = true;
  normalize_audio_ = asr_config.normalize_audio;

  if (asr_config.vad_enabled && !asr_config.vad_model_path.empty()) {
    if (!vad_.Init(asr_config.vad_model_path)) {
      fprintf(stderr, "vinput: VAD model not available, continuing without VAD\n");
    }
  }

  fprintf(
      stderr,
      "vinput: sherpa-onnx ASR initialized successfully (type: %s, lang: %s)\n",
      type.c_str(), asr_config.language.c_str());
  return true;
}

std::string SherpaOnnxBackend::Infer(const std::vector<int16_t> &pcm_data) {
  if (!initialized_ || pcm_data.empty()) {
    return "";
  }

  if (pcm_data.size() < AsrEngine::kMinSamplesForInference) {
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

  // Phase 2: peak normalization for low-volume recordings
  if (normalize_audio_) {
    vinput::audio::PeakNormalize(samples);
  }

  // Phase 3: VAD silence trimming
  if (vad_.Available()) {
    samples = vad_.Trim(samples, 16000);
    if (samples.size() < AsrEngine::kMinSamplesForInference) {
      fprintf(stderr, "vinput: audio too short after VAD trim, skipping\n");
      return "";
    }
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

void SherpaOnnxBackend::Shutdown() {
  if (initialized_) {
    SherpaOnnxDestroyOfflineRecognizer(recognizer_);
    recognizer_ = nullptr;
    initialized_ = false;
  }
}

// =============================================================================
// QwenHttpBackend Implementation
// =============================================================================

QwenHttpBackend::QwenHttpBackend(const AsrBackend::QwenHttp &config)
    : http_config_(config) {}

bool QwenHttpBackend::Init(const AsrConfig &config) {
  if (initialized_) {
    return true;
  }

  qwen_asr::ClientConfig client_config;
  client_config.base_url = http_config_.url;
  client_config.timeout_ms = http_config_.timeoutMs;
  client_config.streaming = http_config_.streaming;
  client_config.language = config.language;

  client_ = std::make_unique<qwen_asr::Client>();
  if (!client_->Init(client_config)) {
    fprintf(stderr, "vinput: failed to initialize Qwen HTTP client\n");
    return false;
  }

  // Check server health
  if (!client_->HealthCheck()) {
    fprintf(stderr, "vinput: Qwen ASR server not reachable at %s\n",
            http_config_.url.c_str());
    return false;
  }

  language_ = config.language;
  initialized_ = true;

  fprintf(stderr, "vinput: Qwen HTTP ASR initialized (url: %s, lang: %s)\n",
          http_config_.url.c_str(), config.language.c_str());
  return true;
}

std::string QwenHttpBackend::Infer(const std::vector<int16_t> &pcm_data) {
  if (!initialized_ || !client_ || pcm_data.empty()) {
    return "";
  }

  if (pcm_data.size() < AsrEngine::kMinSamplesForInference) {
    fprintf(stderr,
            "vinput: skipping ASR for short audio: %zu samples (%.1f ms)\n",
            pcm_data.size(),
            static_cast<double>(pcm_data.size()) * 1000.0 / 16000.0);
    return "";
  }

  auto result = client_->Transcribe(pcm_data, language_);

  if (!result.success) {
    fprintf(stderr, "vinput: Qwen HTTP transcription failed: %s\n",
            result.error.c_str());
    return "";
  }

  return result.text;
}

void QwenHttpBackend::Shutdown() {
  client_.reset();
  initialized_ = false;
}

bool QwenHttpBackend::LoadHotwords(
    const std::map<std::string, std::map<std::string, float>> &hotwords) {
  if (!initialized_ || !client_) {
    return false;
  }
  return client_->LoadHotwords(hotwords, false);
}

// =============================================================================
// AsrEngine Implementation (Wrapper)
// =============================================================================

AsrEngine::AsrEngine() = default;

AsrEngine::~AsrEngine() { Shutdown(); }

bool AsrEngine::Init(const ModelInfo &info, const AsrConfig &asr_config) {
  backend_ = std::make_unique<SherpaOnnxBackend>(info);
  if (!backend_->Init(asr_config)) {
    backend_.reset();
    return false;
  }
  initialized_ = true;
  return true;
}

bool AsrEngine::Init(const AsrBackend::QwenHttp &http_config,
                     const AsrConfig &asr_config) {
  backend_ = std::make_unique<QwenHttpBackend>(http_config);
  if (!backend_->Init(asr_config)) {
    backend_.reset();
    return false;
  }
  initialized_ = true;
  return true;
}

bool AsrEngine::Init(const CoreConfig &core_config, const AsrConfig &asr_config) {
  if (core_config.asrBackend.IsQwenHttp()) {
    return Init(core_config.asrBackend.qwenHttp, asr_config);
  } else {
    // Local backend - need ModelInfo
    ModelManager model_mgr(
        ResolveModelBaseDir(core_config).string(),
        core_config.activeModel
    );
    if (!model_mgr.EnsureModels()) {
      fprintf(stderr, "vinput: model check failed\n");
      return false;
    }
    return Init(model_mgr.GetModelInfo(), asr_config);
  }
}

std::string AsrEngine::Infer(const std::vector<int16_t> &pcm_data) {
  if (!initialized_ || !backend_) {
    return "";
  }
  return backend_->Infer(pcm_data);
}

void AsrEngine::Shutdown() {
  if (backend_) {
    backend_->Shutdown();
    backend_.reset();
  }
  initialized_ = false;
}

bool AsrEngine::IsInitialized() const { return initialized_; }

std::string AsrEngine::BackendName() const {
  if (backend_) {
    return backend_->Name();
  }
  return "none";
}