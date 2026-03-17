#include "vad_trimmer.h"

#include <sherpa-onnx/c-api/c-api.h>

#include <cstdio>

VadTrimmer::VadTrimmer() = default;

VadTrimmer::~VadTrimmer() { Shutdown(); }

bool VadTrimmer::Init(const std::string &model_path, int sample_rate) {
  if (vad_) return true;

  SherpaOnnxVadModelConfig config = {};
  config.silero_vad.model = model_path.c_str();
  config.silero_vad.threshold = 0.5f;
  config.silero_vad.min_silence_duration = 0.5f;
  config.silero_vad.min_speech_duration = 0.25f;
  config.silero_vad.window_size = 512;
  config.silero_vad.max_speech_duration = 0.0f;
  config.sample_rate = sample_rate;
  config.num_threads = 1;
  config.provider = "cpu";
  config.debug = 0;

  vad_ = SherpaOnnxCreateVoiceActivityDetector(&config, 30.0f);
  if (!vad_) {
    fprintf(stderr, "vinput: failed to create VAD from '%s'\n",
            model_path.c_str());
    return false;
  }

  sample_rate_ = sample_rate;
  fprintf(stderr, "vinput: VAD initialized from '%s'\n", model_path.c_str());
  return true;
}

std::vector<float> VadTrimmer::Trim(const std::vector<float> &samples,
                                    int /*sample_rate*/) {
  if (!vad_ || samples.empty()) return samples;

  SherpaOnnxVoiceActivityDetectorReset(vad_);

  // Feed audio in window_size chunks
  const int window_size = 512;
  const int n = static_cast<int>(samples.size());
  for (int offset = 0; offset + window_size <= n; offset += window_size) {
    SherpaOnnxVoiceActivityDetectorAcceptWaveform(vad_, samples.data() + offset,
                                                  window_size);
  }
  SherpaOnnxVoiceActivityDetectorFlush(vad_);

  // Collect all speech segments
  std::vector<float> result;
  while (!SherpaOnnxVoiceActivityDetectorEmpty(vad_)) {
    const SherpaOnnxSpeechSegment *seg =
        SherpaOnnxVoiceActivityDetectorFront(vad_);
    if (seg && seg->samples && seg->n > 0) {
      result.insert(result.end(), seg->samples, seg->samples + seg->n);
    }
    if (seg) {
      SherpaOnnxDestroySpeechSegment(seg);
    }
    SherpaOnnxVoiceActivityDetectorPop(vad_);
  }

  if (result.empty()) {
    fprintf(stderr, "vinput: VAD found no speech, returning original audio\n");
    return samples;
  }

  fprintf(stderr, "vinput: VAD trimmed %d -> %zu samples\n", n, result.size());
  return result;
}

bool VadTrimmer::Available() const { return vad_ != nullptr; }

void VadTrimmer::Shutdown() {
  if (vad_) {
    SherpaOnnxDestroyVoiceActivityDetector(vad_);
    vad_ = nullptr;
  }
}
