#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace qwen_asr {

// Hotwords configuration - maps word to weight
using HotwordMap = std::map<std::string, float>;

// Hotwords grouped by category
using HotwordsConfig = std::map<std::string, HotwordMap>;

// Streaming callback types
using StreamCallback = std::function<void(const std::string &text, bool is_final)>;
using ErrorCallback = std::function<void(const std::string &error)>;

// Configuration for Qwen3-ASR HTTP client
struct ClientConfig {
  std::string base_url = "http://127.0.0.1:8000";
  int timeout_ms = 30000;
  int connect_timeout_ms = 5000;
  int sample_rate = 16000;
  std::string language;              // Empty for auto-detect
  HotwordsConfig hotwords;           // Inline hotwords
  bool streaming = false;
  int retry_count = 3;
  int retry_delay_ms = 1000;
};

// Recognition result
struct Result {
  std::string text;
  std::string language;
  bool success = false;
  std::string error;
};

// HTTP client for Qwen3-ASR server
class Client {
public:
  Client();
  ~Client();

  // Prevent copy
  Client(const Client &) = delete;
  Client &operator=(const Client &) = delete;

  // Initialize client with configuration
  bool Init(const ClientConfig &config);

  // Non-streaming: transcribe complete audio buffer
  // pcm_data: int16 PCM samples at 16kHz mono
  Result Transcribe(const std::vector<int16_t> &pcm_data);

  // Non-streaming with hotwords override
  Result Transcribe(const std::vector<int16_t> &pcm_data,
                    const HotwordsConfig &hotwords);

  // Non-streaming with language override
  Result Transcribe(const std::vector<int16_t> &pcm_data,
                    const std::string &language);

  // Non-streaming with both overrides
  Result Transcribe(const std::vector<int16_t> &pcm_data,
                    const std::string &language,
                    const HotwordsConfig &hotwords);

  // Streaming: transcribe with real-time callbacks
  // Returns session ID on success, empty string on failure
  std::string StartStreaming(StreamCallback callback,
                              ErrorCallback error_callback = nullptr);

  // Streaming: push audio chunk
  bool PushAudioChunk(const std::string &session_id,
                      const std::vector<int16_t> &pcm_data);

  // Streaming: signal end and get final result
  Result EndStreaming(const std::string &session_id);

  // Load hotwords on server
  bool LoadHotwords(const HotwordsConfig &hotwords, bool merge = false);

  // Clear hotwords on server
  bool ClearHotwords();

  // Check if server is reachable
  bool HealthCheck();

  // Get last error message
  std::string GetLastError() const;

  // Check if initialized
  bool IsInitialized() const { return initialized_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool initialized_ = false;
};

// Utility: convert simple hotwords file to HotwordsConfig
// Format: each line is "word [weight]" (weight defaults to 1.3)
// Lines starting with # are comments
HotwordsConfig ParseHotwordsFile(const std::string &filepath);

// Utility: convert HotwordsConfig to JSON string
std::string HotwordsToJson(const HotwordsConfig &hotwords);

// Utility: encode PCM data to base64 WAV
std::string EncodeAudioBase64(const std::vector<int16_t> &pcm_data,
                               int sample_rate = 16000);

} // namespace qwen_asr
