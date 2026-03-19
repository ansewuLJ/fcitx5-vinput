#include "qwen_asr_client.h"

#include <cstring>
#include <fstream>
#include <sstream>

// Use libcurl for HTTP requests
#include <curl/curl.h>

// Use nlohmann/json for JSON handling
#include <nlohmann/json.hpp>

namespace qwen_asr {

namespace {

// Base64 encoding table
constexpr char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string Base64Encode(const std::vector<uint8_t> &data) {
  std::string result;
  result.reserve(((data.size() + 2) / 3) * 4);

  size_t i = 0;
  while (i < data.size()) {
    uint32_t octet_a = i < data.size() ? data[i++] : 0;
    uint32_t octet_b = i < data.size() ? data[i++] : 0;
    uint32_t octet_c = i < data.size() ? data[i++] : 0;

    uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

    result += kBase64Table[(triple >> 18) & 0x3F];
    result += kBase64Table[(triple >> 12) & 0x3F];
    result += kBase64Table[(triple >> 6) & 0x3F];
    result += kBase64Table[triple & 0x3F];
  }

  // Add padding
  size_t mod = data.size() % 3;
  if (mod == 1) {
    result[result.size() - 1] = '=';
    result[result.size() - 2] = '=';
  } else if (mod == 2) {
    result[result.size() - 1] = '=';
  }

  return result;
}

// CURL write callback
size_t WriteCallback(void *contents, size_t size, size_t nmemb,
                     std::string *userp) {
  size_t total_size = size * nmemb;
  userp->append(static_cast<char *>(contents), total_size);
  return total_size;
}

// Create WAV header for 16-bit mono PCM
std::vector<uint8_t> CreateWavHeader(size_t data_size, int sample_rate) {
  std::vector<uint8_t> header(44);

  // RIFF header
  std::memcpy(&header[0], "RIFF", 4);
  uint32_t file_size = static_cast<uint32_t>(data_size + 36);
  std::memcpy(&header[4], &file_size, 4);
  std::memcpy(&header[8], "WAVE", 4);

  // fmt chunk
  std::memcpy(&header[12], "fmt ", 4);
  uint32_t fmt_size = 16;
  std::memcpy(&header[16], &fmt_size, 4);
  uint16_t audio_format = 1; // PCM
  std::memcpy(&header[20], &audio_format, 2);
  uint16_t num_channels = 1;
  std::memcpy(&header[22], &num_channels, 2);
  uint32_t byte_rate = sample_rate * 2; // 16-bit mono
  std::memcpy(&header[24], &sample_rate, 4);
  std::memcpy(&header[28], &byte_rate, 4);
  uint16_t block_align = 2;
  std::memcpy(&header[32], &block_align, 2);
  uint16_t bits_per_sample = 16;
  std::memcpy(&header[34], &bits_per_sample, 2);

  // data chunk
  std::memcpy(&header[36], "data", 4);
  uint32_t data_size_32 = static_cast<uint32_t>(data_size);
  std::memcpy(&header[40], &data_size_32, 4);

  return header;
}

} // namespace

// Implementation details
struct Client::Impl {
  ClientConfig config;
  std::string last_error;
  CURL *curl = nullptr;
  struct curl_slist *headers = nullptr;

  Impl() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl) {
      headers = curl_slist_append(headers, "Content-Type: application/json");
      headers = curl_slist_append(headers, "Accept: application/json");
    }
  }

  ~Impl() {
    if (headers) {
      curl_slist_free_all(headers);
    }
    if (curl) {
      curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
  }

  bool DoPost(const std::string &endpoint, const std::string &body,
              std::string *response) {
    if (!curl) {
      last_error = "CURL not initialized";
      return false;
    }

    // Reset curl handle
    curl_easy_reset(curl);

    // Set URL
    std::string url = config.base_url + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    // Set headers
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Set POST body
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());

    // Set callbacks
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);

    // Set timeouts
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config.timeout_ms);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS,
                     config.connect_timeout_ms);

    // Perform request
    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
      last_error = curl_easy_strerror(res);
      return false;
    }

    // Check HTTP status code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code >= 400) {
      last_error = "HTTP error: " + std::to_string(http_code);
      return false;
    }

    return true;
  }

  bool DoGet(const std::string &endpoint, std::string *response) {
    if (!curl) {
      last_error = "CURL not initialized";
      return false;
    }

    curl_easy_reset(curl);

    std::string url = config.base_url + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config.timeout_ms);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS,
                     config.connect_timeout_ms);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
      last_error = curl_easy_strerror(res);
      return false;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code >= 400) {
      last_error = "HTTP error: " + std::to_string(http_code);
      return false;
    }

    return true;
  }

  bool DoDelete(const std::string &endpoint, std::string *response) {
    if (!curl) {
      last_error = "CURL not initialized";
      return false;
    }

    curl_easy_reset(curl);

    std::string url = config.base_url + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config.timeout_ms);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS,
                     config.connect_timeout_ms);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
      last_error = curl_easy_strerror(res);
      return false;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code >= 400) {
      last_error = "HTTP error: " + std::to_string(http_code);
      return false;
    }

    return true;
  }
};

Client::Client() : impl_(std::make_unique<Impl>()) {}

Client::~Client() = default;

bool Client::Init(const ClientConfig &config) {
  impl_->config = config;
  initialized_ = true;
  return true;
}

Result Client::Transcribe(const std::vector<int16_t> &pcm_data) {
  return Transcribe(pcm_data, impl_->config.language, impl_->config.hotwords);
}

Result Client::Transcribe(const std::vector<int16_t> &pcm_data,
                           const HotwordsConfig &hotwords) {
  return Transcribe(pcm_data, impl_->config.language, hotwords);
}

Result Client::Transcribe(const std::vector<int16_t> &pcm_data,
                           const std::string &language) {
  return Transcribe(pcm_data, language, impl_->config.hotwords);
}

Result Client::Transcribe(const std::vector<int16_t> &pcm_data,
                           const std::string &language,
                           const HotwordsConfig &hotwords) {
  Result result;

  if (!initialized_) {
    result.error = "Client not initialized";
    return result;
  }

  if (pcm_data.empty()) {
    result.error = "Empty audio data";
    return result;
  }

  // Encode audio to base64 WAV
  std::string audio_base64 = EncodeAudioBase64(pcm_data);

  // Build JSON request
  nlohmann::json request_json;
  request_json["audio_base64"] = audio_base64;

  if (!language.empty()) {
    request_json["language"] = language;
  }

  if (!hotwords.empty()) {
    request_json["hotwords"] = hotwords;
  }

  // Send request
  std::string response;
  if (!impl_->DoPost("/transcribe", request_json.dump(), &response)) {
    result.error = impl_->last_error;
    return result;
  }

  // Parse response
  try {
    nlohmann::json response_json = nlohmann::json::parse(response);
    result.text = response_json.value("text", "");
    result.language = response_json.value("language", "");
    result.success = response_json.value("success", false);

    if (!result.success) {
      result.error = response_json.value("error", "Unknown error");
    }
  } catch (const nlohmann::json::exception &e) {
    result.error = std::string("JSON parse error: ") + e.what();
  }

  return result;
}

std::string Client::StartStreaming(StreamCallback callback,
                                    ErrorCallback error_callback) {
  // TODO: Implement WebSocket streaming
  // For now, return empty string (streaming not implemented)
  impl_->last_error = "Streaming not yet implemented";
  return "";
}

bool Client::PushAudioChunk(const std::string &session_id,
                             const std::vector<int16_t> &pcm_data) {
  // TODO: Implement WebSocket streaming
  return false;
}

Result Client::EndStreaming(const std::string &session_id) {
  // TODO: Implement WebSocket streaming
  Result result;
  result.error = "Streaming not yet implemented";
  return result;
}

bool Client::LoadHotwords(const HotwordsConfig &hotwords, bool merge) {
  if (!initialized_) {
    return false;
  }

  nlohmann::json request_json;
  request_json["hotwords"] = hotwords;
  request_json["merge"] = merge;

  std::string response;
  if (!impl_->DoPost("/hotwords/load", request_json.dump(), &response)) {
    return false;
  }

  try {
    nlohmann::json response_json = nlohmann::json::parse(response);
    return response_json.value("success", false);
  } catch (...) {
    return false;
  }
}

bool Client::ClearHotwords() {
  if (!initialized_) {
    return false;
  }

  std::string response;
  if (!impl_->DoDelete("/hotwords", &response)) {
    return false;
  }

  return true;
}

bool Client::HealthCheck() {
  if (!initialized_) {
    return false;
  }

  std::string response;
  if (!impl_->DoGet("/health", &response)) {
    return false;
  }

  try {
    nlohmann::json response_json = nlohmann::json::parse(response);
    return response_json.value("status", "") == "healthy";
  } catch (...) {
    return false;
  }
}

std::string Client::GetLastError() const { return impl_->last_error; }

// Utility functions implementation
HotwordsConfig ParseHotwordsFile(const std::string &filepath) {
  HotwordsConfig config;
  HotwordMap &default_category = config["default"];

  std::ifstream file(filepath);
  if (!file.is_open()) {
    return config;
  }

  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse line: "word [weight]"
    std::istringstream iss(line);
    std::string word;
    float weight = 1.3f; // Default weight

    if (iss >> word) {
      if (iss >> weight) {
        // Weight specified
      }
      default_category[word] = weight;
    }
  }

  return config;
}

std::string HotwordsToJson(const HotwordsConfig &hotwords) {
  nlohmann::json j = hotwords;
  return j.dump();
}

std::string EncodeAudioBase64(const std::vector<int16_t> &pcm_data,
                               int sample_rate) {
  // Create WAV file in memory
  size_t data_size = pcm_data.size() * sizeof(int16_t);
  auto header = CreateWavHeader(data_size, sample_rate);

  // Combine header and data
  std::vector<uint8_t> wav_data;
  wav_data.reserve(header.size() + data_size);
  wav_data.insert(wav_data.end(), header.begin(), header.end());

  const uint8_t *pcm_bytes = reinterpret_cast<const uint8_t *>(pcm_data.data());
  wav_data.insert(wav_data.end(), pcm_bytes, pcm_bytes + data_size);

  // Encode to base64
  return Base64Encode(wav_data);
}

} // namespace qwen_asr
