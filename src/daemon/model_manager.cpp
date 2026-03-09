#include "model_manager.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

ModelManager::ModelManager(const std::string &base_dir,
                           const std::string &model_name) {
  if (!base_dir.empty()) {
    base_dir_ = base_dir;
  } else {
    const char *xdg = std::getenv("XDG_DATA_HOME");
    if (xdg && xdg[0] != '\0') {
      base_dir_ = std::string(xdg) + "/fcitx5-vinput/models";
    } else {
      const char *home = std::getenv("HOME");
      base_dir_ = std::string(home ? home : "/tmp") +
                  "/.local/share/fcitx5-vinput/models";
    }
  }

  model_name_ = model_name.empty() ? "paraformer-zh" : model_name;
}

bool ModelManager::EnsureModels() {
  auto dir = fs::path(base_dir_) / model_name_;
  auto json_path = dir / "vinput-model.json";

  if (!fs::exists(json_path)) {
    fprintf(stderr,
            "vinput: ASR model configuration not found for '%s' at %s\n"
            "vinput: Missing 'vinput-model.json'. Please ensure you have a "
            "valid model installed.\n",
            model_name_.c_str(), json_path.string().c_str());
    return false;
  }

  auto info = GetModelInfo();
  if (info.model.empty() || info.tokens.empty() || info.model_type.empty()) {
    fprintf(
        stderr,
        "vinput: 'vinput-model.json' for '%s' is missing required fields.\n",
        model_name_.c_str());
    return false;
  }

  if (!fs::exists(info.model)) {
    fprintf(stderr, "vinput: ASR model file not found at %s\n",
            info.model.c_str());
    return false;
  }

  if (!fs::exists(info.tokens)) {
    fprintf(stderr, "vinput: tokens.txt not found at %s\n",
            info.tokens.c_str());
    return false;
  }

  return true;
}

ModelInfo ModelManager::GetModelInfo() const {
  ModelInfo info;
  auto dir = fs::path(base_dir_) / model_name_;
  auto json_path = dir / "vinput-model.json";

  if (!fs::exists(json_path)) {
    return info;
  }

  try {
    std::ifstream file(json_path);
    json j;
    file >> j;

    info.model_type = j.value("model_type", "");
    info.language = j.value("language", "auto");

    if (j.contains("files") && j["files"].is_object()) {
      info.model = (dir / j["files"].value("model", "")).string();
      info.tokens = (dir / j["files"].value("tokens", "")).string();
    }

    if (j.contains("params") && j["params"].is_object()) {
      info.modeling_unit = j["params"].value("modeling_unit", "");
      info.use_itn = j["params"].value("use_itn", false);
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "vinput: failed to parse %s: %s\n",
            json_path.string().c_str(), e.what());
  }

  return info;
}

std::string ModelManager::GetBaseDir() const { return base_dir_; }

std::vector<std::string> ModelManager::ListModels() const {
  std::vector<std::string> models;
  const auto root = fs::path(base_dir_);
  if (!fs::exists(root) || !fs::is_directory(root)) {
    return models;
  }

  for (const auto &entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }

    const auto model_name = entry.path().filename().string();
    if (IsValidModelDir(model_name)) {
      models.push_back(model_name);
    }
  }

  std::sort(models.begin(), models.end());
  return models;
}

std::string ModelManager::GetModelName() const { return model_name_; }

bool ModelManager::IsValidModelDir(const std::string &model_name) const {
  const auto dir = fs::path(base_dir_) / model_name;
  const auto json_path = dir / "vinput-model.json";
  return fs::exists(json_path) && fs::is_regular_file(json_path);
}
