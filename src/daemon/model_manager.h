#pragma once

#include <string>
#include <vector>

struct ModelInfo {
  std::string model_type; // e.g. "paraformer", "sense_voice", "whisper"
  std::string language;
  std::string model;  // Abs path to model.onnx or model.int8.onnx
  std::string tokens; // Abs path to tokens.txt
  std::string modeling_unit;
  bool use_itn = false;
};

class ModelManager {
public:
  explicit ModelManager(const std::string &base_dir = "",
                        const std::string &model_name = "paraformer-zh");

  bool EnsureModels();
  ModelInfo GetModelInfo() const;
  std::vector<std::string> ListModels() const;
  std::string GetBaseDir() const;
  std::string GetModelName() const;

private:
  bool IsValidModelDir(const std::string &model_name) const;
  std::string base_dir_;
  std::string model_name_;
};
