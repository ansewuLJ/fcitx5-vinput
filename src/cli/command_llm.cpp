#include "cli/command_llm.h"

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>

#include "cli/cli_helpers.h"
#include "common/i18n.h"
#include "common/core_config.h"
#include "common/string_utils.h"

static std::string MaskApiKey(const std::string &key) {
  if (key.size() <= 8)
    return std::string(key.size(), '*');
  return key.substr(0, 4) + std::string(key.size() - 8, '*') +
         key.substr(key.size() - 4);
}

int RunLlmList(Formatter &fmt, const CliContext &ctx) {
  auto config = LoadCoreConfig();
  const auto &providers = config.llm.providers;

  if (ctx.json_output) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &p : providers) {
      arr.push_back({{"name", p.name},
                     {"base_url", p.base_url},
                     {"model", p.model},
                     {"api_key", ""}});
    }
    fmt.PrintJson(arr);
    return 0;
  }

  std::vector<std::string> headers = {_("NAME"), _("BASE_URL"), _("MODEL"), _("API_KEY")};
  std::vector<std::vector<std::string>> rows;
  for (const auto &p : providers) {
    rows.push_back({p.name, p.base_url, p.model, MaskApiKey(p.api_key)});
  }
  fmt.PrintTable(headers, rows);
  return 0;
}

int RunLlmAdd(const std::string &name, const std::string &base_url,
              const std::string &model, const std::string &api_key,
              int timeout_ms, Formatter &fmt, const CliContext &ctx) {
  (void)ctx;
  auto config = LoadCoreConfig();
  for (const auto &p : config.llm.providers) {
    if (p.name == name) {
      fmt.PrintError(vinput::str::FmtStr(_("LLM provider '%s' already exists."), name));
      return 1;
    }
  }

  LlmProvider provider;
  provider.name = name;
  provider.base_url = base_url;
  provider.model = model;
  provider.api_key = api_key;
  provider.timeout_ms = timeout_ms;
  config.llm.providers.push_back(provider);

  if (!SaveConfigOrFail(config, fmt)) return 1;
  fmt.PrintSuccess(vinput::str::FmtStr(_("LLM provider '%s' added."), name));
  return 0;
}

int RunLlmRemove(const std::string &name, bool force, Formatter &fmt,
                 const CliContext &ctx) {
  (void)ctx;
  (void)force;
  auto config = LoadCoreConfig();
  auto &providers = config.llm.providers;
  auto it =
      std::find_if(providers.begin(), providers.end(),
                   [&name](const LlmProvider &p) { return p.name == name; });
  if (it == providers.end()) {
    fmt.PrintError(vinput::str::FmtStr(_("LLM provider '%s' not found."), name));
    return 1;
  }
  providers.erase(it);
  if (!SaveConfigOrFail(config, fmt)) return 1;
  fmt.PrintSuccess(vinput::str::FmtStr(_("LLM provider '%s' removed."), name));
  return 0;
}
