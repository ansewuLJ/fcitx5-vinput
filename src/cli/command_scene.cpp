#include "cli/command_scene.h"
#include "cli/cli_helpers.h"
#include "common/core_config.h"
#include "common/i18n.h"
#include "common/postprocess_scene.h"
#include "common/string_utils.h"
#include <nlohmann/json.hpp>

int RunSceneList(Formatter &fmt, const CliContext &ctx) {
  CoreConfig config = LoadCoreConfig();
  const auto &scenes = config.scenes.definitions;

  if (ctx.json_output) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &scene : scenes) {
      bool active = (scene.id == config.scenes.activeScene);
      arr.push_back({{"id", scene.id},
                     {"label", vinput::scene::DisplayLabel(scene)},
                     {"active", active}});
    }
    fmt.PrintJson(arr);
    return 0;
  }

  std::vector<std::string> headers = {_("ID"), _("LABEL"), _("STATUS")};
  std::vector<std::vector<std::string>> rows;
  for (const auto &scene : scenes) {
    std::string label = vinput::scene::DisplayLabel(scene);
    std::string status =
        (scene.id == config.scenes.activeScene) ? "[*]" : "[ ]";
    rows.push_back({scene.id, label, status});
  }
  fmt.PrintTable(headers, rows);
  return 0;
}

int RunSceneAdd(const std::string &id, const std::string &label,
                const std::string &prompt, Formatter &fmt,
                const CliContext & /*ctx*/) {
  CoreConfig config = LoadCoreConfig();

  vinput::scene::Definition def;
  def.id = id;
  def.label = label;
  def.prompt = prompt;

  vinput::scene::Config scene_config = ToSceneConfig(config.scenes);
  std::string error;
  if (!vinput::scene::AddScene(&scene_config, def, &error)) {
    fmt.PrintError(error);
    return 1;
  }
  FromSceneConfig(config.scenes, scene_config);

  if (!SaveConfigOrFail(config, fmt)) return 1;

  fmt.PrintSuccess(vinput::str::FmtStr(_("Scene '%s' added."), id));
  return 0;
}

int RunSceneUse(const std::string &id, Formatter &fmt, const CliContext &ctx) {
  (void)ctx;
  CoreConfig config = LoadCoreConfig();

  vinput::scene::Config scene_config = ToSceneConfig(config.scenes);
  std::string error;
  if (!vinput::scene::SetActiveScene(&scene_config, id, &error)) {
    fmt.PrintError(error);
    return 1;
  }
  FromSceneConfig(config.scenes, scene_config);

  if (!SaveConfigOrFail(config, fmt)) return 1;

  fmt.PrintSuccess(vinput::str::FmtStr(_("Default scene set to '%s'."), id));
  return 0;
}

int RunSceneRemove(const std::string &id, bool force, Formatter &fmt,
                   const CliContext &ctx) {
  (void)ctx;
  CoreConfig config = LoadCoreConfig();

  vinput::scene::Config scene_config = ToSceneConfig(config.scenes);
  std::string error;
  if (!vinput::scene::RemoveScene(&scene_config, id, force, &error)) {
    fmt.PrintError(error);
    return 1;
  }
  FromSceneConfig(config.scenes, scene_config);

  if (!SaveConfigOrFail(config, fmt)) return 1;

  fmt.PrintSuccess(vinput::str::FmtStr(_("Scene '%s' removed."), id));
  return 0;
}
