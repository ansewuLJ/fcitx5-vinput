#include "cli/command_device.h"
#include "cli/cli_helpers.h"
#include "common/core_config.h"
#include "common/i18n.h"
#include "common/pipewire_device.h"
#include "common/string_utils.h"

#include <nlohmann/json.hpp>
#include <vector>

int RunDeviceList(Formatter &fmt, const CliContext &ctx) {
  CoreConfig config = LoadCoreConfig();
  std::string active_device = config.captureDevice;

  auto devices = vinput::pw::EnumerateAudioSources();
  if (devices.empty()) {
    fmt.PrintInfo(
        _("No audio capture devices found or PipeWire is not running."));
    return 0;
  }

  if (ctx.json_output) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &d : devices) {
      bool is_active = (d.name == active_device) ||
                       (active_device == "default" && d.name == "default");
      arr.push_back({{"id", d.id},
                     {"name", d.name},
                     {"description", d.description},
                     {"active", is_active}});
    }
    fmt.PrintJson(arr);
    return 0;
  }

  std::vector<std::string> headers = {_("NAME"), _("DESCRIPTION"), _("STATUS")};
  std::vector<std::vector<std::string>> rows;

  {
    std::string status = (active_device == "default" || active_device.empty())
                             ? std::string("[*] ") + _("Active")
                             : "[ ]";
    rows.push_back({"default", _("System Default"), status});
  }

  for (const auto &d : devices) {
    std::string status =
        (active_device == d.name) ? std::string("[*] ") + _("Active") : "[ ]";
    rows.push_back({d.name, d.description, status});
  }

  fmt.PrintTable(headers, rows);
  return 0;
}

int RunDeviceUse(const std::string &name, Formatter &fmt,
                 const CliContext &ctx) {
  (void)ctx;
  CoreConfig config = LoadCoreConfig();

  if (name != "default") {
    auto devices = vinput::pw::EnumerateAudioSources();
    bool found = false;
    for (const auto &d : devices) {
      if (d.name == name) {
        found = true;
        break;
      }
    }
    if (!found) {
      fmt.PrintWarning(vinput::str::FmtStr(
          _("Device '%s' not found in PipeWire. Setting it anyway."), name));
    }
  }

  config.captureDevice = name;

  if (!SaveConfigOrFail(config, fmt)) return 1;

  fmt.PrintSuccess(vinput::str::FmtStr(
      _("Capture device set to '%s'. Restart daemon to apply changes."), name));
  return 0;
}
