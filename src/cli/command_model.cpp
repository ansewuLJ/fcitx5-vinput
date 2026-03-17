#include "cli/command_model.h"
#include "cli/cli_helpers.h"
#include "cli/systemd_client.h"
#include "common/i18n.h"
#include "cli/progress.h"
#include "common/core_config.h"
#include "common/model_manager.h"
#include "common/model_repository.h"
#include "common/string_utils.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

int RunModelList(bool remote, Formatter& fmt, const CliContext& ctx) {
    auto config = LoadCoreConfig();
    NormalizeCoreConfig(&config);
    auto base_dir = ResolveModelBaseDir(config);
    ModelManager mgr(base_dir.string());

    if (!remote) {
        auto models = mgr.ListDetailed(config.activeModel);

        if (ctx.json_output) {
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& m : models) {
                std::string state_str;
                if (m.state == ModelState::Active) state_str = "active";
                else if (m.state == ModelState::Broken) state_str = "broken";
                else state_str = "installed";
                uint64_t size = m.size_bytes;
                arr.push_back({
                    {"name", m.name},
                    {"model_type", m.model_type},
                    {"language", m.language},
                    {"status", state_str},
                    {"supports_hotwords", m.supports_hotwords},
                    {"size_bytes", size},
                    {"size", vinput::str::FormatSize(size)}
                });
            }
            fmt.PrintJson(arr);
            return 0;
        }

        std::vector<std::string> headers = {_("NAME"), _("TYPE"), _("LANGUAGE"), _("SIZE"), _("HOTWORDS"), _("STATUS")};
        std::vector<std::vector<std::string>> rows;
        for (const auto& m : models) {
            std::string status_str;
            if (m.state == ModelState::Active) {
                status_str = std::string("[*] ") + _("Active");
            } else if (m.state == ModelState::Broken) {
                status_str = std::string("[!] ") + _("Broken");
            } else {
                status_str = std::string("[ ] ") + _("Installed");
            }
            std::string hotwords = m.supports_hotwords ? _("yes") : _("no");
            rows.push_back({m.name, m.model_type, m.language, vinput::str::FormatSize(m.size_bytes), hotwords, status_str});
        }
        fmt.PrintTable(headers, rows);
        return 0;
    }

    // Remote listing
    if (config.registryUrl.empty()) {
        fmt.PrintError(_("No registry URL configured. Set with: vinput config set extra.registry_url <url>"));
        return 1;
    }

    ModelRepository repo(base_dir.string());
    std::string err;
    auto remote_models = repo.FetchRegistry(config.registryUrl, &err);
    if (!err.empty()) {
        fmt.PrintError(err);
        return 1;
    }

    // Get local model names for comparison
    auto local_models = mgr.ListDetailed(config.activeModel);
    auto is_installed = [&](const std::string& name) {
        for (const auto& lm : local_models) {
            if (lm.name == name) return true;
        }
        return false;
    };

    if (ctx.json_output) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& m : remote_models) {
            arr.push_back({
                {"name", m.name},
                {"display_name", m.display_name},
                {"model_type", m.model_type()},
                {"language", m.language()},
                {"size", vinput::str::FormatSize(m.size_bytes())},
                {"size_bytes", m.size_bytes()},
                {"status", is_installed(m.name) ? "installed" : "available"},
                {"supports_hotwords", m.supports_hotwords()},
                {"description", m.description}
            });
        }
        fmt.PrintJson(arr);
        return 0;
    }

    std::vector<std::string> headers = {_("NAME"), _("TYPE"), _("LANGUAGE"), _("SIZE"), _("HOTWORDS"), _("STATUS")};
    std::vector<std::vector<std::string>> rows;
    for (const auto& m : remote_models) {
        std::string status = is_installed(m.name) ? _("installed") : _("available");
        std::string hotwords = m.supports_hotwords() ? _("yes") : _("no");
        rows.push_back({m.name, m.model_type(), m.language(), vinput::str::FormatSize(m.size_bytes()), hotwords, status});
    }
    fmt.PrintTable(headers, rows);
    return 0;
}

int RunModelAdd(const std::string& name, Formatter& fmt, const CliContext& ctx) {
    auto config = LoadCoreConfig();
    NormalizeCoreConfig(&config);
    auto base_dir = ResolveModelBaseDir(config);

    if (config.registryUrl.empty()) {
        fmt.PrintError(_("No registry URL configured. Set with: vinput config set extra.registry_url <url>"));
        return 1;
    }

    ModelRepository repo(base_dir.string());

    // Fetch registry first to get total size for progress bar
    std::string err;
    auto remote_models = repo.FetchRegistry(config.registryUrl, &err);
    if (!err.empty()) {
        fmt.PrintError(err);
        return 1;
    }

    uint64_t total_size = 0;
    for (const auto& m : remote_models) {
        if (m.name == name) {
            total_size = m.size_bytes();
            break;
        }
    }

    char label_buf[256];
    snprintf(label_buf, sizeof(label_buf), _("Downloading %s..."), name.c_str());
    ProgressBar bar(label_buf, total_size, ctx.is_tty);

    bool install_ok = repo.InstallModel(
        config.registryUrl, name,
        [&](const InstallProgress& p) {
            bar.Update(p.downloaded_bytes, p.speed_bps);
        },
        &err);

    bar.Finish();

    if (!install_ok) {
        fmt.PrintError(err);
        return 1;
    }

    char success_buf[256];
    snprintf(success_buf, sizeof(success_buf), _("Model '%s' installed successfully."), name.c_str());
    fmt.PrintSuccess(success_buf);
    fmt.PrintInfo(vinput::str::FmtStr(_("Run `vinput model use %s` to activate"), name));
    return 0;
}

int RunModelUse(const std::string& name, Formatter& fmt, const CliContext& /*ctx*/) {
    auto config = LoadCoreConfig();
    NormalizeCoreConfig(&config);
    auto base_dir = ResolveModelBaseDir(config);

    ModelManager mgr(base_dir.string());
    std::string err;
    if (!mgr.Validate(name, &err)) {
        fmt.PrintError(vinput::str::FmtStr(_("Model '%s' is not valid: %s"), name, err));
        return 1;
    }

    config.activeModel = name;
    if (!SaveConfigOrFail(config, fmt)) return 1;

    // Restart daemon via systemctl
    vinput::cli::SystemctlRestart();

    fmt.PrintSuccess(vinput::str::FmtStr(_("Active model set to '%s'. Daemon restarted."), name));
    return 0;
}

int RunModelRemove(const std::string& name, bool force, Formatter& fmt, const CliContext& /*ctx*/) {
    auto config = LoadCoreConfig();
    NormalizeCoreConfig(&config);
    auto base_dir = ResolveModelBaseDir(config);

    if (name == config.activeModel && !force) {
        fmt.PrintError(vinput::str::FmtStr(_("Cannot remove active model '%s'. Use --force to override."), name));
        return 1;
    }

    ModelManager mgr(base_dir.string());
    std::string err;
    if (!mgr.Remove(name, &err)) {
        fmt.PrintError(err);
        return 1;
    }

    fmt.PrintSuccess(vinput::str::FmtStr(_("Model '%s' removed."), name));
    return 0;
}

int RunModelInfo(const std::string& name, Formatter& fmt, const CliContext& ctx) {
    auto config = LoadCoreConfig();
    NormalizeCoreConfig(&config);
    auto base_dir = ResolveModelBaseDir(config);
    auto model_dir = base_dir / name;
    auto json_path = model_dir / "vinput-model.json";

    if (!std::filesystem::exists(json_path)) {
        fmt.PrintError(vinput::str::FmtStr(_("Model '%s' not found at: %s"), name, json_path.string()));
        return 1;
    }

    nlohmann::json j;
    {
        std::ifstream f(json_path);
        if (!f) {
            fmt.PrintError(vinput::str::FmtStr(_("Failed to read: %s"), json_path.string()));
            return 1;
        }
        try {
            f >> j;
        } catch (const std::exception& e) {
            fmt.PrintError(vinput::str::FmtStr(_("Failed to parse vinput-model.json: %s"), e.what()));
            return 1;
        }
    }

    uint64_t size = j.value("size_bytes", uint64_t{0});

    if (ctx.json_output) {
        nlohmann::json out = j;
        out["name"] = name;
        out["size_bytes"] = size;
        out["size"] = vinput::str::FormatSize(size);
        out["path"] = model_dir.string();
        fmt.PrintJson(out);
        return 0;
    }

    fmt.PrintKeyValue(_("Name"), name);
    fmt.PrintKeyValue(_("Path"), model_dir.string());
    fmt.PrintKeyValue(_("Size"), vinput::str::FormatSize(size));

    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.value().is_string()) {
            fmt.PrintKeyValue(it.key(), it.value().get<std::string>());
        } else if (it.value().is_boolean()) {
            fmt.PrintKeyValue(it.key(), it.value().get<bool>() ? "true" : "false");
        } else if (it.value().is_number()) {
            fmt.PrintKeyValue(it.key(), it.value().dump());
        }
    }

    return 0;
}
