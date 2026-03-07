#include "vinput_config.h"

#include "common/postprocess_scene.h"
#include "daemon/model_manager.h"

#include <fcitx-config/iniparser.h>
#include <fcitx-utils/standardpath.h>

#include <pipewire/keys.h>
#include <pipewire/pipewire.h>
#include <spa/utils/dict.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

namespace {

fcitx::ListConstrain<fcitx::KeyConstrain> TriggerKeyListConstrain() {
    return fcitx::KeyListConstrain(fcitx::KeyConstrainFlags{
        fcitx::KeyConstrainFlag::AllowModifierOnly,
        fcitx::KeyConstrainFlag::AllowModifierLess,
    });
}

fcitx::ListConstrain<fcitx::KeyConstrain> SceneMenuKeyListConstrain() {
    return fcitx::KeyListConstrain(fcitx::KeyConstrainFlags{
        fcitx::KeyConstrainFlag::AllowModifierOnly,
        fcitx::KeyConstrainFlag::AllowModifierLess,
    });
}

std::string UserPkgConfigPath(std::string_view relative_path) {
    return (std::filesystem::path(fcitx::StandardPath::global().userDirectory(
                fcitx::StandardPath::Type::PkgConfig)) /
            std::string(relative_path))
        .string();
}

std::string BuiltInPkgDataPath(std::string_view relative_path) {
    const std::string path(relative_path);
    return fcitx::StandardPath::fcitxPath("pkgdatadir", path.c_str());
}

std::string CompactHomePath(std::string path) {
    const char* home = std::getenv("HOME");
    if (!home || home[0] == '\0') {
        return path;
    }

    const std::string_view home_view(home);
    if (path.size() >= home_view.size() &&
        path.compare(0, home_view.size(), home_view) == 0) {
        return "~" + path.substr(home_view.size());
    }
    return path;
}

std::string TriggerKeyLabel(bool chinese_ui) {
    return chinese_ui ? "触发键" : "Trigger Key";
}

std::string TriggerKeyTooltip(bool chinese_ui) {
    return chinese_ui
               ? "可配置多个触发键。支持普通按键、修饰键，以及带修饰键的组合键。"
                 "配置保存在 " + UserPkgConfigPath(kVinputConfigPath) + "。"
               : "Press and hold this key to record. Release it to start "
                 "recognition. Supports regular keys, modifier keys, and "
                 "modified key combinations. You can configure multiple "
                 "trigger keys. The config is stored at " +
                 UserPkgConfigPath(kVinputConfigPath) + ".";
}

std::string SceneMenuKeyLabel(bool chinese_ui) {
    return chinese_ui ? "后处理选单键" : "Postprocess Menu Keys";
}

std::string SceneMenuKeyTooltip(bool chinese_ui) {
    return chinese_ui
               ? "可配置多个按键，用于打开后处理选单。默认是 F9。"
               : "Configure one or more keys to open the postprocess "
                 "menu. The default is F9.";
}

std::string PagePrevKeysLabel(bool chinese_ui) {
    return chinese_ui ? "上一页键" : "Previous Page Keys";
}

std::string PagePrevKeysTooltip(bool chinese_ui) {
    return chinese_ui
               ? "用于后处理选单和后处理候选菜单翻到上一页。默认是 Page Up 和小键盘 Page Up。"
               : "Keys for paging to the previous page in the postprocess "
                 "menu and "
                 "the postprocess candidate menu. Defaults to Page Up and "
                 "keypad Page Up.";
}

std::string PageNextKeysLabel(bool chinese_ui) {
    return chinese_ui ? "下一页键" : "Next Page Keys";
}

std::string PageNextKeysTooltip(bool chinese_ui) {
    return chinese_ui
               ? "用于后处理选单和后处理候选菜单翻到下一页。默认是 Page Down 和小键盘 Page Down。"
               : "Keys for paging to the next page in the postprocess menu "
                 "and the "
                 "postprocess candidate menu. Defaults to Page Down and "
                 "keypad Page Down.";
}

std::string CaptureDeviceLabel(bool chinese_ui) {
    return chinese_ui ? "录音设备" : "Capture Device";
}

std::string CaptureDeviceTooltip(bool chinese_ui) {
    return chinese_ui
               ? "从下拉列表选择 PipeWire 输入设备。留空表示系统默认麦克风。"
               : "Choose a PipeWire input device from the list. Leave it empty "
                 "to use the system default microphone.";
}

std::string ModelLabel(bool chinese_ui) {
    return chinese_ui ? "模型" : "Model";
}

std::string ModelTooltip(bool chinese_ui, const std::string& base_dir) {
    return chinese_ui
               ? ("从固定目录选择模型：" + base_dir +
                  "。修改后需要重启 vinput-daemon 才会生效。")
               : ("Choose a model from the fixed directory: " + base_dir +
                  ". Restart vinput-daemon after changing this option.");
}

std::string LlmEnabledLabel(bool chinese_ui) {
    const std::string scene_path =
        CompactHomePath(UserPkgConfigPath(vinput::scene::kConfigPath));
    return chinese_ui ? ("启用后处理（配置：" + scene_path + "）")
                      : ("Enable Postprocess (Config: " + scene_path +
                         ")");
}

std::string LlmEnabledTooltip(bool chinese_ui) {
    return chinese_ui
               ? "识别完成后，按所选场景将结果发送到兼容聊天补全接口进行后处理。"
                 "场景配置默认优先读取 " +
                     UserPkgConfigPath(vinput::scene::kConfigPath) +
                     "，找不到时回退到 " +
                     BuiltInPkgDataPath(vinput::scene::kPkgDataPath) + "。"
               : "After ASR completes, postprocess the result with a "
                 "compatible chat completions endpoint according to the "
                 "selected scene. Scene config is read from " +
                 UserPkgConfigPath(vinput::scene::kConfigPath) +
                 " first and falls back to " +
                 BuiltInPkgDataPath(vinput::scene::kPkgDataPath) + ".";
}

std::string LlmBaseUrlLabel(bool chinese_ui) {
    return chinese_ui ? "接口地址" : "Endpoint URL";
}

std::string LlmBaseUrlTooltip(bool chinese_ui) {
    return chinese_ui
               ? "后处理接口的基础地址，例如 "
                 "http://127.0.0.1:11434/v1"
               : "Base URL for the postprocess endpoint, for example "
                 "http://127.0.0.1:11434/v1";
}

std::string LlmApiKeyLabel(bool chinese_ui) {
    return chinese_ui ? "访问密钥" : "Access Key";
}

std::string LlmApiKeyTooltip(bool chinese_ui) {
    return chinese_ui
               ? "访问后处理接口时使用的密钥。本地服务可留空。"
               : "Access key for the postprocess endpoint. Leave empty for "
                 "local services that do not require authentication.";
}

std::string LlmModelLabel(bool chinese_ui) {
    return chinese_ui ? "后处理模型" : "Postprocess Model";
}

std::string LlmModelTooltip(bool chinese_ui) {
    return chinese_ui
               ? "发送给后处理接口的模型名，不做远程下拉发现。"
               : "Model name sent to the postprocess endpoint. No remote "
                 "model discovery is performed.";
}

std::string LlmCandidateCountLabel(bool chinese_ui) {
    return chinese_ui ? "后处理候选数" : "Postprocess Candidate Count";
}

std::string LlmCandidateCountTooltip(bool chinese_ui) {
    return chinese_ui
               ? "设为 1 时直接上屏。设为 2 到 9 时，请求多个后处理候选，"
                 "并显示纵向候选列表供选择。"
               : "Set to 1 to commit directly. Set to 2 through 9 to request "
                 "multiple postprocess candidates and show a vertical "
                 "candidate list.";
}

std::string LlmTimeoutLabel(bool chinese_ui) {
    return chinese_ui ? "请求超时（毫秒）" : "Request Timeout (ms)";
}

std::string LlmTimeoutTooltip(bool chinese_ui) {
    return chinese_ui
               ? "后处理请求超时时间。超时或失败时会回退到原始识别结果。"
               : "Timeout for postprocess requests. On timeout or failure, "
                 "the raw ASR result will be used.";
}

std::vector<std::pair<std::string, std::string>>
BuildCaptureDeviceEnum(const std::vector<CaptureDeviceChoice>& devices,
                       const std::string& current_value, bool chinese_ui) {
    std::vector<std::pair<std::string, std::string>> items;
    items.emplace_back("", chinese_ui ? "系统默认" : "System Default");

    bool has_current = current_value.empty();
    for (const auto& device : devices) {
        items.emplace_back(device.value, device.label);
        if (device.value == current_value) {
            has_current = true;
        }
    }

    if (!has_current) {
        items.emplace_back(
            current_value,
            chinese_ui ? ("当前值: " + current_value)
                       : ("Current value: " + current_value));
    }

    return items;
}

std::vector<std::pair<std::string, std::string>>
BuildModelEnum(const std::vector<std::string>& models,
               const std::string& current_value, bool chinese_ui) {
    std::vector<std::pair<std::string, std::string>> items;

    bool has_current = false;
    for (const auto& model : models) {
        items.emplace_back(model, model);
        if (model == current_value) {
            has_current = true;
        }
    }

    if (!current_value.empty() && !has_current) {
        items.emplace_back(
            current_value,
            chinese_ui ? ("当前值: " + current_value)
                       : ("Current value: " + current_value));
    }

    return items;
}

const char* LookupProp(const spa_dict* props, const char* key) {
    return props ? spa_dict_lookup(props, key) : nullptr;
}

bool IsAudioSourceNode(const spa_dict* props) {
    const char* media_class = LookupProp(props, PW_KEY_MEDIA_CLASS);
    if (!media_class) {
        return false;
    }
    constexpr std::string_view kPrefix = "Audio/Source";
    return std::strncmp(media_class, kPrefix.data(), kPrefix.size()) == 0;
}

std::string BuildDeviceLabel(const spa_dict* props, const std::string& value) {
    const char* description = LookupProp(props, PW_KEY_NODE_DESCRIPTION);
    const char* nick = LookupProp(props, PW_KEY_NODE_NICK);

    std::string label = description ? description : (nick ? nick : value);
    if (label.empty()) {
        label = value;
    }
    if (label != value) {
        label += " (";
        label += value;
        label += ")";
    }
    return label;
}

struct DeviceEnumerator {
    pw_main_loop* loop = nullptr;
    pw_context* context = nullptr;
    pw_core* core = nullptr;
    pw_registry* registry = nullptr;
    spa_hook core_listener = {};
    spa_hook registry_listener = {};
    int sync_seq = -1;
    std::vector<CaptureDeviceChoice> devices;
};

void OnRegistryGlobal(void* data, uint32_t id, uint32_t permissions,
                      const char* type, uint32_t version,
                      const spa_dict* props) {
    (void)id;
    (void)permissions;
    (void)version;

    auto* state = static_cast<DeviceEnumerator*>(data);
    if (!type || std::strcmp(type, PW_TYPE_INTERFACE_Node) != 0 ||
        !IsAudioSourceNode(props)) {
        return;
    }

    const char* node_name = LookupProp(props, PW_KEY_NODE_NAME);
    const char* object_serial = LookupProp(props, PW_KEY_OBJECT_SERIAL);
    const char* value = node_name ? node_name : object_serial;
    if (!value || value[0] == '\0') {
        return;
    }

    state->devices.push_back(
        CaptureDeviceChoice{value, BuildDeviceLabel(props, value)});
}

void OnRegistryGlobalRemove(void* data, uint32_t id) {
    (void)data;
    (void)id;
}

void OnCoreDone(void* data, uint32_t id, int seq) {
    auto* state = static_cast<DeviceEnumerator*>(data);
    if (id == PW_ID_CORE && seq == state->sync_seq && state->loop) {
        pw_main_loop_quit(state->loop);
    }
}

void OnCoreError(void* data, uint32_t id, int seq, int res,
                 const char* message) {
    (void)id;
    (void)seq;
    (void)res;
    (void)message;

    auto* state = static_cast<DeviceEnumerator*>(data);
    if (state->loop) {
        pw_main_loop_quit(state->loop);
    }
}

std::vector<CaptureDeviceChoice> EnumerateCaptureDevices() {
    pw_init(nullptr, nullptr);

    DeviceEnumerator state;
    state.loop = pw_main_loop_new(nullptr);
    if (!state.loop) {
        pw_deinit();
        return {};
    }

    state.context =
        pw_context_new(pw_main_loop_get_loop(state.loop), nullptr, 0);
    if (!state.context) {
        pw_main_loop_destroy(state.loop);
        pw_deinit();
        return {};
    }

    state.core = pw_context_connect(state.context, nullptr, 0);
    if (!state.core) {
        pw_context_destroy(state.context);
        pw_main_loop_destroy(state.loop);
        pw_deinit();
        return {};
    }

    static const pw_core_events core_events = {
        .version = PW_VERSION_CORE_EVENTS,
        .done = OnCoreDone,
        .error = OnCoreError,
    };
    pw_core_add_listener(state.core, &state.core_listener, &core_events, &state);

    state.registry = pw_core_get_registry(state.core, PW_VERSION_REGISTRY, 0);
    if (!state.registry) {
        spa_hook_remove(&state.core_listener);
        pw_core_disconnect(state.core);
        pw_context_destroy(state.context);
        pw_main_loop_destroy(state.loop);
        pw_deinit();
        return {};
    }

    static const pw_registry_events registry_events = {
        .version = PW_VERSION_REGISTRY_EVENTS,
        .global = OnRegistryGlobal,
        .global_remove = OnRegistryGlobalRemove,
    };
    pw_registry_add_listener(state.registry, &state.registry_listener,
                             &registry_events, &state);

    state.sync_seq = pw_core_sync(state.core, PW_ID_CORE, 0);
    pw_main_loop_run(state.loop);

    spa_hook_remove(&state.registry_listener);
    spa_hook_remove(&state.core_listener);
    pw_proxy_destroy(reinterpret_cast<pw_proxy*>(state.registry));
    pw_core_disconnect(state.core);
    pw_context_destroy(state.context);
    pw_main_loop_destroy(state.loop);
    pw_deinit();

    std::sort(state.devices.begin(), state.devices.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.label < rhs.label;
              });
    state.devices.erase(
        std::unique(state.devices.begin(), state.devices.end(),
                    [](const auto& lhs, const auto& rhs) {
                        return lhs.value == rhs.value;
                    }),
        state.devices.end());
    return state.devices;
}

}  // namespace

void StringEnumAnnotation::dumpDescription(fcitx::RawConfig& config) const {
    config.setValueByPath("IsEnum", "True");
    if (!tooltip.empty()) {
        config.setValueByPath("Tooltip", tooltip);
    }
    for (size_t i = 0; i < items.size(); ++i) {
        config.setValueByPath("Enum/" + std::to_string(i), items[i].first);
        config.setValueByPath("EnumI18n/" + std::to_string(i), items[i].second);
    }
}

VinputConfig::VinputConfig(const VinputSettings& settings,
                           std::vector<CaptureDeviceChoice> devices,
                           std::vector<std::string> models,
                           std::string model_base_dir,
                           bool chinese_ui)
    : triggerKey(this, "TriggerKey", TriggerKeyLabel(chinese_ui),
                 settings.triggerKeys,
                 TriggerKeyListConstrain(), {},
                 fcitx::ToolTipAnnotation(TriggerKeyTooltip(chinese_ui))),
      captureDevice(
          this, "CaptureDevice", CaptureDeviceLabel(chinese_ui),
          settings.captureDevice, {}, {},
          StringEnumAnnotation{BuildCaptureDeviceEnum(devices,
                                                      settings.captureDevice,
                                                      chinese_ui),
                               CaptureDeviceTooltip(chinese_ui)}),
      modelName(this, "ModelName", ModelLabel(chinese_ui), settings.modelName,
                {}, {},
                StringEnumAnnotation{BuildModelEnum(models, settings.modelName,
                                                    chinese_ui),
                                     ModelTooltip(chinese_ui, model_base_dir)}),
      llmEnabled(this, "LlmEnabled", LlmEnabledLabel(chinese_ui),
                 settings.llmEnabled, {}, {},
                 fcitx::ToolTipAnnotation(LlmEnabledTooltip(chinese_ui))),
      sceneMenuKey(this, "SceneMenuKey", SceneMenuKeyLabel(chinese_ui),
                   settings.sceneMenuKey, SceneMenuKeyListConstrain(), {},
                   fcitx::ToolTipAnnotation(SceneMenuKeyTooltip(chinese_ui))),
      pagePrevKeys(this, "PagePrevKeys", PagePrevKeysLabel(chinese_ui),
                   settings.pagePrevKeys, TriggerKeyListConstrain(), {},
                   fcitx::ToolTipAnnotation(PagePrevKeysTooltip(chinese_ui))),
      pageNextKeys(this, "PageNextKeys", PageNextKeysLabel(chinese_ui),
                   settings.pageNextKeys, TriggerKeyListConstrain(), {},
                   fcitx::ToolTipAnnotation(PageNextKeysTooltip(chinese_ui))),
      llmBaseUrl(this, "LlmBaseUrl", LlmBaseUrlLabel(chinese_ui),
                 settings.llmBaseUrl, {}, {},
                 fcitx::ToolTipAnnotation(LlmBaseUrlTooltip(chinese_ui))),
      llmApiKey(this, "LlmApiKey", LlmApiKeyLabel(chinese_ui),
                settings.llmApiKey, {}, {},
                fcitx::ToolTipAnnotation(LlmApiKeyTooltip(chinese_ui))),
      llmModel(this, "LlmModel", LlmModelLabel(chinese_ui),
               settings.llmModel, {}, {},
               fcitx::ToolTipAnnotation(LlmModelTooltip(chinese_ui))),
      llmCandidateCount(
          this, "LlmCandidateCount", LlmCandidateCountLabel(chinese_ui),
          settings.llmCandidateCount, fcitx::IntConstrain(1, 9), {},
          fcitx::ToolTipAnnotation(LlmCandidateCountTooltip(chinese_ui))),
      llmTimeoutMs(this, "LlmTimeoutMs", LlmTimeoutLabel(chinese_ui),
                   settings.llmTimeoutMs, fcitx::IntConstrain(100, 600000), {},
                   fcitx::ToolTipAnnotation(LlmTimeoutTooltip(chinese_ui))) {
}

VinputSettings VinputConfig::settings() const {
    VinputSettings settings;
    settings.triggerKeys = triggerKey.value();
    settings.sceneMenuKey = sceneMenuKey.value();
    settings.pagePrevKeys = pagePrevKeys.value();
    settings.pageNextKeys = pageNextKeys.value();
    settings.captureDevice = captureDevice.value();
    settings.modelName = modelName.value();
    settings.llmEnabled = llmEnabled.value();
    settings.llmBaseUrl = llmBaseUrl.value();
    settings.llmApiKey = llmApiKey.value();
    settings.llmModel = llmModel.value();
    settings.llmCandidateCount = llmCandidateCount.value();
    settings.llmTimeoutMs = llmTimeoutMs.value();
    return settings;
}

bool UseChineseUi() {
    const char* locale = std::getenv("LC_ALL");
    if (!locale || locale[0] == '\0') {
        locale = std::getenv("LC_MESSAGES");
    }
    if (!locale || locale[0] == '\0') {
        locale = std::getenv("LANG");
    }
    return locale && std::strncmp(locale, "zh", 2) == 0;
}

std::vector<CaptureDeviceChoice> ListCaptureDevices() {
    return EnumerateCaptureDevices();
}

VinputSettings LoadVinputSettings() {
    ModelManager model_manager;
    VinputConfig config(VinputSettings{}, {}, {}, model_manager.GetBaseDir(),
                        false);
    fcitx::readAsIni(config, fcitx::StandardPath::Type::PkgConfig,
                     kVinputConfigPath);
    return config.settings();
}

std::string NormalizeLlmBaseUrl(std::string url) {
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    if (url.empty()) {
        return url;
    }

    constexpr std::string_view kV1 = "/v1";
    constexpr std::string_view kChatCompletions = "/chat/completions";

    if (url.size() >= kChatCompletions.size() &&
        url.compare(url.size() - kChatCompletions.size(),
                     kChatCompletions.size(), kChatCompletions) == 0) {
        url.erase(url.size() - kChatCompletions.size());
        while (!url.empty() && url.back() == '/') {
            url.pop_back();
        }
    }

    if (url.size() >= kV1.size() &&
        url.compare(url.size() - kV1.size(), kV1.size(), kV1) == 0) {
        return url;
    }

    url += "/v1";
    return url;
}

bool SaveVinputSettings(const VinputSettings& settings) {
    VinputSettings normalized = settings;
    normalized.llmBaseUrl = NormalizeLlmBaseUrl(normalized.llmBaseUrl);
    ModelManager model_manager;
    VinputConfig config(normalized, {}, {}, model_manager.GetBaseDir(), false);
    return fcitx::safeSaveAsIni(config, fcitx::StandardPath::Type::PkgConfig,
                                kVinputConfigPath);
}

std::unique_ptr<VinputConfig> BuildVinputConfig(const VinputSettings& settings) {
    ModelManager model_manager;
    return std::make_unique<VinputConfig>(
        settings, ListCaptureDevices(), model_manager.ListModels(),
        model_manager.GetBaseDir(), UseChineseUi());
}
