#pragma once
#include "cli/formatter.h"
#include "common/core_config.h"
#include "common/i18n.h"
#include "common/postprocess_scene.h"

inline bool SaveConfigOrFail(const CoreConfig& config, Formatter& fmt) {
    if (!SaveCoreConfig(config)) {
        fmt.PrintError(_("Failed to save config."));
        return false;
    }
    return true;
}

inline vinput::scene::Config ToSceneConfig(const CoreConfig::Scenes& s) {
    vinput::scene::Config c;
    c.activeSceneId = s.activeScene;
    c.scenes = s.definitions;
    return c;
}

inline void FromSceneConfig(CoreConfig::Scenes& s, const vinput::scene::Config& c) {
    s.activeScene = c.activeSceneId;
    s.definitions = c.scenes;
}

inline std::string ResolveSceneId(const std::string& explicit_id) {
    if (!explicit_id.empty()) return explicit_id;
    CoreConfig config = LoadCoreConfig();
    auto sc = ToSceneConfig(config.scenes);
    return vinput::scene::Resolve(sc, sc.activeSceneId).id;
}
