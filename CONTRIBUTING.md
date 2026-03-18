# Contributing

Thanks for your interest in fcitx5-vinput!

## Build from Source

```bash
# Install dependencies (Arch Linux example)
sudo pacman -S cmake ninja fcitx5 pipewire libcurl nlohmann-json qt6-base

# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Project Structure

- `src/addon/` — Fcitx5 addon (key events, D-Bus client, UI)
- `src/daemon/` — ASR daemon (audio capture, inference, post-processing)
- `src/common/` — Shared code (configs, D-Bus interface, scenes)
- `src/cli/` — `vinput` CLI tool

## Submitting Changes

1. Fork the repo and create a branch from `main`
2. Make your changes — keep commits focused and atomic
3. Test locally: enable the addon in Fcitx5, verify the daemon works
4. Open a PR with a clear description of what and why

## Reporting Bugs

Use the [Bug Report](https://github.com/xifan2333/fcitx5-vinput/issues/new?template=bug_report.yml) template. Include:
- Version (`vinput --version`)
- Distribution
- Daemon logs: `journalctl --user -u vinput-daemon.service`

## Translations

Translation files are in `po/`. To add a new language:

1. Copy `po/fcitx5-vinput.pot` to `po/<lang>.po`
2. Fill in the translations
3. Submit a PR

## Code Style

- C++20
- Follow the existing style in the codebase
