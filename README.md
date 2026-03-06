# fcitx5-vinput

`fcitx5-vinput` 是一个基于 Fcitx5 的本地离线语音输入法。

当前这份 README 只说明配置逻辑。

## 配置入口

配置方式有两种：

- 在 Fcitx5 配置界面里打开 `Vinput`
- 直接编辑配置文件 `~/.config/fcitx5/conf/vinput.conf`

当前可配置项有 8 个：

- `触发键`
- `录音设备`
- `模型`
- `启用 LLM 后处理`
- `LLM Base URL`
- `LLM API Key`
- `LLM Model`
- `LLM 超时毫秒`

## 配置项说明

### 触发键

- 类型：`KeyList`
- 支持单键、修饰键、组合键
- 可以配置多个触发键
- 录音逻辑是“按下开始，松开结束”

示例：

- `Control_R`
- `F8`
- `Control+Alt+V`

### 录音设备

- 配置界面里显示为下拉列表
- 列表内容来自当前 PipeWire 输入设备
- 留空表示使用系统默认麦克风

配置文件里对应项：

```ini
CaptureDevice=
```

### 模型

- 类型：字符串枚举下拉列表
- 不是自由输入路径
- 配置界面里显示为下拉列表
- 列表内容来自固定模型目录
- 只有满足文件约定的子目录才会出现在列表里

配置文件里对应项：

```ini
ModelName=paraformer-zh
```

### LLM 后处理

- `启用 LLM 后处理` 控制是否在 ASR 完成后调用 OpenAI 兼容接口
- 当前只支持 `OpenAI-compatible chat completions` 风格接口
- 不做远程模型列表发现，`LLM Model` 直接填写模型名
- `LLM Base URL` 示例：`http://127.0.0.1:11434/v1`
- `LLM API Key` 可留空，适合不鉴权的本地服务
- `LLM 超时毫秒` 超时或请求失败时会自动回退到原始识别文本

配置文件里对应项：

```ini
LlmEnabled=False
LlmBaseUrl=http://127.0.0.1:11434/v1
LlmApiKey=
LlmModel=
LlmCandidateCount=1
LlmTimeoutMs=4000
```

按键类配置项会以 `KeyList` 形式保存到 ini，比如 `TriggerKey`、`SceneMenuKey`、`PagePrevKeys`、`PageNextKeys`。

## 后处理场景

- 运行时按后处理选单键打开后处理选单，默认是 `F9`
- 当前内置场景：
  - `default`
  - `formal`
  - `code`
  - `translate`
- `default` 只做轻量文本整理，不调用 LLM
- `formal` / `code` / `translate` 会在启用 LLM 后处理时调用 OpenAI 兼容接口
- 如果 LLM 关闭、配置不完整、请求失败、超时，都会回退到原始 ASR 文本

## 模型下载

当前项目**只支持 `sherpa-onnx` 的离线 `paraformer` 模型**。

也就是说，下面这些目前**不在当前支持范围内**：

- `SenseVoice`
- `Whisper`
- `CTC`
- `streaming paraformer`
- 其他任意 ONNX 模型

当前代码是按 `offline paraformer + tokens.txt + 单文件 model.onnx/model.int8.onnx` 这套格式去加载的。

官方模型列表：

- https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

官方 GitHub Releases：

- https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

## 模型选择规范

选择模型时，按下面这几个规则：

1. 只选 `offline paraformer`
2. 目录里必须有 `tokens.txt`
3. 目录里必须有 `model.int8.onnx` 或 `model.onnx`
4. 最终目录名会直接作为 `ModelName`

适合当前项目的模型，优先顺序建议如下：

- 想要更低内存、更快速度：`sherpa-onnx-paraformer-zh-small-2024-03-09`
- 想要更高识别上限：`sherpa-onnx-paraformer-zh-2024-03-09`
- 想要兼容旧测试结果：`sherpa-onnx-paraformer-zh-2023-09-14`

如果你只想找当前已经验证过、并且和本项目兼容的中文离线 paraformer 模型，可以直接下载下面这些 GitHub 资产：

- `sherpa-onnx-paraformer-zh-small-2024-03-09`
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2
- `sherpa-onnx-paraformer-zh-2024-03-09`
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2
- `sherpa-onnx-paraformer-zh-2023-09-14`
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

## 下载与放置方式

推荐做法是：

1. 从官方模型页或 GitHub Releases 下载对应压缩包
2. 解压后得到一个模型目录
3. 把整个目录放进 `~/.local/share/fcitx5-vinput/models/`
4. 保持目录内的 `tokens.txt` 和 `model.int8.onnx`/`model.onnx` 不变

例如下载 `sherpa-onnx-paraformer-zh-small-2024-03-09` 后，目录应类似：

```text
~/.local/share/fcitx5-vinput/models/sherpa-onnx-paraformer-zh-small-2024-03-09/
├── model.int8.onnx
├── tokens.txt
└── test_wavs/
```

其中真正被程序使用的是：

- `model.int8.onnx` 或 `model.onnx`
- `tokens.txt`

目录里的其他文件是否存在，不影响下拉列表识别。

## 默认模型目录

默认根目录：

```text
~/.local/share/fcitx5-vinput/models/
```

每个模型都应该放在自己的子目录里，例如：

```text
~/.local/share/fcitx5-vinput/models/paraformer-zh/
~/.local/share/fcitx5-vinput/models/sherpa-onnx-paraformer-zh-small-2024-03-09/
```

一个子目录要想出现在“模型”下拉列表中，至少需要包含：

- `tokens.txt`
- `model.int8.onnx` 或 `model.onnx`

例如：

```text
~/.local/share/fcitx5-vinput/models/paraformer-zh/
├── model.int8.onnx
└── tokens.txt
```

## 模型切换逻辑

- 配置界面点击“确定”后会保存 `ModelName`
- 如果模型发生变化，addon 会请求重启 `vinput-daemon`
- daemon 启动时按 `ModelName` 加载对应模型

也就是说，“模型”这个配置项的选择逻辑是：

- 扫描 `~/.local/share/fcitx5-vinput/models/` 下的子目录
- 只保留包含 `tokens.txt` 和 `model.int8.onnx`/`model.onnx` 的目录
- 用目录名作为 `ModelName`
- 在配置界面里以下拉列表方式选择

## 配置文件示例

```ini
# Capture Device
CaptureDevice=
# Enable OpenAI-compatible postprocess
LlmEnabled=False
# Base URL for the OpenAI-compatible endpoint
LlmBaseUrl=http://127.0.0.1:11434/v1
# API key for the OpenAI-compatible endpoint
LlmApiKey=
# Model name sent to the OpenAI-compatible endpoint
LlmModel=
# Timeout for LLM requests
LlmTimeoutMs=4000
# Model
ModelName=sherpa-onnx-paraformer-zh-small-2024-03-09

[TriggerKey]
0=Control+Control_R
```

## 当前行为说明

- `触发键` 修改后会立即作用于 addon
- `录音设备` 会在下一次开始录音时读取
- `模型` 会在 daemon 重启后生效
- `LLM` 相关配置会在下一次识别后处理时读取
- 识别后处理配置通过后处理选单键在运行时切换，默认是 `F9`

## Release 打包

- 仓库内置了 `GitHub Actions` 发布流程，入口在 `.github/workflows/release.yml`
- 推送形如 `v0.1.0` 的 tag 后，workflow 会自动构建并上传这些产物到 GitHub Release：
  - 源码包 `fcitx5-vinput-<version>.tar.gz`
  - Ubuntu 24.04 `.deb`
  - Arch Linux `.pkg.tar.zst`
- `Debian/Ubuntu` 打包复用 `CMake + CPack`
- `Arch` 打包复用 `packaging/arch/PKGBUILD.in`
