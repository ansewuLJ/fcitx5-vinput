# Qwen3-ASR 集成到 fcitx5-vinput 完整指南

本指南介绍如何将 Qwen3-ASR-0.6B 模型集成到 fcitx5-vinput，实现高质量离线语音输入。

## 目录

1. [系统要求](#系统要求)
2. [安装依赖](#安装依赖)
3. [构建 fcitx5-vinput](#构建-fcitx5-vinput)
4. [启动 ASR 服务](#启动-asr-服务)
5. [配置 vinput](#配置-vinput)
6. [使用方法](#使用方法)
7. [热词功能](#热词功能)
8. [常见问题](#常见问题)

---

## 系统要求

- Ubuntu 24.04 (或其他 Linux 发行版)
- NVIDIA GPU (推荐，用于加速推理)
- Python 3.10+
- CMake 3.16+
- GCC 13+

---

## 安装依赖

### 1. 系统依赖

```bash
# 构建依赖
sudo apt install -y cmake g++ pkg-config

# fcitx5 依赖
sudo apt install -y fcitx5-dev fcitx5-modules-dev extra-cmake-modules

# 音频依赖
sudo apt install -y libpipewire-0.3-dev

# 网络依赖
sudo apt install -y libcurl4-openssl-dev

# JSON 库
sudo apt install -y nlohmann-json3-dev

# systemd DBus
sudo apt install -y libsystemd-dev

# 国际化
sudo apt install -y gettext

# Qt6 (GUI 可选)
sudo apt install -y qt6-base-dev qt6-base-dev-tools
```

### 2. Python 环境

使用 `uv` 管理 Python 环境：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
cd /home/lijie/code/fcitx5-vinput
uv venv
source .venv/bin/activate

# 安装 Python 依赖
uv pip install torch transformers numpy librosa fastapi uvicorn
uv pip install qwen-asr  # 可选，用于更好的性能
```

---

## 构建 fcitx5-vinput

```bash
cd /home/lijie/code/fcitx5-vinput

# 配置
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 编译 (使用所有 CPU 核心)
cmake --build build -j$(nproc)

# 安装
sudo cmake --install build
```

---

## 启动 ASR 服务

### 方式一：使用 transformers 后端

```bash
cd /home/lijie/code/fcitx5-vinput/server

# 激活虚拟环境
source ../.venv/bin/activate

# 启动服务 (使用 HuggingFace 模型)
uv run python qwen_asr_server.py \
    --model Qwen/Qwen3-ASR-0.6B \
    --host 0.0.0.0 \
    --port 8086 \
    --device cuda:0
```

### 方式二：使用本地模型路径

```bash
uv run python qwen_asr_server.py \
    --model /path/to/your/Qwen3-ASR-0.6B \
    --host 0.0.0.0 \
    --port 8086 \
    --device cuda:0
```

### 方式三：使用 vLLM 后端 (更快)

```bash
# 安装 vLLM 支持
uv pip install qwen-asr[vllm]

# 启动服务
uv run python qwen_asr_server.py \
    --model Qwen/Qwen3-ASR-0.6B \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8086
```

### 验证服务

```bash
# 健康检查
curl http://127.0.0.1:8086/health
# 输出: {"status":"healthy","model_loaded":true,"backend":"transformers","hotwords_count":0}

# 查看 API 文档
# 浏览器访问: http://127.0.0.1:8086/docs
```

---

## 配置 vinput

### 1. 初始化配置

```bash
vinput init
```

### 2. 修改配置文件

编辑 `~/.config/vinput/config.json`：

```json
{
  "asr_backend": {
    "type": "qwen-http",
    "qwen_http": {
      "url": "http://127.0.0.1:8086",
      "timeout_ms": 30000
    }
  },
  "default_language": "zh"
}
```

### 3. 启动守护进程

```bash
# 重载 systemd
systemctl --user daemon-reload

# 启动 vinput-daemon
systemctl --user enable --now vinput-daemon

# 检查状态
systemctl --user status vinput-daemon
```

### 4. 重启 fcitx5

```bash
fcitx5 -r -d
```

---

## 使用方法

### 快捷键

默认配置 (可在 `~/.config/fcitx5/conf/vinput.conf` 修改)：

| 功能 | 快捷键 |
|------|--------|
| 开始/停止录音 | `右 Alt` |
| 命令模式 | `右 Ctrl` |
| 场景菜单 | `右 Shift` |

### 使用步骤

1. 确保输入焦点在文本框
2. 按 `右 Alt` 开始录音
3. 说话
4. 再按 `右 Alt` 停止录音
5. 等待识别结果输入

### 命令行测试

```bash
# 开始录音
vinput recording start

# 停止并识别 (在另一个终端执行)
vinput recording stop

# 查看状态
vinput status

# 列出录音设备
vinput device list
```

### 查看日志

```bash
# 实时查看 vinput-daemon 日志
journalctl --user -u vinput-daemon -f

# 查看 ASR 服务日志
# 直接在运行服务的终端查看
```

---

## 热词功能

### 热词格式

```json
{
  "category_name": {
    "热词": 权重,
    "另一个热词": 权重
  }
}
```

权重范围：1.0 - 2.0，越高越重要。

### 示例热词文件

创建 `hotwords.json`：

```json
{
  "tech_terms": {
    "Qwen3-ASR": 1.5,
    "fcitx5": 1.4,
    "vinput": 1.4,
    "Ubuntu": 1.3,
    "Linux": 1.3
  },
  "names": {
    "张三": 1.5,
    "李四": 1.5
  }
}
```

### 加载热词

#### 方式一：启动时加载

```bash
uv run python qwen_asr_server.py \
    --model Qwen/Qwen3-ASR-0.6B \
    --port 8086 \
    --hotwords-file hotwords.json
```

#### 方式二：运行时加载

```bash
# 加载热词
curl -X POST http://127.0.0.1:8086/hotwords/load \
    -H "Content-Type: application/json" \
    -d '{
        "hotwords": {
            "tech": {"vinput": 1.5, "fcitx5": 1.4}
        },
        "merge": true
    }'

# 查看当前热词
curl http://127.0.0.1:8086/hotwords

# 清空热词
curl -X DELETE http://127.0.0.1:8086/hotwords
```

---

## 常见问题

### Q: 录音后没有识别结果？

1. 检查 ASR 服务是否运行：
   ```bash
   curl http://127.0.0.1:8086/health
   ```

2. 检查 vinput-daemon 日志：
   ```bash
   journalctl --user -u vinput-daemon -n 20
   ```

3. 检查录音设备：
   ```bash
   vinput device list
   ```

### Q: 提示 "Unsupported language" 错误？

服务端已自动将 `zh` 转换为 `Chinese`。如果仍有问题，手动指定语言：

```json
{
  "asr_backend": {
    "type": "qwen-http",
    "qwen_http": {
      "url": "http://127.0.0.1:8086"
    }
  },
  "default_language": "Chinese"
}
```

支持的语言：Chinese, English, Cantonese, Japanese, Korean 等。

### Q: 识别速度慢？

1. 确保使用 GPU：
   ```bash
   uv run python qwen_asr_server.py --device cuda:0
   ```

2. 尝试 vLLM 后端：
   ```bash
   uv pip install qwen-asr[vllm]
   uv run python qwen_asr_server.py --backend vllm
   ```

### Q: 如何修改快捷键？

编辑 `~/.config/fcitx5/conf/vinput.conf`：

```ini
[TriggerKey]
0=Control+Shift+V

[CommandKeys]
0=Control+Shift+C
```

修改后重启 fcitx5：`fcitx5 -r -d`

### Q: 如何查看识别详细日志？

```bash
# ASR 服务日志
# 启动时添加 --log-level debug

# vinput-daemon 日志
journalctl --user -u vinput-daemon -f
```

---

## API 参考

### POST /transcribe

语音识别接口。

**请求 (JSON)**：
```json
{
  "audio_base64": "<base64 编码的 WAV 音频>",
  "language": "Chinese",
  "hotwords": {
    "category": {"word": 1.5}
  }
}
```

**响应**：
```json
{
  "text": "识别结果",
  "language": "Chinese",
  "success": true,
  "error": null
}
```

### GET /health

健康检查。

**响应**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "transformers",
  "hotwords_count": 0
}
```

### POST /hotwords/load

加载热词。

**请求**：
```json
{
  "hotwords": {
    "category": {"word": 1.5}
  },
  "merge": false
}
```

### WebSocket /transcribe/stream

流式识别 (实验性)。

---

## 文件位置

| 文件 | 路径 |
|------|------|
| vinput 配置 | `~/.config/vinput/config.json` |
| fcitx5 插件配置 | `~/.config/fcitx5/conf/vinput.conf` |
| vinput-daemon 服务 | `/usr/local/share/systemd/user/vinput-daemon.service` |
| ASR 服务脚本 | `/home/lijie/code/fcitx5-vinput/server/qwen_asr_server.py` |
| 日志 | `journalctl --user -u vinput-daemon` |
