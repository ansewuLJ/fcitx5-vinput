# Qwen3-ASR Integration for fcitx5-vinput

This integration adds support for Qwen3-ASR-0.6B as an alternative ASR backend, providing:
- Higher accuracy for complex audio
- Support for 52 languages and dialects
- Streaming transcription capability
- Hotwords support for domain-specific terms

## Quick Start

### 1. Start the Qwen3-ASR Server

**Option A: Direct Python**

```bash
cd /home/lijie/code/fcitx5-vinput/server
pip install qwen-asr fastapi uvicorn librosa
python qwen_asr_server.py --model Qwen/Qwen3-ASR-0.6B --port 8000
```

**Option B: Docker**

```bash
cd /home/lijie/code/fcitx5-vinput/server
docker-compose up -d
```

### 2. Configure fcitx5-vinput

Edit `~/.config/vinput/config.json`:

```json
{
  "asr_backend": {
    "type": "qwen-http",
    "qwen_http": {
      "url": "http://127.0.0.1:8000",
      "model": "Qwen/Qwen3-ASR-0.6B",
      "timeout_ms": 30000,
      "streaming": false
    }
  },
  "default_language": "zh",
  "hotwords_file": "~/.config/vinput/hotwords.txt"
}
```

### 3. Restart the daemon

```bash
systemctl --user restart vinput-daemon
```

## Hotwords Configuration

### Simple Format (hotwords.txt)

```
# Each line: word [weight]
# Weight defaults to 1.3 if not specified
人工智能 1.5
深度学习 1.4
机器学习 1.4
PostgreSQL
CUDA
```

### JSON Format (hotwords.json)

```json
{
    "technical_terms": {
        "人工智能": 1.5,
        "机器学习": 1.4,
        "深度学习": 1.4
    },
    "company_names": {
        "阿里云": 1.4
    }
}
```

### Weight Guidelines

| Type | Weight Range |
|------|-------------|
| Normal terms | 1.0 - 1.3 |
| Important terms | 1.3 - 1.5 |
| Critical terms | 1.5 - 2.0 |

## API Reference

### Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/transcribe` | POST | Transcribe audio |
| `/hotwords/load` | POST | Load hotwords |
| `/hotwords` | GET | Get current hotwords |
| `/hotwords` | DELETE | Clear hotwords |
| `/transcribe/stream` | WebSocket | Streaming transcription |

### Transcribe Request

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@test.wav" \
  -F "language=Chinese"
```

### Hotwords Request

```bash
curl -X POST http://localhost:8000/hotwords/load \
  -H "Content-Type: application/json" \
  -d '{
    "hotwords": {
      "technical_terms": {"Python": 1.3, "CUDA": 1.4}
    },
    "merge": false
  }'
```

## Switching Backends

### CLI Commands (planned)

```bash
# List available backends
vinput asr backend list

# Use local sherpa-onnx
vinput asr backend use local

# Use Qwen HTTP
vinput asr backend use qwen-http
```

### Manual Configuration

To switch between backends, edit `~/.config/vinput/config.json`:

**Local (sherpa-onnx):**
```json
{
  "asr_backend": {"type": "local"},
  "active_model": "paraformer-zh"
}
```

**Qwen HTTP:**
```json
{
  "asr_backend": {
    "type": "qwen-http",
    "qwen_http": {"url": "http://127.0.0.1:8000"}
  }
}
```

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for Qwen3-ASR-0.6B
- 16GB+ GPU memory for Qwen3-ASR-1.7B

## Troubleshooting

### Server not reachable

```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
docker logs qwen3-asr-server
```

### GPU memory issues

```bash
# Use smaller model
python qwen_asr_server.py --model Qwen/Qwen3-ASR-0.6B --gpu-memory-utilization 0.5
```

### Low recognition accuracy

1. Add hotwords for domain-specific terms
2. Ensure audio quality is good (16kHz, mono)
3. Try forcing the language if auto-detection fails
