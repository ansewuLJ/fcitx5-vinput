#!/usr/bin/env python3
"""
Qwen3-ASR Server - HTTP API for speech recognition with hotwords support.

Provides:
  - POST /transcribe - Non-streaming transcription
  - WebSocket /transcribe/stream - Streaming transcription
  - POST /hotwords/load - Load hotwords JSON
  - GET /health - Health check

Usage:
    python qwen_asr_server.py --model Qwen/Qwen3-ASR-0.6B --host 0.0.0.0 --port 8000
"""

import os
# Disable transformers verbose warnings (must be set before importing transformers)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import base64
import io
import json
import logging
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instance
asr_model = None
asr_processor = None
current_hotwords: Dict[str, Dict[str, float]] = {}


class TranscribeRequest(BaseModel):
    """Request model for transcription."""
    audio_base64: Optional[str] = None  # Base64 encoded audio
    language: Optional[str] = None  # Force language (e.g., "Chinese", "English")
    hotwords: Optional[Dict[str, Dict[str, float]]] = None  # Inline hotwords
    hotword_weights: float = 1.5  # Global hotword weight multiplier
    streaming: bool = False


class TranscribeResponse(BaseModel):
    """Response model for transcription."""
    text: str
    language: str
    success: bool
    error: Optional[str] = None


class HotwordsLoadRequest(BaseModel):
    """Request model for loading hotwords."""
    hotwords: Dict[str, Dict[str, float]]
    merge: bool = False  # Merge with existing or replace


class AppState:
    """Application state container."""
    model: Any = None
    processor: Any = None
    backend: str = "transformers"  # "transformers" or "vllm"
    device: str = "cuda:0"
    hotwords: Dict[str, Dict[str, float]] = {}


state = AppState()

# Language code mapping: ISO code -> Qwen3-ASR language name
LANGUAGE_CODE_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hk": "Cantonese",
    "yue": "Cantonese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def normalize_language(lang: Optional[str]) -> Optional[str]:
    """Normalize language code to Qwen3-ASR format."""
    if lang is None:
        return None
    lang_lower = lang.lower().strip()
    return LANGUAGE_CODE_MAP.get(lang_lower, lang)  # Return original if not in map


def load_model(model_path: str, backend: str = "transformers", device: str = "cuda:0"):
    """Load Qwen3-ASR model."""
    global asr_model, asr_processor
    
    logger.info(f"Loading model: {model_path} (backend: {backend})")
    
    if backend == "transformers":
        try:
            from qwen_asr import Qwen3ASRModel
            state.model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=device,
            )
            # Also load processor for hotwords support
            from transformers import AutoProcessor
            state.processor = AutoProcessor.from_pretrained(model_path)
            state.backend = "transformers"
            logger.info("Model loaded successfully with transformers backend")
        except ImportError:
            logger.warning("qwen-asr not installed, falling back to transformers directly")
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            state.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            state.processor = AutoProcessor.from_pretrained(model_path)
            state.backend = "transformers_raw"
    
    elif backend == "vllm":
        try:
            from qwen_asr import Qwen3ASRModel
            state.model = Qwen3ASRModel.LLM(
                model=model_path,
                gpu_memory_utilization=0.7,
            )
            state.backend = "vllm"
            logger.info("Model loaded successfully with vLLM backend")
        except ImportError:
            raise RuntimeError("vLLM backend requires 'pip install qwen-asr[vllm]'")
    
    state.device = device


def parse_audio_base64(audio_base64: str) -> tuple:
    """Parse base64 encoded audio to numpy array."""
    # Decode base64
    audio_bytes = base64.b64decode(audio_base64)
    
    # Try to parse as WAV
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            
            # Convert to numpy
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float and normalize
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
            
            # Convert to mono if stereo
            if n_channels > 1:
                audio_float = audio_float.reshape(-1, n_channels).mean(axis=1)
            
            return audio_float, 16000
    except Exception as e:
        logger.error(f"Failed to parse audio: {e}")
        raise ValueError(f"Invalid audio format: {e}")


def transcribe_audio(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    language: Optional[str] = None,
    hotwords: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple:
    """
    Transcribe audio using the loaded model.
    
    Returns:
        tuple: (text, detected_language)
    """
    global current_hotwords
    
    # Merge hotwords
    effective_hotwords = current_hotwords.copy()
    if hotwords:
        for category, words in hotwords.items():
            if category not in effective_hotwords:
                effective_hotwords[category] = {}
            effective_hotwords[category].update(words)
    
    # Prepare hotwords for Qwen3-ASR API
    # Qwen3-ASR expects context parameter (a string with hotwords/keywords)
    context_str = ""
    if effective_hotwords:
        # Flatten hotwords into a comma-separated string
        hw_list = []
        for category, words in effective_hotwords.items():
            for word, weight in words.items():
                hw_list.append(word)
        context_str = ",".join(hw_list)
        logger.info(f"Using context (hotwords): {context_str}")
    
    try:
        if state.backend == "transformers":
            transcribe_kwargs = {
                "audio": (audio_data, sample_rate),
            }
            if language:
                transcribe_kwargs["language"] = language
            if context_str:
                transcribe_kwargs["context"] = context_str
            
            result = state.model.transcribe(**transcribe_kwargs)
            text = result[0].text if hasattr(result[0], 'text') else result[0]['text']
            detected_lang = result[0].language if hasattr(result[0], 'language') else result[0].get('language', 'unknown')
            return text, detected_lang
        
        elif state.backend == "vllm":
            transcribe_kwargs = {
                "audio": (audio_data, sample_rate),
            }
            if language:
                transcribe_kwargs["language"] = language
            if context_str:
                transcribe_kwargs["context"] = context_str
            
            result = state.model.transcribe(**transcribe_kwargs)
            text = result[0].text if hasattr(result[0], 'text') else result[0]['text']
            detected_lang = result[0].language if hasattr(result[0], 'language') else result[0].get('language', 'unknown')
            return text, detected_lang
        
        else:
            # Raw transformers backend
            inputs = state.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(state.device)
            
            outputs = state.model.generate(**inputs)
            text = state.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return text, language or "auto"
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")


# FastAPI app
app = FastAPI(
    title="Qwen3-ASR Server",
    description="HTTP API for Qwen3-ASR speech recognition with hotwords support",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": state.model is not None,
        "backend": state.backend,
        "hotwords_count": sum(len(words) for words in state.hotwords.values())
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request):
    """
    Transcribe audio file.
    
    Supports both JSON body and multipart/form-data.
    
    JSON body:
    {
        "audio_base64": "<base64>",
        "language": "Chinese",
        "hotwords": {"category": {"word": weight}}
    }
    
    Form data:
    - audio: uploaded audio file
    - audio_base64: base64 encoded audio
    - language: language code
    - hotwords: JSON string
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    content_type = request.headers.get("content-type", "")
    
    audio = None
    audio_base64 = None
    language = None
    hotwords = None
    hotword_weights = 1.5
    
    if "application/json" in content_type:
        # Handle JSON request
        try:
            body = await request.json()
            audio_base64 = body.get("audio_base64")
            language = normalize_language(body.get("language"))
            hotwords = body.get("hotwords")  # Already a dict
            hotword_weights = body.get("hotword_weights", 1.5)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    else:
        # Handle multipart/form-data
        form = await request.form()
        audio = form.get("audio")
        audio_base64 = form.get("audio_base64")
        language = normalize_language(form.get("language"))
        hotwords_str = form.get("hotwords")
        if hotwords_str:
            try:
                hotwords = json.loads(hotwords_str)
            except json.JSONDecodeError:
                logger.warning(f"Invalid hotwords JSON: {hotwords_str}")
        hotword_weights = float(form.get("hotword_weights", 1.5))
    
    if audio is None and audio_base64 is None:
        raise HTTPException(status_code=400, detail="Either 'audio' or 'audio_base64' is required")
    
    try:
        # Parse audio
        if audio:
            # Read uploaded file
            audio_bytes = await audio.read()
            # Save to temp file and process
            with tempfile.NamedTemporaryFile(suffix=Path(audio.filename).suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            try:
                import librosa
                audio_data, sr = librosa.load(tmp_path, sr=16000, mono=True)
            finally:
                Path(tmp_path).unlink()
        else:
            audio_data, sr = parse_audio_base64(audio_base64)
        
        # Parse hotwords
        hw_dict = None
        if hotwords:
            try:
                hw_dict = json.loads(hotwords)
            except json.JSONDecodeError:
                logger.warning(f"Invalid hotwords JSON: {hotwords}")
        
        # Transcribe
        text, detected_lang = transcribe_audio(
            audio_data=audio_data,
            sample_rate=16000,
            language=language,
            hotwords=hw_dict,
        )
        
        return TranscribeResponse(
            text=text,
            language=detected_lang,
            success=True,
        )
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return TranscribeResponse(
            text="",
            language="",
            success=False,
            error=str(e)
        )


@app.post("/hotwords/load")
async def load_hotwords(request: HotwordsLoadRequest):
    """
    Load hotwords for recognition enhancement.
    
    Format:
    {
        "hotwords": {
            "technical_terms": {"Qwen3-ASR": 1.5, "FP16": 1.3},
            "company_names": {"阿里云": 1.4}
        },
        "merge": false  // If true, merge with existing; if false, replace
    }
    """
    global current_hotwords
    
    if request.merge:
        for category, words in request.hotwords.items():
            if category not in current_hotwords:
                current_hotwords[category] = {}
            current_hotwords[category].update(words)
    else:
        current_hotwords = request.hotwords.copy()
    
    state.hotwords = current_hotwords
    
    logger.info(f"Loaded {sum(len(w) for w in current_hotwords.values())} hotwords")
    
    return {
        "success": True,
        "hotwords_count": sum(len(w) for w in current_hotwords.values()),
        "categories": list(current_hotwords.keys())
    }


@app.get("/hotwords")
async def get_hotwords():
    """Get current hotwords configuration."""
    return {
        "hotwords": current_hotwords,
        "count": sum(len(w) for w in current_hotwords.values())
    }


@app.delete("/hotwords")
async def clear_hotwords():
    """Clear all hotwords."""
    global current_hotwords
    current_hotwords = {}
    state.hotwords = {}
    return {"success": True, "message": "Hotwords cleared"}


# WebSocket for streaming transcription
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@app.websocket("/transcribe/stream")
async def transcribe_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming transcription.
    
    Protocol:
    1. Client sends: {"type": "config", "language": "Chinese", "hotwords": {...}}
    2. Client sends: {"type": "audio", "data": "<base64>"}
    3. Server responds: {"type": "partial", "text": "..."} or {"type": "final", "text": "..."}
    4. Client sends: {"type": "end"} to finish
    """
    await manager.connect(websocket)
    
    audio_buffer = []
    config = {
        "language": None,
        "hotwords": None,
    }
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "config":
                config["language"] = data.get("language")
                config["hotwords"] = data.get("hotwords")
                await websocket.send_json({"type": "ready"})
            
            elif data.get("type") == "audio":
                # Accumulate audio
                audio_base64 = data.get("data", "")
                audio_data, sr = parse_audio_base64(audio_base64)
                audio_buffer.append(audio_data)
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "received",
                    "duration_ms": len(audio_data) * 1000 // 16000
                })
            
            elif data.get("type") == "transcribe":
                # Transcribe accumulated audio
                if audio_buffer:
                    combined_audio = np.concatenate(audio_buffer)
                    
                    # Use vLLM streaming if available
                    if state.backend == "vllm" and hasattr(state.model, 'transcribe_stream'):
                        # Streaming transcription
                        async for chunk in state.model.transcribe_stream(
                            audio=(combined_audio, 16000),
                            language=config["language"],
                        ):
                            await websocket.send_json({
                                "type": "partial",
                                "text": chunk.text
                            })
                        
                        await websocket.send_json({
                            "type": "final",
                            "text": chunk.text,
                            "language": chunk.language
                        })
                    else:
                        # Non-streaming fallback
                        text, lang = transcribe_audio(
                            combined_audio,
                            language=config["language"],
                            hotwords=config["hotwords"],
                        )
                        await websocket.send_json({
                            "type": "final",
                            "text": text,
                            "language": lang
                        })
                    
                    # Clear buffer
                    audio_buffer = []
            
            elif data.get("type") == "end":
                # Transcribe any remaining audio and close
                if audio_buffer:
                    combined_audio = np.concatenate(audio_buffer)
                    text, lang = transcribe_audio(
                        combined_audio,
                        language=config["language"],
                        hotwords=config["hotwords"],
                    )
                    await websocket.send_json({
                        "type": "final",
                        "text": text,
                        "language": lang
                    })
                break
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        manager.disconnect(websocket)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR HTTP Server")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-0.6B",
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind"
    )
    parser.add_argument(
        "--hotwords-file",
        type=str,
        default=None,
        help="Path to hotwords JSON file to load on startup"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization for vLLM backend"
    )
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model, args.backend, args.device)
    
    # Load initial hotwords
    if args.hotwords_file:
        try:
            with open(args.hotwords_file, 'r', encoding='utf-8') as f:
                hotwords = json.load(f)
                global current_hotwords
                current_hotwords = hotwords
                state.hotwords = hotwords
                logger.info(f"Loaded {sum(len(w) for w in hotwords.values())} hotwords from {args.hotwords_file}")
        except Exception as e:
            logger.error(f"Failed to load hotwords file: {e}")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
