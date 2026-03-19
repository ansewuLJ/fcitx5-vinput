#!/usr/bin/env python3
"""
Test script for Qwen3-ASR server.

Usage:
    python test_server.py --audio test.wav --server http://localhost:8000
"""

import argparse
import base64
import json
import sys
import wave

import requests


def load_audio(audio_path: str) -> str:
    """Load audio file and return base64 encoded WAV."""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_health(server_url: str) -> bool:
    """Test server health endpoint."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server healthy (backend: {data.get('backend', 'unknown')})")
            print(f"  Hotwords count: {data.get('hotwords_count', 0)}")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_transcribe(server_url: str, audio_path: str, language: str = None) -> dict:
    """Test transcription endpoint."""
    audio_base64 = load_audio(audio_path)
    
    data = {"audio_base64": audio_base64}
    if language:
        data["language"] = language
    
    try:
        response = requests.post(
            f"{server_url}/transcribe",
            data=data,
            timeout=60
        )
        result = response.json()
        
        if result.get("success"):
            print(f"✓ Transcription successful")
            print(f"  Language: {result.get('language', 'unknown')}")
            print(f"  Text: {result.get('text', '')}")
        else:
            print(f"✗ Transcription failed: {result.get('error', 'unknown error')}")
        
        return result
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        return {"success": False, "error": str(e)}


def test_hotwords(server_url: str) -> bool:
    """Test hotwords loading."""
    hotwords = {
        "technical_terms": {
            "Qwen3-ASR": 1.5,
            "Python": 1.3,
            "GPU": 1.3
        }
    }
    
    try:
        # Load hotwords
        response = requests.post(
            f"{server_url}/hotwords/load",
            json={"hotwords": hotwords, "merge": False},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Loaded {data.get('hotwords_count', 0)} hotwords")
            return True
        else:
            print(f"✗ Hotwords load failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Hotwords test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-ASR server")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Server URL")
    parser.add_argument("--audio", help="Audio file to transcribe")
    parser.add_argument("--language", help="Force language")
    parser.add_argument("--skip-health", action="store_true",
                        help="Skip health check")
    parser.add_argument("--skip-hotwords", action="store_true",
                        help="Skip hotwords test")
    
    args = parser.parse_args()
    
    print(f"Testing Qwen3-ASR server at {args.server}")
    print("-" * 50)
    
    # Health check
    if not args.skip_health:
        if not test_health(args.server):
            print("\nServer not available. Start with:")
            print(f"  python server/qwen_asr_server.py --model Qwen/Qwen3-ASR-0.6B")
            sys.exit(1)
    
    # Hotwords test
    if not args.skip_hotwords:
        test_hotwords(args.server)
    
    # Transcription test
    if args.audio:
        print(f"\nTranscribing: {args.audio}")
        test_transcribe(args.server, args.audio, args.language)
    
    print("\n" + "-" * 50)
    print("Tests completed.")


if __name__ == "__main__":
    main()
