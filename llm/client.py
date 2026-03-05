import json
import re
from typing import Callable, Optional

import requests

from config import OLLAMA_URL, OLLAMA_MODEL

_GENERATE_URL   = f"{OLLAMA_URL}/api/generate"
_SENTENCE_END   = re.compile(r'(?<=[.!?])\s')  # split after . ! ? followed by space


def warmup() -> None:
    """
    Block until the Ollama model is fully loaded into memory.
    Uses keep_alive=-1 so the model stays resident for the whole session.
    Response is discarded — this is a pure warm-up call.
    """
    print(f"[Friday] Warming up Ollama '{OLLAMA_MODEL}'... (blocking until ready)")
    try:
        resp = requests.post(
            _GENERATE_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": "System warmup.",
                "keep_alive": -1,
                "stream": False,
            },
            timeout=120,  # model load can take a moment on first run
        )
        resp.raise_for_status()
        print(f"[Friday] Ollama '{OLLAMA_MODEL}' is warm and ready.")
    except requests.exceptions.ConnectionError:
        print(f"[Friday] WARNING: Cannot reach Ollama at {OLLAMA_URL}. "
              "LLM replies will be skipped.")
    except Exception as exc:
        print(f"[Friday] WARNING: Ollama warmup failed — {exc}")


def ask(prompt: str, on_sentence: Optional[Callable[[str], None]] = None) -> None:
    """
    Stream a response from the Ollama model, printing tokens as they arrive.
    When a sentence boundary is detected, on_sentence(sentence) is called
    so the caller can speak each sentence immediately (TTS).
    """
    try:
        with requests.post(
            _GENERATE_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            print("[Friday] ", end="", flush=True)

            buffer = ""
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data  = json.loads(raw_line)
                token = data.get("response", "")
                print(token, end="", flush=True)

                if on_sentence:
                    buffer += token
                    # Flush complete sentences to TTS as they form
                    parts = _SENTENCE_END.split(buffer)
                    for sentence in parts[:-1]:      # all but the trailing fragment
                        if sentence.strip():
                            try:
                                on_sentence(sentence.strip())
                            except Exception as tts_exc:
                                print(f"\n[Friday] WARNING: TTS error — {tts_exc}")
                    buffer = parts[-1]               # keep the incomplete fragment

                if data.get("done"):
                    print()  # newline after streamed output
                    # Speak any remaining text after the last sentence boundary
                    if on_sentence and buffer.strip():
                        try:
                            on_sentence(buffer.strip())
                        except Exception as tts_exc:
                            print(f"\n[Friday] WARNING: TTS error — {tts_exc}")
                    break

    except requests.exceptions.ConnectionError:
        print("\n[Friday] WARNING: Ollama not reachable — skipping LLM reply.")
    except Exception as exc:
        print(f"\n[Friday] WARNING: Ollama error — {exc}")
