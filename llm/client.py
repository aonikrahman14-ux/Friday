import json
import requests

from config import OLLAMA_URL, OLLAMA_MODEL

_GENERATE_URL = f"{OLLAMA_URL}/api/generate"


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


def ask(prompt: str) -> None:
    """
    Stream a response from the Ollama model and print tokens as they arrive.
    Prints a warning and returns silently if Ollama is unreachable.
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
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                print(data.get("response", ""), end="", flush=True)
                if data.get("done"):
                    print()  # final newline
                    break

    except requests.exceptions.ConnectionError:
        print("\n[Friday] WARNING: Ollama not reachable — skipping LLM reply.")
    except Exception as exc:
        print(f"\n[Friday] WARNING: Ollama error — {exc}")
