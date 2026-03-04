"""
Friday — Voice Pipeline
-----------------------
Flow:
  1. Block until Ollama model is warm
  2. Load Whisper
  3. Loop forever:
       a. Listen for 'Hey Friday' wake word
       b. Record voice until 1 s of silence
       c. Transcribe speech → text
       d. Stream Ollama (phi3) reply to console
       e. Go back to (a)

Run:
    ..\\fridayWake\\Scripts\\python.exe main.py
"""

import sys
from pathlib import Path

# Ensure the friday_wake/ root is always in sys.path so that
# wakeword/, listener/, and llm/ can all resolve "from config import ..."
sys.path.insert(0, str(Path(__file__).parent))

from wakeword.wake_detector import WakeWordDetector
from listener.recorder     import record_until_silence
from listener.transcriber  import Transcriber
from llm.client            import warmup, ask


def main():
    # ── 1. Warm up Ollama (blocking — model is ready before first interaction)
    warmup()

    # ── 2. Load Whisper once
    transcriber = Transcriber()

    # ── 3. Initialise wake word detector
    detector = WakeWordDetector()

    print("\n[Friday] All systems ready. Say 'Hey Friday' to begin.\n")

    try:
        while True:
            # a. Wait for wake word
            if not detector.listen():
                break  # Ctrl+C inside listen()

            print("[Friday] Wake word detected!\n")

            # b. Record until silence
            audio = record_until_silence()

            # c. Speech → text
            text = transcriber.transcribe(audio)
            if not text:
                print("[Friday] (nothing heard)\n")
                continue

            # d. Print what the user said, then stream the LLM reply
            print(f"[You]    {text}")
            ask(text)
            print()

    except KeyboardInterrupt:
        pass

    finally:
        detector.cleanup()
        print("\n[Friday] Shut down.")


if __name__ == "__main__":
    main()
