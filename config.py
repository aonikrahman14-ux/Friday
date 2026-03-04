# ─── Porcupine Wake Word ─────────────────────────────────────────────────────
ACCESS_KEY   = "DVy4O8aZO1YB+gK1IyrmdkonnfK4kxPJpxjCoTKeEGd82BhvstkMVA=="
DEVICE_INDEX = None           # None = system default microphone

# ─── Whisper (faster-whisper) ────────────────────────────────────────────────
WHISPER_MODEL = "base"        # tiny | base | small | medium | large
WHISPER_LANG  = "en"          # None for auto-detect

# ─── Audio Recording ─────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000    # Hz (Whisper native rate)
CHANNELS           = 1        # mono
CHUNK_SECONDS      = 0.1      # seconds per read chunk
SILENCE_THRESHOLD  = 500      # RMS below this = silence
SILENCE_DURATION   = 1.0      # seconds of consecutive silence → stop recording
MAX_RECORD_SECONDS = 30       # safety cap: abort recording after this long

# ─── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "phi3"
