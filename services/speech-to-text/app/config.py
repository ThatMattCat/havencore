import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE_ID = "speech-to-text"
DEVICE_TYPE = "AI-STT"
AUDIO_FOLDER = "audio_files"
TRANSCRIPT_FOLDER = "transcript_files"
WARMUP_FILE = "assets/jfk-warmup.wav"
SAMPLE_RATE = 16000
MIN_CHUNK_SIZE = 1
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 6000
WHISPER_MODEL = "distil-large-v3"
