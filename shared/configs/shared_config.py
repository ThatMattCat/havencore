import logging
import os

HOST_IP_ADDRESS = os.getenv('HOST_IP_ADDRESS', '127.0.0.1')
LLM_API_BASE = f"http://{HOST_IP_ADDRESS}:8000/v1" # Can be changed if not using built-in LLM service

DEBUG = os.getenv('DEBUG_LOGGING', 0)
if DEBUG:
    LOG_LEVEL_APP = logging.DEBUG
else:
    LOG_LEVEL_APP = logging.INFO
LOG_LEVEL_OTHERS = LOG_LEVEL_APP

LOKI_URL = os.getenv('LOKI_URL', 'http://localhost:3100/loki/api/v1/push')

CURRENT_LOCATION = os.getenv('CURRENT_LOCATION', "Somewhere on Earth, probably")
CURRENT_TIMEZONE = os.getenv('CURRENT_TIMEZONE', "America/Los_Angeles")
CURRENT_ZIPCODE = os.getenv('CURRENT_ZIPCODE', "UNKNOWN")
HAOS_URL = os.getenv('HAOS_URL', "https://localhost:8123/api")
HAOS_TOKEN = os.getenv('HAOS_TOKEN', "NO_TOKEN_CONFIGURED")

WOLFRAM_ALPHA_API_KEY = os.getenv('WOLFRAM_ALPHA_API_KEY', "NO_WOLFRAM_TOKEN_CONFIGURED")
BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY', "NO_BRAVE_TOKEN_CONFIGURED")

SRC_LAN = os.getenv('SRC_LAN', "en")
SOURCE_IP = os.getenv('SOURCE_IP', "10.0.0.100") # edge device IP address, TODO: remove this and ensure IP passed by edge devices
TTS_LANGUAGE = os.getenv('TTS_LANGUAGE', "a") # Kokoro TTS AI Model language option
TTS_VOICE = os.getenv('TTS_VOICE', "af_heart")
TTS_DEVICE = os.getenv('TTS_DEVICE', "cuda:0") # GPU index to use for text-to-speech model

STT_DEVICE = os.getenv('STT_DEVICE', "0") # GPU index to use for speech-to-text model


LLM_API_KEY = os.getenv('DEV_CUSTOM_API_KEY', "1234")