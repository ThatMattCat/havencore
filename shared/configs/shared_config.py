import logging
import os

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', 5432)
POSTGRES_DB = os.getenv('POSTGRES_DB', 'mydatabase')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'myuser')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'mypassword')

# MCP Configuration
MCP_ENABLED = os.getenv('MCP_ENABLED', 'false').lower() == 'true'
MCP_PREFER_OVER_LEGACY = os.getenv('MCP_PREFER_OVER_LEGACY', 'false').lower() == 'true'

# MCP Server Configurations (JSON format in env vars)
# Example: MCP_SERVERS='[{"name": "example", "command": "node", "args": ["server.js"], "enabled": true}]'
MCP_SERVERS = os.getenv('MCP_SERVERS', '[]')

HOST_IP_ADDRESS = os.getenv('HOST_IP_ADDRESS', '127.0.0.1')
LLM_API_BASE = f"http://{HOST_IP_ADDRESS}:8000/v1" # Can be changed if not using built-in LLM service

DEBUG = os.getenv('DEBUG_LOGGING', 0)
if DEBUG:
    LOG_LEVEL_APP = logging.DEBUG
else:
    LOG_LEVEL_APP = logging.INFO
LOG_LEVEL_OTHERS = logging.INFO

LOKI_URL = os.getenv('LOKI_URL', 'http://localhost:3100/loki/api/v1/push')

CURRENT_LOCATION = os.getenv('CURRENT_LOCATION', "Somewhere on Earth, probably")
CURRENT_TIMEZONE = os.getenv('CURRENT_TIMEZONE', "America/Los_Angeles")
CURRENT_ZIPCODE = os.getenv('CURRENT_ZIPCODE', "UNKNOWN")
AGENT_NAME = os.getenv('AGENT_NAME', "Selene") # Default agent name, can be overridden by environment variable
HAOS_URL = os.getenv('HAOS_URL', "https://localhost:8123/api")
HAOS_TOKEN = os.getenv('HAOS_TOKEN', "NO_TOKEN_CONFIGURED")
HAOS_USE_SSL = True

PLEX_URL = os.getenv('PLEX_URL', "http://localhost:32400")
PLEX_TOKEN = os.getenv('PLEX_TOKEN', "NO_PLEX_TOKEN_CONFIGURED")

MASS_URL = os.getenv('MASS_URL', "http://localhost:8095")
MASS_TOKEN = os.getenv('MASS_TOKEN', "")

WOLFRAM_ALPHA_API_KEY = os.getenv('WOLFRAM_ALPHA_API_KEY', "NO_WOLFRAM_TOKEN_CONFIGURED")
BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY', "NO_BRAVE_TOKEN_CONFIGURED")
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', "NO_WEATHER_TOKEN_CONFIGURED")

SRC_LAN = os.getenv('SRC_LAN', "en")
SOURCE_IP = os.getenv('SOURCE_IP', "10.0.0.100") # edge device IP address, TODO: remove this and ensure IP passed by edge devices
TTS_LANGUAGE = os.getenv('TTS_LANGUAGE', "a") # Kokoro TTS AI Model language option
TTS_VOICE = os.getenv('TTS_VOICE', "af_heart")
TTS_DEVICE = os.getenv('TTS_DEVICE', "cuda:0") # GPU index to use for text-to-speech model

STT_DEVICE = os.getenv('STT_DEVICE', "0") # GPU index to use for speech-to-text model

LLM_API_KEY = os.getenv('LLM_API_KEY', "1234")

# Autonomy Engine
AUTONOMY_ENABLED = os.getenv('AUTONOMY_ENABLED', 'true').lower() == 'true'
AUTONOMY_DISPATCH_INTERVAL_SECONDS = int(os.getenv('AUTONOMY_DISPATCH_INTERVAL_SECONDS', '30'))
AUTONOMY_BRIEFING_CRON = os.getenv('AUTONOMY_BRIEFING_CRON', '0 8 * * *')
AUTONOMY_ANOMALY_CRON = os.getenv('AUTONOMY_ANOMALY_CRON', '*/15 * * * *')
AUTONOMY_ANOMALY_COOLDOWN_MIN = int(os.getenv('AUTONOMY_ANOMALY_COOLDOWN_MIN', '30'))
AUTONOMY_MAX_RUNS_PER_HOUR = int(os.getenv('AUTONOMY_MAX_RUNS_PER_HOUR', '20'))
AUTONOMY_TURN_TIMEOUT_SEC = int(os.getenv('AUTONOMY_TURN_TIMEOUT_SEC', '60'))
AUTONOMY_BRIEFING_NOTIFY_TO = os.getenv('AUTONOMY_BRIEFING_NOTIFY_TO', '') or os.getenv('AUTONOMY_BRIEFING_EMAIL_TO', '')
AUTONOMY_HA_NOTIFY_TARGET = os.getenv('AUTONOMY_HA_NOTIFY_TARGET', '')
AUTONOMY_BRIEFING_CAMERA_ENTITIES = [
    e.strip() for e in os.getenv('AUTONOMY_BRIEFING_CAMERA_ENTITIES', '').split(',') if e.strip()
]
AUTONOMY_ANOMALY_WATCH_DOMAINS = [
    d.strip() for d in os.getenv('AUTONOMY_ANOMALY_WATCH_DOMAINS', 'binary_sensor,lock,cover').split(',') if d.strip()
]

# --- v3 reactive autonomy ---
AUTONOMY_WEBHOOK_ENABLED = os.getenv('AUTONOMY_WEBHOOK_ENABLED', 'false').lower() == 'true'
AUTONOMY_MQTT_ENABLED = os.getenv('AUTONOMY_MQTT_ENABLED', 'false').lower() == 'true'
AUTONOMY_MQTT_CLIENT_ID = os.getenv('AUTONOMY_MQTT_CLIENT_ID', 'selene-autonomy')
AUTONOMY_MQTT_RECONNECT_MAX_SEC = int(os.getenv('AUTONOMY_MQTT_RECONNECT_MAX_SEC', '60'))
AUTONOMY_DEFAULT_QUIET_START = os.getenv('AUTONOMY_DEFAULT_QUIET_START', '')
AUTONOMY_DEFAULT_QUIET_END = os.getenv('AUTONOMY_DEFAULT_QUIET_END', '')
AUTONOMY_DEFAULT_QUIET_POLICY = os.getenv('AUTONOMY_DEFAULT_QUIET_POLICY', 'defer')
AUTONOMY_DEFAULT_EVENT_RATE_LIMIT = os.getenv('AUTONOMY_DEFAULT_EVENT_RATE_LIMIT', '10/min')

# --- v2 memory consolidation ---
AUTONOMY_MEMORY_REVIEW_CRON = os.getenv("AUTONOMY_MEMORY_REVIEW_CRON", "0 3 * * *")
AUTONOMY_MEMORY_MAX_SCAN = int(os.getenv("AUTONOMY_MEMORY_MAX_SCAN", "5000"))
AUTONOMY_MEMORY_LLM_CALL_CAP = int(os.getenv("AUTONOMY_MEMORY_LLM_CALL_CAP", "20"))

MEMORY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_HALF_LIFE_DAYS", "60"))
MEMORY_ACCESS_COEF = float(os.getenv("MEMORY_ACCESS_COEF", "0.5"))

MEMORY_HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("MEMORY_HDBSCAN_MIN_CLUSTER_SIZE", "5"))
MEMORY_HDBSCAN_MIN_SAMPLES = int(os.getenv("MEMORY_HDBSCAN_MIN_SAMPLES", "3"))

MEMORY_L4_MIN_IMPORTANCE = float(os.getenv("MEMORY_L4_MIN_IMPORTANCE", "4"))
MEMORY_L4_MIN_AGE_DAYS = int(os.getenv("MEMORY_L4_MIN_AGE_DAYS", "14"))
MEMORY_L4_MIN_ACCESS_COUNT = int(os.getenv("MEMORY_L4_MIN_ACCESS_COUNT", "3"))

MEMORY_L2_PRUNE_AGE_DAYS = int(os.getenv("MEMORY_L2_PRUNE_AGE_DAYS", "180"))
MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD = float(os.getenv("MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD", "0.5"))

MEMORY_L3_RANK_BOOST = float(os.getenv("MEMORY_L3_RANK_BOOST", "1.2"))
MEMORY_L4_MAX_ENTRIES = int(os.getenv("MEMORY_L4_MAX_ENTRIES", "20"))
MEMORY_L4_WARN_TOKENS = int(os.getenv("MEMORY_L4_WARN_TOKENS", "1500"))

SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a friendly AI assistant. Respond as though the user is a close friend.
        Current Location: {CURRENT_LOCATION}
        Zip Code: {CURRENT_ZIPCODE}

        Guidelines:
        - Be brief while still resolving the user's request. Avoid filler words.
        - Use simple language and short sentences.
        - Check memories (search_memories) before answering anything about the user's preferences, people, places, or history — it's often where the context lives. Save new durable facts with create_memory when the user shares something worth remembering later.
        - For math, unit conversions, or precise factual computation, prefer Wolfram Alpha over Brave Search.
        - The weather tool returns astronomy data too (sunrise, sunset, moon phase) — use it for those, don't search the web.

        HARD RULES:
        - Never invent entity_ids. Call ha_get_entities_in_area or ha_get_domain_entity_states first to confirm the exact id.
        - When one tool's output feeds another tool's input, make those calls in separate turns — do not emit them in parallel.
        - No emojis or special characters in responses (TTS cannot read them).
        """
