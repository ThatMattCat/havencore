import logging
import os
from urllib.parse import urlparse

DEBUG = os.getenv('DEBUG_LOGGING', 0)
if DEBUG:
    LOG_LEVEL_APP = logging.DEBUG
else:
    LOG_LEVEL_APP = logging.INFO
LOG_LEVEL_OTHERS = logging.INFO

LLM_API_BASE = os.getenv("LLM_API_BASE", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

MCP_SERVERS = os.getenv("MCP_SERVERS", "{}")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

AGENT_NAME = os.getenv("AGENT_NAME", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
CURRENT_TIMEZONE = os.getenv("CURRENT_TIMEZONE", "")

HAOS_TOKEN = os.getenv("HAOS_TOKEN", "")
HAOS_URL = os.getenv("HAOS_URL", "")
HAOS_USE_SSL = os.getenv("HAOS_USE_SSL", "")

PLEX_URL = os.getenv("PLEX_URL", "")
PLEX_TOKEN = os.getenv("PLEX_TOKEN", "")
PLEX_CLIENT_HA_MAP = os.getenv("PLEX_CLIENT_HA_MAP", "")

MASS_URL = os.getenv("MASS_URL", "")
MASS_TOKEN = os.getenv("MASS_TOKEN", "")

LOKI_URL = os.getenv("LOKI_URL", "")

parsed_url = urlparse(HAOS_URL)
HAOS_HOST = parsed_url.hostname

_ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
_ws_netloc = parsed_url.netloc or HAOS_HOST or ""
HA_WS_URL = f"{_ws_scheme}://{_ws_netloc}/api/websocket" if _ws_netloc else ""


# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Embeddings service configuration
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# Collection names
COLLECTION_NAMES = ["user_data"]

# Optional settings
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_IMPORTANCE = 3
MAX_SEARCH_RESULTS = 20

CONVERSATION_TIMEOUT = int(os.getenv("CONVERSATION_TIMEOUT", "180"))
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "8000"))

CURRENT_LOCATION = os.getenv("CURRENT_LOCATION", "New York, NY")
CURRENT_ZIPCODE = os.getenv("CURRENT_ZIPCODE", "10001")

# Autonomy Engine
AUTONOMY_ENABLED = os.getenv("AUTONOMY_ENABLED", "true").lower() == "true"
AUTONOMY_DISPATCH_INTERVAL_SECONDS = int(os.getenv("AUTONOMY_DISPATCH_INTERVAL_SECONDS", "30"))
AUTONOMY_BRIEFING_CRON = os.getenv("AUTONOMY_BRIEFING_CRON", "0 8 * * *")
AUTONOMY_ANOMALY_CRON = os.getenv("AUTONOMY_ANOMALY_CRON", "*/15 * * * *")
AUTONOMY_ANOMALY_COOLDOWN_MIN = int(os.getenv("AUTONOMY_ANOMALY_COOLDOWN_MIN", "30"))
AUTONOMY_MAX_RUNS_PER_HOUR = int(os.getenv("AUTONOMY_MAX_RUNS_PER_HOUR", "20"))
AUTONOMY_TURN_TIMEOUT_SEC = int(os.getenv("AUTONOMY_TURN_TIMEOUT_SEC", "60"))
AUTONOMY_BRIEFING_EMAIL_TO = os.getenv("AUTONOMY_BRIEFING_EMAIL_TO", "")
AUTONOMY_HA_NOTIFY_TARGET = os.getenv("AUTONOMY_HA_NOTIFY_TARGET", "")
AUTONOMY_BRIEFING_CAMERA_ENTITIES = [
    e.strip() for e in os.getenv("AUTONOMY_BRIEFING_CAMERA_ENTITIES", "").split(",") if e.strip()
]
AUTONOMY_ANOMALY_WATCH_DOMAINS = [
    d.strip() for d in os.getenv("AUTONOMY_ANOMALY_WATCH_DOMAINS", "binary_sensor,lock,cover").split(",") if d.strip()
]

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

SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a friendly personal assistant with access to various tools.
        Current Location: {CURRENT_LOCATION}
        Zip Code: {CURRENT_ZIPCODE}

        Use provided tools to assist the user and fulfill their requests. Tool-calling guidelines:
        - Use Home Assistant controls for smart home devices including various media device control
        - Brave Search returns URL search results and will often need to be followed by using "fetch" tool on the chosen URL
        - Wolfram Alpha is useful for complex math problems but can also provide encyclopedic knowledge
        - "create_memory" and "search_memories" utilize a vector database with embeddings. Use these tools for managing data about the user, preferences, house, and more. Search memories when relevant to improve responses to user requests.
        - Camera snapshots are returned as URLs and will often need to be sent for analysis using "query_multimodal_ai" before responding to the user.
        - Chain your tool calls across multiple messages, using one tool's response as another's input, when needed to fulfill user requests.
        - Be mindful of the user's context and preferences when using tools.

        Be concise and informal in your responses. Respond to the user as though they are a close friend.
        When responding to the user follow these rules:
        - Be brief while still resolving the user's request
        - Avoid filler words and unnecessary details
        - Use simple language and short sentences
        - Do NOT use special characters or emojis, they cannot be translated to audio properly
        - Use the Qdrant memories whenever it might be relevant
        """