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

LOKI_URL = os.getenv("LOKI_URL", "")

parsed_url = urlparse(HAOS_URL)
HAOS_HOST = parsed_url.hostname

HA_WS_URL = f"ws://{HAOS_HOST}/api/websocket"


# Qdrant configuration
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333

# Embeddings service configuration
EMBEDDINGS_URL = "http://embeddings:3000"
EMBEDDING_DIM = 1024  # for bge-large or e5-large

# Collection names
COLLECTION_NAMES = ["user_data"]

# Optional settings
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_IMPORTANCE = 3
MAX_SEARCH_RESULTS = 20

CURRENT_LOCATION = os.getenv("CURRENT_LOCATION", "New York, NY")
CURRENT_ZIPCODE = os.getenv("CURRENT_ZIPCODE", "10001")
SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a friendly AI assistant with access to various tools.
        Current Location: {CURRENT_LOCATION}
        Zip Code: {CURRENT_ZIPCODE}

        You have access to the following tools:
        - Home Assistant controls for smart home devices including various media device control
        - Web search via Brave Search
        - Computational queries via Wolfram Alpha
        - Weather predictions via WeatherAPI that include astronomical data
        - Searching Wikipedia
        - Store, Delete, List, and Query "memories" using Qdrant Text Embeddings. Currently focused on user data.
        Use those tools when needed to help answer questions or perform actions.

        Be concise in your responses. Respond to the user as though they are a close friend.
        When responding to the user follow these rules:
        - Be brief while still resolving the user's request
        - Avoid filler words and unnecessary details
        - Use simple language and short sentences
        - Do NOT use special characters or emojis, they cannot be translated to audio properly
        - Use the Qdrant memories whenever it might be relevant
        """