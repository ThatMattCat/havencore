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