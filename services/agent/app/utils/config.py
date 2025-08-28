from logging.config import dictConfig
import os
from shared.configs import shared_config
from urllib.parse import urlparse

MCP_SERVERS = shared_config.MCP_SERVERS

POSTGRES_HOST = shared_config.POSTGRES_HOST
POSTGRES_PORT = shared_config.POSTGRES_PORT
POSTGRES_DB = shared_config.POSTGRES_DB
POSTGRES_USER = shared_config.POSTGRES_USER
POSTGRES_PASSWORD = shared_config.POSTGRES_PASSWORD

AGENT_NAME = shared_config.AGENT_NAME
WEATHER_API_KEY = shared_config.WEATHER_API_KEY
BRAVE_SEARCH_API_KEY = shared_config.BRAVE_SEARCH_API_KEY
TIMEZONE = shared_config.CURRENT_TIMEZONE

HAOS_TOKEN = shared_config.HAOS_TOKEN
HAOS_URL = shared_config.HAOS_URL
HAOS_USE_SSL = shared_config.HAOS_USE_SSL

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