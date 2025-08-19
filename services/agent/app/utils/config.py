from logging.config import dictConfig
import os
from shared.configs import shared_config
from urllib.parse import urlparse

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TIMEZONE = shared_config.CURRENT_TIMEZONE

HAOS_TOKEN = shared_config.HAOS_TOKEN
HAOS_URL = shared_config.HAOS_URL
HAOS_USE_SSL = shared_config.HAOS_USE_SSL

parsed_url = urlparse(HAOS_URL)
HAOS_HOST = parsed_url.hostname

HA_WS_URL = f"ws://{HAOS_HOST}/api/websocket"