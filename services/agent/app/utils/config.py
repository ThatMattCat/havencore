from logging.config import dictConfig
import os
from shared.configs import shared_config

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
TIMEZONE = shared_config.CURRENT_TIMEZONE   