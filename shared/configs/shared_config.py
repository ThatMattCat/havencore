import logging
import os

# Add these lines to shared/configs/shared_config.py after the existing configuration

# MCP Configuration
MCP_ENABLED = os.getenv('MCP_ENABLED', 'false').lower() == 'true'
MCP_PREFER_OVER_LEGACY = os.getenv('MCP_PREFER_OVER_LEGACY', 'false').lower() == 'true'

# MCP Server Configurations (JSON format in env vars)
# Example: MCP_SERVERS='[{"name": "example", "command": "node", "args": ["server.js"], "enabled": true}]'
MCP_SERVERS = os.getenv('MCP_SERVERS', '[]')

# Individual MCP server configs (alternative to JSON)
# These will be parsed if MCP_SERVERS is not provided
MCP_SERVER_EXAMPLE_ENABLED = os.getenv('MCP_SERVER_EXAMPLE_ENABLED', 'false').lower() == 'true'
MCP_SERVER_EXAMPLE_COMMAND = os.getenv('MCP_SERVER_EXAMPLE_COMMAND', '')
MCP_SERVER_EXAMPLE_ARGS = os.getenv('MCP_SERVER_EXAMPLE_ARGS', '')  # Comma-separated

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

WOLFRAM_ALPHA_API_KEY = os.getenv('WOLFRAM_ALPHA_API_KEY', "NO_WOLFRAM_TOKEN_CONFIGURED")
BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY', "NO_BRAVE_TOKEN_CONFIGURED")
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', "NO_WEATHER_TOKEN_CONFIGURED")

SRC_LAN = os.getenv('SRC_LAN', "en")
SOURCE_IP = os.getenv('SOURCE_IP', "10.0.0.100") # edge device IP address, TODO: remove this and ensure IP passed by edge devices
TTS_LANGUAGE = os.getenv('TTS_LANGUAGE', "a") # Kokoro TTS AI Model language option
TTS_VOICE = os.getenv('TTS_VOICE', "af_heart")
TTS_DEVICE = os.getenv('TTS_DEVICE', "cuda:0") # GPU index to use for text-to-speech model

STT_DEVICE = os.getenv('STT_DEVICE', "0") # GPU index to use for speech-to-text model

LLM_API_KEY = os.getenv('DEV_CUSTOM_API_KEY', "1234")

SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a friendly AI assistant with access to various tools.
        Current Location: {CURRENT_LOCATION}
        Zip Code: {CURRENT_ZIPCODE}

        You have access to the following tools:
        - Home Assistant controls for smart home devices including various media device control
        - Web search via Brave Search
        - Computational queries via Wolfram Alpha
        - Weather predictions via WeatherAPI that include astronomical data
        - Searching Wikipedia
        - Store, Delete, List, and Query "memories" using Qdrant Text Embeddings. Including user preferences, conversation history, facts, tasks, and general information
        Use those tools when needed to help answer questions or perform actions.

        Be concise in your responses. Respond to the user as though they are a close friend.
        When responding to the user follow these rules:
        - Be brief while still resolving the user's request
        - Avoid filler words and unnecessary details
        - Use simple language and short sentences
        - Do NOT use special characters or emojis, they cannot be translated to audio properly
        - Use the Qdrant memories whenever it might be relevant
        """