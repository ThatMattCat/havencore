# General tools definitions

BRAVE_SEARCH = {
    "type": "function",
    "function": {
        "name": "brave_search",
        "description": "Query the Brave search engine.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for."
                }
            },
            "required": ["query"]
        }
    }
}

WOLFRAM_ALPHA = {
    "type": "function",
    "function": {
        "name": "wolfram_alpha",
        "description": "Query the Wolfram Alpha computational engine.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for."
                }
            },
            "required": ["query"]
        }
    }
}

WEATHER_FORECAST = {
    "type": "function",
    "function": {
        "name": "get_weather_forecast",
        "description": "Get weather forecast and astronomy data for a location. Returns temperature, conditions, precipitation, wind, humidity, UV index, and astronomy information (sunrise/sunset, moon phases).",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g., 'Seattle, WA'), ZIP code (e.g., '98101'), or coordinates (e.g., '47.6,-122.3')"
                },
                "date": {
                    "type": "string",
                    "description": "Date for forecast in YYYY-MM-DD format (e.g., '2025-09-15'). If omitted, returns today's weather."
                }
            },
            "required": ["location"]
        }
    }
}

TOOLS = [BRAVE_SEARCH, WOLFRAM_ALPHA, WEATHER_FORECAST]

def GeneralTools():
    return TOOLS