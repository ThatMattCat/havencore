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


TOOLS = [BRAVE_SEARCH, WOLFRAM_ALPHA]

def GeneralTools():
    return TOOLS