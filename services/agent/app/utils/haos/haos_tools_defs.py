GET_DOMAIN_ENTITY_STATES = {
    "type": "function",
    "function": {
        "name": "home_assistant.get_domain_entity_states",
        "description": "Get the current states of all entities in a domain. The domain is specified as a parameter. The response is a dictionary where the keys are the entity IDs and the values are the entity states.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "The domain for which to get the entity states. The domain is a string that specifies the type of entities to get the states for."
                }
            },
            "required": ["domain"]
        }
    }
}

GET_DOMAIN_SERVICES = {
    "type": "function",
    "function": {
        "name": "home_assistant.get_domain_services",
        "description": "Get the services available in a domain. The domain is specified as a parameter. The response is a dictionary where the keys are the service IDs and the values are the service descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "The domain for which to get the services. The domain is a string that specifies the type of services to get."
                }
            },
            "required": ["domain"]
        }
    }
}

EXECUTE_SERVICE = {
    "type": "function",
    "function": {
        "name": "home_assistant.execute_service",
        "description": "Execute a service on an entity. The entity ID and service name are specified as parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The entity ID on which to execute the service. The entity ID is a string that specifies the entity to act upon."
                },
                "service": {
                    "type": "string",
                    "description": "The name of the service to execute. The service name is a string that specifies the service to execute."
                }
            },
            "required": ["entity_id", "service"]
        }
    }
}

TOOLS = [GET_DOMAIN_ENTITY_STATES, GET_DOMAIN_SERVICES, EXECUTE_SERVICE]

def HaosTools():
    return TOOLS