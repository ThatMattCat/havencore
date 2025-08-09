import json
from typing import List
from homeassistant_api import Client, Group, Entity, Service, State
import requests
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)

import haos.haos_tools_defs as haos_tools_defs
import config
import shared.scripts.logger as logger_module
import shared.configs.shared_config as shared_config

logger = logger_module.get_logger('loki')

class HomeAssistant:

    def __init__(self):
        self._api_url = shared_config.HAOS_URL
        self._token = shared_config.HAOS_TOKEN

    def get_domain_entity_states(self, domain: str) -> str:
        with Client(self._api_url, self._token) as client:
            domain_entity_states = {}
            entities = client.get_entities()
            domain_entity_states = {entity.state.entity_id: entity.state.state for _, entity in entities[domain].entities.items()}
            return json.dumps(domain_entity_states)

    def get_domain_services(self, domain: str) -> str:
        with Client(self._api_url, self._token) as client:
            domain_obj = client.get_domain(domain)
            return json.dumps({service.service_id: service.description for _, service in domain_obj.services.items()})
        
    def get_entity_state(self, entity_id: str) -> str:
        with Client(self._api_url, self._token) as client:
            entity = client.get_entity(entity_id=entity_id)
            return entity.state.state

    def execute_service(self, entity_id: str, service: str):
        with Client(self._api_url, self._token) as client:
            domain_name = entity_id.split('.')[0]
            domain_obj = client.get_domain(domain_name)

            service_obj = domain_obj.services[service]
            changes = service_obj.trigger(entity_id=entity_id)
            # BROKEN - check changes for entity_id to ensure state changed
            final_state = self.get_entity_state(entity_id)
            return f"Service {service} executed on {entity_id}"
    
    def get_tools_defs(self) -> list[dict[str, any]]:
        """
        Retrieve the list of tool definitions in OAI-Compatible /chat/completions "tools" format.

        Returns:
            list[dict[str, any]]: A list of dictionaries containing the definitions of tools.
        """
        return haos_tools_defs.HaosTools()

# For testing    
def main():
    ha = HomeAssistant()
    light_states = ha.get_domain_entity_states("light")
    switch_states = ha.get_domain_entity_states("switch")
    tmp = ha.get_entity_state('light.reading_light_1')
    change_response = ha.execute_service("light.reading_light_1", "turn_on")
    change_response2 = ha.execute_service("light.reading_light_1", "turn_off")
    for key, value in light_states.items():
        print(f"{key}: {value}")
    for key, value in switch_states.items():
        print(f"{key}: {value}")
    light_services = ha.get_domain_services("light")
    for key, value in light_services.items():
        print(f"{key}: {value}")
    switch_services = ha.get_domain_services("switch")

if __name__ == "__main__":
    main()
        