import logging
import ipaddress
import os
import shared.configs.shared_config as shared_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

AUDIO_DIR = os.path.join(BASE_DIR, "output")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6003
BASE_URL = f"http://{shared_config.IP_ADDRESS}:{SERVER_PORT}/"