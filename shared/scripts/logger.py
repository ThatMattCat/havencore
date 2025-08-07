import logging
import requests
from logging.config import dictConfig
import json

LOG_LEVEL_APP = logging.INFO # Log level for my custom code
LOG_LEVEL_OTHERS = logging.INFO  # Log level for other loggers


LOKI_URL = 'http://10.0.50.105:3100/loki/api/v1/push'  # Update with your Loki URL
# If using Grafana Cloud, the URL format is: https://logs-prod-xxx.grafana.net/loki/api/v1/push
# LOKI_USERNAME = 'your-username'  # Only needed for Grafana Cloud
# LOKI_PASSWORD = 'your-api-key'   # Only needed for Grafana Cloud

class LokiHandler(logging.Handler):
    def __init__(self, url, username=None, password=None):
        super().__init__()
        self.url = url
        self.headers = {
            'Content-Type': 'application/json'
        }
        # Add authentication for Grafana Cloud
        if username and password:
            import base64
            credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
            self.headers['Authorization'] = f'Basic {credentials}'

    def emit(self, record):
        trace_id = getattr(record, 'trace_id', '')
        
        labels = {
            'job': 'ai',  # You can customize this
            'level': record.levelname,
            'logger': record.name,
            'filename': record.filename,
            'function': record.funcName
        }

        if trace_id:
            labels['trace_id'] = trace_id
        
        timestamp_ns = str(int(record.created * 1_000_000_000))
        
        log_line = record.getMessage()
        
        payload = {
            'streams': [
                {
                    'stream': labels,
                    'values': [
                        [timestamp_ns, log_line]
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(
                self.url, 
                data=json.dumps(payload), 
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            # Avoid infinite recursion by not using the logger here
            print(f"Failed to send log to Loki: {e}")
            self.handleError(record)

def get_loki_handler():
    return LokiHandler

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'loki': {
            '()': get_loki_handler(),
            'level': 'DEBUG',
            'formatter': 'standard',
            'url': LOKI_URL,
            # 'username': LOKI_USERNAME,  # Uncomment for Grafana Cloud
            # 'password': LOKI_PASSWORD,  # Uncomment for Grafana Cloud
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': LOG_LEVEL_OTHERS,
            'propagate': True
        },
        'custom': {  # your logger
            'handlers': ['loki', 'console'],
            'level': LOG_LEVEL_APP,
            'propagate': False
        },
    }
}

def get_logger(name):
    dictConfig(LOGGING_CONFIG)
    return logging.getLogger(name)