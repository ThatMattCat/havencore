# Selene Agent - Tool Calling/etc

This is essentially responsible for the perceived "Agent" persona, since it controls inference and final response.

The layer between any services and the final LLM Model. Controls tool/function-calling, system prompts, inference configs, and similar. Receives text, and responds with text, on port 6002. 6002 hosts a Gradio API, it can called called as an API at `http://HOST:6002/predict` or accessed directly at `http://HOST:6002`

Example API call to this service:

```
from gradio_client import Client

agent_client = Client(f"http://{shared_config.IP_ADDRESS}:6002/")
response_data = await loop.run_in_executor(
    None,
    lambda: agent_client.predict(
        transcript,
        api_name="/predict"
    )
)
print(repsonse_data) # Pure text reponse, nothing else
```
