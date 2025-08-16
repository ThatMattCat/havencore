import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import traceback
import asyncio
from typing import Dict, List, Optional, Any
import re
import time

import requests
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from wolframalpha import Client as WolframClient
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
import uvicorn
import threading

from utils import config
from utils.haos.haos import HomeAssistant
import utils.haos.haos_tools_defs as haos_tools_defs
import utils.general_tools_defs as general_tools_defs
import utils.tools as custom_tools
from shared.scripts.trace_id import with_trace, get_trace_id, set_trace_id
import shared.scripts.logger as logger_module
import shared.configs.shared_config as shared_config


logger = logger_module.get_logger('loki')
logger.setLevel(logging.DEBUG)


# OpenAI API Compatible Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "selene"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class SeleneAgent:
    """AI Agent that integrates with OpenAI-Compatible APIs and various tools."""
    
    def __init__(self, api_base: str = None, api_key: str = None):
        self.agent_name = "Selene"
        self.client = OpenAI(
            base_url=api_base or shared_config.LLM_API_BASE,
            api_key=api_key or shared_config.LLM_API_KEY or "dummy-key"
        )
        
        self.model_name = self._detect_model()
        logger.info(f"Using model: {self.model_name}")

        self.temperature = 0
        self.top_p = 0.95
        self.top_k = 20
        self.max_tokens = 1024
        
        self.haos = HomeAssistant()
        self.wolfram = WolframClient(shared_config.WOLFRAM_ALPHA_API_KEY)
        self.tools = self._setup_tools() #OpenAI format
        self.tool_functions = self._setup_tool_functions()
        self.messages = []
        self.last_query_time = time.time()

    def _detect_model(self) -> str:
        """Auto-detect the loaded model from the API"""
        try:
            models_response = self.client.models.list()
            
            if models_response and hasattr(models_response, 'data') and models_response.data:
                detected_model = models_response.data[0].id
                return detected_model

        except Exception as e:
            logger.debug(f"Standard model detection failed: {e}")

            try:
                base_url = str(self.client.base_url).rstrip('/')
                models_url = f"{base_url}/models"
                
                response = requests.get(models_url, timeout=5)
                response.raise_for_status()
                
                data = response.json()

                if 'data' in data and data['data']:
                    return data['data'][0]['id']
                elif 'models' in data and data['models']:
                    return data['models'][0]['name']
                    
            except Exception as e2:
                logger.debug(f"Direct request also failed: {e2}")
            
            logger.warning("Could not detect specific model, using default 'llama'")
            return "llama"
        
    def _setup_tools(self) -> List[Dict[str, Any]]:
        """Concatenate all tool sources into one list"""
        tools = []
        haos_tools = haos_tools_defs.HaosTools()
        general_tools = general_tools_defs.GeneralTools()
        tools = haos_tools + general_tools
        return tools
    
    def _setup_tool_functions(self) -> Dict[str, callable]:
        """Map tool names to their implementation functions"""
        return {
            'home_assistant.get_domain_entity_states': self.haos.get_domain_entity_states,
            'home_assistant.get_domain_services': self.haos.get_domain_services,
            'home_assistant.execute_service': self.haos.execute_service,
            'brave_search': self.brave_search,
            'wolfram_alpha': self.wolfram_alpha,
            'get_weather_forecast': custom_tools.get_weather_forecast,
            'query_wikipedia': custom_tools.query_wikipedia
        }
    
    async def init(self):
        """Initialize the agent with system prompt"""
        system_prompt = self.get_system_prompt()
        
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def clear_messages(self):
        """Clear conversation history and reinitialize"""
        asyncio.create_task(self.init())
    
    def get_system_prompt(self) -> str:
        prompt = f"""You are {self.agent_name}, a friendly AI assistant with access to various tools.
        Current Location: {shared_config.CURRENT_LOCATION}
        Zip Code: {shared_config.CURRENT_ZIPCODE}

        You have access to the following tools:
        - Home Assistant controls for smart home devices
        - Web search via Brave Search
        - Computational queries via Wolfram Alpha
        - Weather predictions via WeatherAPI that include astronomical data
        Use these tools when needed to help answer questions or perform actions.

        Be concise in your responses. Respond to the user as though they are a close friend.
        When responding to the user follow these rules:
        - Be brief while still resolving the user's request
        - Avoid filler words and unnecessary details
        - Convert numbers to words, eg: "One hundred and two" instead of "102"
        - Use simple language and short sentences
        - Do NOT use special characters or emojis, they cannot be translated to audio properly

        """

        return prompt

    @with_trace
    def query(self, query: str) -> str:
        """Process a user query using chat completion with tools"""

        query = f"""
### System Context
- Current date and time: {datetime.now(ZoneInfo(shared_config.CURRENT_TIMEZONE)).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}

### User Message
{query}
"""
        trace_id = get_trace_id()
        logger.debug(f"Last message time: {self.last_query_time} - Current Time: {time.time()}")
        if self.last_query_time and time.time() - self.last_query_time > 180:
            logger.debug("3 minutes without a message, resetting conversation")
            # self.clear_messages()
            system_prompt = self.get_system_prompt()
                
            self.messages = [{"role": "system", "content": system_prompt}]

        self.last_query_time = time.time()

        try:
            self.messages.append({"role": "user", "content": query})
            
            logger.info(f"Query: {query}", extra={"trace_id": trace_id})

            max_iterations = 5  #  prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.debug(f"Iteration {iteration} of tool calling loop. Calling assistant now", 
                        extra={"trace_id": trace_id})
                logger.debug(f"Current messages: {self.messages}", 
                        extra={"trace_id": trace_id})
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    # extra_body={"cache_prompt": True}
                )

                assistant_message = response.choices[0].message
                logger.debug(f"Assistant response: {assistant_message}", 
                        extra={"trace_id": trace_id})

                if assistant_message.content and not assistant_message.tool_calls:
                    tool_calls_extracted = self._extract_tool_calls_from_content(
                        assistant_message.content
                    )
                    
                    if tool_calls_extracted:
                        logger.debug(f"Extracted {len(tool_calls_extracted)} tool calls from content", 
                                extra={"trace_id": trace_id})
                        
                        formatted_tool_calls = []
                        for idx, tool_data in enumerate(tool_calls_extracted):
                            tool_call = ChatCompletionMessageToolCall(
                                id=f"call_{trace_id}_{iteration}_{idx}",
                                type="function",
                                function=Function(
                                    name=tool_data["name"],
                                    arguments=json.dumps(tool_data["arguments"])
                                )
                            )
                            formatted_tool_calls.append(tool_call)

                        assistant_message.tool_calls = formatted_tool_calls
                        assistant_message.content = None
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls == []:
                    assistant_message.tool_calls = None
                self.messages.append(assistant_message.model_dump())

                if assistant_message.tool_calls:
                    logger.debug(f"Model requested {len(assistant_message.tool_calls)} tool calls", 
                            extra={"trace_id": trace_id})

                    for tool_call in assistant_message.tool_calls:
                        logger.debug(f"Executing tool: {tool_call.function.name}", 
                                extra={"trace_id": trace_id})
                        result = self._execute_tool_call(tool_call)
                        logger.debug(f"Tool result: {result}", 
                                extra={"trace_id": trace_id})
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                    continue

                if assistant_message.content:
                    logger.info(f"Got final response after {iteration} iteration(s)", 
                            extra={"trace_id": trace_id})
                    return assistant_message.content

                logger.warning(f"Response had neither tool calls nor content", 
                            extra={"trace_id": trace_id})
                break

            if iteration >= max_iterations:
                logger.error(f"Hit maximum iterations ({max_iterations}) in tool calling loop", 
                        extra={"trace_id": trace_id})
                return "ERROR: Maximum tool calling iterations reached. The model may be stuck in a loop."
            
            # Fallback
            return "ERROR: No valid response generated"
            
        except Exception as e:
            logger.error(f"Error in query: {e}\n{traceback.format_exc()}", 
                        extra={"trace_id": trace_id})
            return f"ERROR: {str(e)}"

    @with_trace
    def query_for_api(self, messages: List[Dict[str, str]]) -> str:
        """Process messages from API endpoint - extracts the last user message"""
        trace_id = get_trace_id()
        
        try:
            user_content = None
            for message in reversed(messages):
                if message.get("role") == "user" and message.get("content"):
                    user_content = message["content"]
                    break
            
            if not user_content:
                return "ERROR: No user message found in request"
            
            logger.info(f"API Query: {user_content}", extra={"trace_id": trace_id})
            
            return self.query(user_content)
            
        except Exception as e:
            logger.error(f"Error in API query: {e}\n{traceback.format_exc()}", 
                        extra={"trace_id": trace_id})
            return f"ERROR: {str(e)}"

    def _extract_tool_calls_from_content(self, content: str) -> Optional[list]:
        """
        Extract tool calls from content wrapped in <tool_call> tags.
        
        Args:
            content: The response content that may contain tool calls
            
        Returns:
            List of tool call dictionaries or None if no tool calls found
        """
        if not content:
            return None

        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            return None
        
        tool_calls = []
        for match in matches:
            try:
                tool_data = json.loads(match.strip())

                if "name" in tool_data and "arguments" in tool_data:
                    tool_calls.append(tool_data)
                else:
                    logger.warning(f"Invalid tool call structure: {tool_data}", 
                                extra={"trace_id": get_trace_id()})
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {e}\nContent: {match}", 
                            extra={"trace_id": get_trace_id()})
                continue
        
        return tool_calls if tool_calls else None
    
    @with_trace
    def _execute_tool_call(self, tool_call) -> str:
        """Execute a single tool call"""
        trace_id = get_trace_id()
        
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.debug(f"Executing tool: {function_name} with args: {function_args}", 
                        extra={"trace_id": trace_id})
            
            if function_name in self.tool_functions:
                result = self.tool_functions[function_name](**function_args)
                return str(result)
            else:
                return f"ERROR: Unknown tool '{function_name}'"
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function.name}: {e}", 
                        extra={"trace_id": trace_id})
            return f"ERROR executing tool: {str(e)}"
    
    @with_trace
    def brave_search(self, query: str) -> str:
        """Search the web using Brave Search API"""
        trace_id = get_trace_id()
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": shared_config.BRAVE_SEARCH_API_KEY,
            "Cache-Control": "no-cache"
        }
        params = {
            "q": query,
            "count": 4,
            "safesearch": "off"
        }
        
        try:
            logger.debug(f"Brave Search Query: {query}", extra={"trace_id": trace_id})
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            results = response.json()['web']['results']
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result['title']}\n   {result['url']}\n   {result.get('description', '')}"
                )
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error in Brave Search: {e}", extra={"trace_id": trace_id})
            return f"Error searching: {str(e)}"
    
    @with_trace
    def wolfram_alpha(
        self,
        query: str,
        max_chars: Optional[int] = 1000,
        timeout: Optional[int] = 30
    ) -> str:
        """ Query the WolframAlpha LLM API."""
        
        url = "https://www.wolframalpha.com/api/v1/llm-api"
        
        params = {
            "input": query,
            "appid": shared_config.WOLFRAM_ALPHA_API_KEY,
            "maxchars": max_chars
        }
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error occurred: {e}"
            if response.status_code == 403:
                error_msg = "Invalid API key or unauthorized access"
            elif response.status_code == 400:
                error_msg = "Bad request - check your query format"
            raise ValueError(error_msg)
        except requests.exceptions.ConnectionError:
            raise ValueError("Failed to connect to WolframAlpha API")
        except requests.exceptions.Timeout:
            raise ValueError(f"Request timed out after {timeout} seconds") 
        except requests.exceptions.RequestException as e:
            raise ValueError(f"An error occurred: {e}")
            
        return response.text
    # def wolfram_alpha(self, query: str) -> str:
    #     """Query Wolfram Alpha"""
    #     trace_id = get_trace_id()
    #     resp = self.wolfram.query(input=query)
    #     logger.debug(f"Wolfram Alpha Query: {query}\nResponse: {str(resp)}", extra={"trace_id": trace_id})
    #     return str(resp)


class SeleneRunner:
    """Runner class for both Gradio interface and FastAPI server"""
    
    def __init__(self):
        self.agent = SeleneAgent()
        
    async def init(self):
        await self.agent.init()
    
    @with_trace
    def process_input(self, text: str, history=None) -> str:
        """Process user input and return response for Gradio"""
        trace_id = set_trace_id()
        
        response = self.agent.query(text)
        logger.info(f"Response: {response}", extra={"trace_id": trace_id})
        
        return response
    
    @with_trace
    def process_api_request(self, messages: List[Dict[str, str]]) -> str:
        """Process API request messages and return response"""
        trace_id = set_trace_id()
        
        response = self.agent.query_for_api(messages)
        logger.info(f"API Response: {response}", extra={"trace_id": trace_id})
        
        return response
    
    def close(self):
        """Cleanup if needed"""
        pass


# Global runner instance
runner = None

# FastAPI app for OpenAI-compatible API
app = FastAPI(title="Selene Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    global runner
    
    try:
        if not runner:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Extract messages and convert to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Process through existing agent
        response_content = runner.process_api_request(messages)
        
        # Create OpenAI-compatible response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(str(messages)),
                completion_tokens=len(response_content),
                total_tokens=len(str(messages)) + len(response_content)
            )
        )
        
        return completion_response
        
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "selene",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "selene-agent"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "selene"}


def run_fastapi_server():
    """Run FastAPI server in a separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=6006, log_level="info")


@with_trace
async def main():
    """Main entry point"""
    trace_id = get_trace_id()
    global runner
    
    try:
        logger.info("Starting Selene Agent", extra={"trace_id": trace_id})
        
        runner = SeleneRunner()
        await runner.init()

        # Start FastAPI server in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
        fastapi_thread.start()
        logger.info("Started FastAPI server on port 6003", extra={"trace_id": trace_id})

        # Setup Gradio interface
        iface = gr.Interface(
            fn=runner.process_input,
            inputs=gr.Textbox(label="Enter your message", lines=3),
            outputs=gr.Textbox(label="Selene's Response", lines=10),
            title="Selene Agent",
            description="AI Assistant with tools integration. Also serving OpenAI-compatible API at :6006/v1/chat/completions",
            theme=gr.themes.Soft()
        )
        
        logger.info("Starting Gradio Interface on port 6002", extra={"trace_id": trace_id})
        iface.launch(server_name="0.0.0.0", server_port=6002)
        
    except Exception as e:
        logger.error(f"Error in main: {e}\n{traceback.format_exc()}", 
                    extra={"trace_id": trace_id})
        raise
    finally:
        if runner:
            runner.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Selene Agent with auto-detected model')
    parser.add_argument(
        '--api-base', 
        type=str, 
        default=None,
        help='Override API base URL'
    )
    parser.add_argument(
        '--api-key', 
        type=str, 
        default=None,
        help='Override API key'
    )
    
    args = parser.parse_args()
    if args.api_base or args.api_key:
        if args.api_base:
            shared_config.LLM_API_BASE = args.api_base
        if args.api_key:
            shared_config.LLM_API_KEY = args.api_key
    
    asyncio.run(main())