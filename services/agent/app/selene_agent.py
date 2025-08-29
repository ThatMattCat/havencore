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
import asyncio
import inspect

from utils import config
import utils.general_tools_defs as general_tools_defs
import utils.tools as custom_tools
from shared.scripts.trace_id import with_trace, get_trace_id, set_trace_id
import shared.scripts.logger as logger_module
import shared.configs.shared_config as shared_config
from utils.mcp_client_manager import MCPClientManager, MCPServerConfig, ToolSource
from utils.unified_tool_registry import UnifiedToolRegistry
from conversation_db import conversation_db
from utils.tool_migration_helper import ToolMigrationHelper
from utils.qdrant_tools import QdrantTools

# TODO: Make everything async

logger = logger_module.get_logger('loki')


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

class AsyncToolExecutor:
    """Manages a single event loop for all async tool executions"""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self._started = False
        self._lock = threading.Lock()
        
    def start(self):
        """Start the async event loop in a separate thread"""
        with self._lock:
            if self._started:
                return
                
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            self._started = True
            
    def _run_loop(self):
        """Run the event loop in a thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def stop(self):
        """Stop the event loop"""
        if self.loop and self._started:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self._started = False
            
    def run_async(self, coro):
        """Run an async coroutine in the persistent event loop"""
        if not self._started:
            self.start()
            
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=900)  # 30 second timeout


class SeleneAgent:
    """AI Agent that integrates with OpenAI-Compatible APIs and various tools."""
    
    def __init__(self, api_base: str = None, api_key: str = None):
        
        self.wolfram = WolframClient(shared_config.WOLFRAM_ALPHA_API_KEY)
        self.qdrant_tools = QdrantTools()
        
        # Initialize MCP if enabled
        self.mcp_enabled = shared_config.MCP_ENABLED if hasattr(shared_config, 'MCP_ENABLED') else False
        self.mcp_manager = None
        self.tool_registry = None

        if self.mcp_enabled:
            logger.info("MCP support is enabled")
            self.mcp_manager = MCPClientManager()
            self._load_mcp_server_configs()
        else:
            logger.info("MCP support is disabled")
        
        # Initialize the unified tool registry
        self.tool_registry = UnifiedToolRegistry(self.mcp_manager)
        self.migration_helper = ToolMigrationHelper(self.tool_registry)
        
        self.async_executor = AsyncToolExecutor()
        self.async_executor.start()
        self.async_executor.run_async(self._async_init())

        self.agent_name = config.AGENT_NAME
        self.client = OpenAI(
            base_url=api_base or shared_config.LLM_API_BASE,
            api_key=api_key or shared_config.LLM_API_KEY or "dummy-key"
        )
        
        self.model_name = self._detect_model()
        logger.info(f"Using model: {self.model_name}")

        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k = 20
        self.max_tokens = 1024
        
        # Setup tools will now use the registry
        self.tools = []  # Will be populated by _setup_tools
        self.tool_functions = {}  # Legacy compatibility
        self.messages = []
        self.last_query_time = time.time()

    def _load_mcp_server_configs(self):
        """Load MCP server configurations from environment"""
        if not self.mcp_manager:
            return
            
        # Try to load from JSON config first
        if hasattr(shared_config, 'MCP_SERVERS'):
            try:
                import json
                servers_config = json.loads(shared_config.MCP_SERVERS)
                for server_cfg in servers_config:
                    mcp_config = MCPServerConfig(
                        name=server_cfg.get('name'),
                        command=server_cfg.get('command'),
                        args=server_cfg.get('args', []),
                        env=server_cfg.get('env', {}),
                        enabled=server_cfg.get('enabled', True)
                    )
                    self.mcp_manager.add_server(mcp_config)
                    logger.info(f"Loaded MCP server config: {mcp_config.name}")
            except Exception as e:
                logger.warning(f"Could not parse MCP_SERVERS JSON: {e}")
        
        # Example of loading individual server configs
        # This is a fallback method if JSON parsing fails
        if hasattr(shared_config, 'MCP_SERVER_EXAMPLE_ENABLED'):
            if shared_config.MCP_SERVER_EXAMPLE_ENABLED:
                command = getattr(shared_config, 'MCP_SERVER_EXAMPLE_COMMAND', '')
                args_str = getattr(shared_config, 'MCP_SERVER_EXAMPLE_ARGS', '')
                args = args_str.split(',') if args_str else []
                
                if command:
                    mcp_config = MCPServerConfig(
                        name="example",
                        command=command,
                        args=args,
                        enabled=True
                    )
                    self.mcp_manager.add_server(mcp_config)

    async def _async_init(self):
        """Initialize async components in the event loop"""
        
        # Initialize database connection
        try:
            await conversation_db.initialize()
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
        
        # Initialize MCP manager if enabled
        if self.mcp_manager:
            try:
                await self.mcp_manager.initialize()
                logger.info("MCP manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MCP manager: {e}")
        
        # Register all tools with the registry
        await self._register_all_tools()

        self.migration_helper.auto_detect_migrations()
        
        # Apply tool preferences from migration helper
        for tool_name in self.migration_helper.migrations:
            if self.migration_helper.should_use_mcp(tool_name):
                # This tool should use MCP
                logger.info(f"Tool '{tool_name}' configured to use MCP version")
        
        # Get the final tool list
        self.tools = await self.tool_registry.get_tools_for_llm()
        
        # Log tool registry status
        status = self.tool_registry.get_registry_status()

        logger.info(f"Tool Registry Status: {json.dumps(status, indent=2)}")

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
        
    async def _register_all_tools(self):
        """Register all legacy tools with the unified registry"""
        
        # Since we've removed the old Home Assistant implementation,
        # we only register general tools as legacy fallbacks
        logger.info("Home Assistant tools now provided via MCP server only")
        
        # Register general tools
        general_tools = general_tools_defs.GeneralTools()
        general_functions = {
            'brave_search': self.brave_search,
            'wolfram_alpha': self.wolfram_alpha,
            'get_weather_forecast': custom_tools.get_weather_forecast,
            'query_wikipedia': custom_tools.query_wikipedia,
        }
        self.tool_registry.register_legacy_tools_bulk(general_tools, general_functions)

        # qdrant_tools = self.qdrant_tools.get_tool_definitions()
        # qdrant_functions = {
        #     'store_memory': self.qdrant_tools.store_memory,
        #     'search_memories': self.qdrant_tools.search_memories,
        #     'update_memory': self.qdrant_tools.update_memory,
        #     'delete_memory': self.qdrant_tools.delete_memory,
        #     'list_recent': self.qdrant_tools.list_recent
        # }
        # self.tool_registry.register_legacy_tools_bulk(qdrant_tools, qdrant_functions)

        # Set tool preference based on config
        if shared_config.MCP_PREFER_OVER_LEGACY:
            self.tool_registry.set_tool_preference(shared_config.MCP_PREFER_OVER_LEGACY)
        
        # Keep legacy tool_functions for compatibility
        self.tool_functions = {**general_functions}

    def _setup_tools(self) -> List[Dict[str, Any]]:
        """This method is now replaced by the registry but kept for compatibility"""
        # Tools are now managed by the registry and populated in _async_init
        return self.tools
    
    def _setup_tool_functions(self) -> Dict[str, callable]:
        """Map tool names to their implementation functions"""
        return {
            'brave_search': self.brave_search,
            'wolfram_alpha': self.wolfram_alpha,
            'get_weather_forecast': custom_tools.get_weather_forecast,
            'query_wikipedia': custom_tools.query_wikipedia,
        }
    
    async def init(self):
        """Initialize the agent with system prompt"""
        system_prompt = self.get_system_prompt()
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # Refresh tools in case they've changed
        if self.tool_registry:
            self.tools = await self.tool_registry.get_tools_for_llm()
    
    def clear_messages(self):
        """Clear conversation history and reinitialize"""
        asyncio.create_task(self.init())
    
    def get_system_prompt(self) -> str:
        return shared_config.SYSTEM_PROMPT

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
            
            # Store conversation history before resetting
            if hasattr(self, 'messages') and self.messages and len(self.messages) > 1:
                try:
                    # Store the conversation asynchronously
                    metadata = {
                        'reset_reason': 'timeout_3_minutes',
                        'message_count': len(self.messages),
                        'last_query_time': self.last_query_time,
                        'agent_name': getattr(self, 'agent_name', 'Selene')
                    }
                    
                    # Run the async store operation
                    async def store_conversation():
                        await conversation_db.store_conversation_history(
                            messages=self.messages,
                            session_id=trace_id,
                            metadata=metadata
                        )
                    
                    # Execute the async operation
                    self.async_executor.run_async(store_conversation())
                    logger.info(f"Stored conversation history with {len(self.messages)} messages before reset")
                    
                except Exception as e:
                    logger.error(f"Failed to store conversation history before reset: {e}")
            
            # self.clear_messages()
            system_prompt = self.get_system_prompt()
                
            self.messages = [{"role": "system", "content": system_prompt}]

        self.last_query_time = time.time()

        try:
            self.messages.append({"role": "user", "content": query})
            
            logger.info(f"Query: {query}", extra={"trace_id": trace_id})

            max_iterations = 6  #  prevent infinite loops
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
        Failsafe to Extract tool calls from content wrapped in <tool_call> tags.
        
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
        """Execute a single tool call - now routes through the registry"""
        trace_id = get_trace_id()
        
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.debug(f"Executing tool: {function_name} with args: {function_args}", 
                        extra={"trace_id": trace_id})
            
            # Route through the unified registry
            if self.tool_registry:
                # Check the source for logging
                tool_source = self.tool_registry.get_tool_source(function_name)
                if tool_source:
                    logger.debug(f"Tool '{function_name}' source: {tool_source.value}", 
                               extra={"trace_id": trace_id})
                
                # Execute through registry (handles both legacy and MCP)
                result = self.async_executor.run_async(
                    self.tool_registry.execute_tool(function_name, function_args)
                )
                return str(result)
            else:
                # Fallback to legacy execution if registry not available
                if function_name in self.tool_functions:
                    func = self.tool_functions[function_name]
                    if inspect.iscoroutinefunction(func):
                        logger.debug(f"Running async tool: {function_name}")
                        result = self.async_executor.run_async(func(**function_args))
                    else:
                        logger.debug(f"Running sync tool: {function_name}")
                        result = func(**function_args)
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
    
    def cleanup(self):
        """Clean up resources"""
        self.async_executor.run_async(self.ha_media_controller.library_manager.disconnect())
        
        # Clean up MCP connections if enabled
        if self.mcp_manager:
            self.async_executor.run_async(self.mcp_manager.cleanup())
        
        self.async_executor.stop()


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
        self.agent.cleanup()
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

@app.get("/tools/status")
async def get_tools_status():
    """Get status of all registered tools"""
    global runner
    
    if not runner or not runner.agent.tool_registry:
        return {"error": "Tool registry not initialized"}
    
    return runner.agent.tool_registry.get_registry_status()

@app.get("/mcp/status")
async def get_mcp_status():
    """Get status of MCP connections"""
    global runner
    
    if not runner or not runner.agent.mcp_manager:
        return {"error": "MCP not enabled or not initialized"}
    
    return runner.agent.mcp_manager.get_server_status()

@app.post("/tools/preference")
async def set_tool_preference(prefer_mcp: bool = False):
    """Set tool preference when conflicts exist"""
    global runner
    
    if not runner or not runner.agent.tool_registry:
        return {"error": "Tool registry not initialized"}
    
    runner.agent.tool_registry.set_tool_preference(prefer_mcp)
    return {"success": True, "prefer_mcp": prefer_mcp}

@app.get("/tools/migrations")
async def get_tool_migrations():
    """Get tool migration status"""
    global runner
    
    if not runner or not runner.agent.migration_helper:
        return {"error": "Migration helper not initialized"}
    
    return runner.agent.migration_helper.get_migration_status()

@app.get("/tools/migrations/{tool_name}")
async def get_tool_migration(tool_name: str):
    """Get migration status for a specific tool"""
    global runner
    
    if not runner or not runner.agent.migration_helper:
        return {"error": "Migration helper not initialized"}
    
    migration = runner.agent.migration_helper.get_tool_migration(tool_name)
    if migration:
        return migration.to_dict()
    else:
        return {"error": f"No migration found for tool '{tool_name}'"}

@app.post("/tools/migrations/{tool_name}/preference")
async def set_tool_migration_preference(tool_name: str, use_mcp: bool = False):
    """Set preference for a specific tool"""
    global runner
    
    if not runner or not runner.agent.migration_helper:
        return {"error": "Migration helper not initialized"}
    
    runner.agent.migration_helper.set_tool_preference(tool_name, use_mcp)
    
    # Also update the registry preference for this specific tool
    runner.agent.tool_registry.set_tool_preference(use_mcp)
    
    return {
        "success": True,
        "tool": tool_name,
        "use_mcp": use_mcp
    }

@app.get("/tools/migrations/{tool_name}/plan")
async def get_migration_plan(tool_name: str):
    """Get migration plan for a specific tool"""
    global runner
    
    if not runner or not runner.agent.migration_helper:
        return {"error": "Migration helper not initialized"}
    
    return runner.agent.migration_helper.create_migration_plan(tool_name)

@app.get("/tools/migrations/report")
async def get_migration_report():
    """Get human-readable migration report"""
    global runner
    
    if not runner or not runner.agent.migration_helper:
        return {"error": "Migration helper not initialized"}
    
    report = runner.agent.migration_helper.generate_migration_report()
    return {"report": report}


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