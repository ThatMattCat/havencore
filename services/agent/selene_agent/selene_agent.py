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
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
import uvicorn
import threading
import asyncio
import inspect

from selene_agent.utils import config
from selene_agent.utils.mcp_client_manager import MCPClientManager, MCPServerConfig, ToolSource
from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


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
        return future.result(timeout=3600)  # 1 hour timeout


class SeleneAgent:
    """AI Agent that integrates with OpenAI-Compatible APIs and various tools."""
    
    def __init__(self, api_base: str = None, api_key: str = None):
        
        # # Initialize MCP if enabled
        # self.mcp_enabled = config.MCP_ENABLED if hasattr(config, 'MCP_ENABLED') else False
        self.mcp_manager = None
        self.tools = []  # Will be populated by _setup_tools

        logger.info("Loading MCP server configurations")
        self.mcp_manager = MCPClientManager()
        self._load_mcp_server_configs()
        
        self.async_executor = AsyncToolExecutor()
        self.async_executor.start()
        self.async_executor.run_async(self._async_init())

        self.agent_name = config.AGENT_NAME
        self.client = OpenAI(
            base_url=api_base or config.LLM_API_BASE,
            api_key=api_key or config.LLM_API_KEY or "dummy-key"
        )
        
        self.model_name = self._detect_model()
        logger.info(f"Using model: {self.model_name}")

        self.temperature = 0.7
        self.top_p = 0.8
        self.top_k = 20
        self.max_tokens = 1024
        
        # self.tool_functions = {}  # Legacy compatibility
        self.messages = []
        self.last_query_time = time.time()

    def _load_mcp_server_configs(self):
        """Load MCP server configurations from environment"""
        if not self.mcp_manager:
            return
            
        # Try to load from JSON config first
        if hasattr(config, 'MCP_SERVERS'):
            try:
                servers_config = json.loads(config.MCP_SERVERS)
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

    async def _async_init(self):
        """Initialize async components in the event loop"""
        
        # Initialize database connection
        try:
            await conversation_db.initialize()
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")

        try:
            await self.mcp_manager.initialize()
            logger.info("MCP manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP manager: {e}")
        
        # Get the final tool list
        tmptools = self.mcp_manager.get_all_mcp_tools()
        if not tmptools:
            logger.warning("No tools registered.")
            # Handle the case where no tools are available
            return
        for tmp in tmptools:
            logger.info(f"Registering tool: {tmp.name}")
            self.tools.append(tmp.to_openai_format())


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
    
    async def init(self):
        """Initialize the agent with system prompt"""
        system_prompt = self.get_system_prompt()
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def clear_messages(self):
        """Clear conversation history and reinitialize"""
        asyncio.create_task(self.init())
    
    def get_system_prompt(self) -> str:
        return config.SYSTEM_PROMPT

    def query(self, query: str) -> str:
        """Process a user query using chat completion with tools"""

        query = f"""
### System Context
- Current date and time: {datetime.now(ZoneInfo(config.CURRENT_TIMEZONE)).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}

### User Message
{query}
"""
        # generate a unique id
        unique_id = f"query_{int(time.time())}"
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
            
            logger.info(f"Query: {query}")

            max_iterations = 8  #  prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.debug(f"Iteration {iteration} of tool calling loop. Calling assistant now")
                logger.debug(f"Current messages: {self.messages}")
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
                logger.debug(f"Assistant response: {assistant_message}")

                if assistant_message.content and not assistant_message.tool_calls:
                    tool_calls_extracted = self._extract_tool_calls_from_content(
                        assistant_message.content
                    )
                    
                    if tool_calls_extracted:
                        logger.debug(f"Extracted {len(tool_calls_extracted)} tool calls from content")

                        formatted_tool_calls = []
                        for idx, tool_data in enumerate(tool_calls_extracted):
                            tool_call = ChatCompletionMessageToolCall(
                                id=f"call_{unique_id}_{iteration}_{idx}",
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
                    logger.debug(f"Model requested {len(assistant_message.tool_calls)} tool calls")

                    for tool_call in assistant_message.tool_calls:
                        # logger.debug(f"Executing tool: {tool_call.function.name}")
                        result = self._execute_tool_call(tool_call)
                        logger.debug(f"Tool result: {result}")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                    continue

                if assistant_message.content:
                    logger.info(f"Got final response after {iteration} iteration(s)")
                    return assistant_message.content

                logger.warning(f"Response had neither tool calls nor content")
                break

            if iteration >= max_iterations:
                logger.error(f"Hit maximum iterations ({max_iterations}) in tool calling loop")
                return "ERROR: Maximum tool calling iterations reached. The model may be stuck in a loop."
            
            # Fallback
            return "ERROR: No valid response generated"
            
        except Exception as e:
            logger.error(f"Error in query: {e}\n{traceback.format_exc()}")
            return f"ERROR: {str(e)}"

    def query_for_api(self, messages: List[Dict[str, str]]) -> str:
        """Process messages from API endpoint - extracts the last user message"""
        
        try:
            user_content = None
            for message in reversed(messages):
                if message.get("role") == "user" and message.get("content"):
                    user_content = message["content"]
                    break
            
            if not user_content:
                return "ERROR: No user message found in request"

            logger.info(f"API Query: {user_content}")

            return self.query(user_content)
            
        except Exception as e:
            logger.error(f"Error in API query: {e}\n{traceback.format_exc()}")
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
                    logger.warning(f"Invalid tool call structure: {tool_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {e}\nContent: {match}")
                continue
        
        return tool_calls if tool_calls else None
    
    def _execute_tool_call(self, tool_call) -> str:
        """Execute a single tool call"""
        
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.debug(f"Executing tool: {function_name} with args: {function_args}")
            

            result = self.async_executor.run_async(
                self.mcp_manager.execute_tool(function_name, function_args))

            return str(result)
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function.name}: {e}")
            return f"ERROR executing tool: {str(e)}"
    
    def cleanup(self):
        """Clean up resources"""
        if self.mcp_manager:
            self.async_executor.run_async(self.mcp_manager.cleanup())
        self.async_executor.stop()


class SeleneRunner:
    """Runner class for both Gradio interface and FastAPI server"""
    
    def __init__(self):
        self.agent = SeleneAgent()
        
    async def init(self):
        await self.agent.init()
    
    def process_input(self, text: str, history=None) -> str:
        """Process user input and return response for Gradio"""
        
        response = self.agent.query(text)
        logger.info(f"Response: {response}")
        
        return response
    
    def process_api_request(self, messages: List[Dict[str, str]]) -> str:
        """Process API request messages and return response"""
        response = self.agent.query_for_api(messages)
        logger.info(f"API Response: {response}")

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

@app.get("/mcp/status")
async def get_mcp_status():
    """Get status of MCP connections"""
    global runner
    
    if not runner or not runner.agent.mcp_manager:
        return {"error": "MCP not enabled or not initialized"}
    
    return runner.agent.mcp_manager.get_server_status()

def run_fastapi_server():
    """Run FastAPI server in a separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=6006, log_level="info")

async def amain():
    """Main entry point"""
    global runner
    
    try:
        logger.info("Starting Selene Agent")

        runner = SeleneRunner()
        await runner.init()

        # Start FastAPI server in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
        fastapi_thread.start()
        logger.info("Started FastAPI server on port 6003")

        # Setup Gradio interface
        iface = gr.Interface(
            fn=runner.process_input,
            inputs=gr.Textbox(label="Enter your message", lines=3),
            outputs=gr.Textbox(label="Selene's Response", lines=10),
            title="Selene Agent",
            description="AI Assistant with tools integration. Also serving OpenAI-compatible API at :6006/v1/chat/completions",
            theme=gr.themes.Soft()
        )

        logger.info("Starting Gradio Interface on port 6002")
        iface.launch(server_name="0.0.0.0", server_port=6002)
        
    except Exception as e:
        logger.error(f"Error in main: {e}\n{traceback.format_exc()}")
        raise
    finally:
        if runner:
            runner.close()

def main():
    asyncio.run(amain())


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
            config.LLM_API_BASE = args.api_base
        if args.api_key:
            config.LLM_API_KEY = args.api_key

    asyncio.run(amain())