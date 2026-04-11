"""
Agent Orchestrator - Event-based agent loop for streaming and tool visibility.

Separates the LLM interaction loop from the FastAPI server, yielding typed events
that can be consumed by both non-streaming (collect all) and streaming (SSE/WebSocket) endpoints.
"""

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')

TOOL_RESULT_MAX_CHARS = config.TOOL_RESULT_MAX_CHARS
MAX_TOOL_ITERATIONS = 8


class EventType(str, Enum):
    """Types of events emitted by the orchestrator"""
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE_CHUNK = "response_chunk"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    """An event emitted during agent processing"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


def truncate_tool_result(result: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Truncate a tool result to prevent unbounded context growth."""
    if len(result) <= max_chars:
        return result
    omitted = len(result) - max_chars
    return result[:max_chars] + f"\n[...truncated, {omitted} chars omitted]"


class AgentOrchestrator:
    """
    Orchestrates the agent query loop, yielding events for each step.

    This separates the agent logic from transport (HTTP/WebSocket),
    enabling both streaming and non-streaming consumption.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        mcp_manager: MCPClientManager,
        model_name: str,
        tools: List[Dict[str, Any]],
    ):
        self.client = client
        self.mcp_manager = mcp_manager
        self.model_name = model_name
        self.tools = tools

        self.messages: List[Dict[str, Any]] = []
        self.last_query_time: float = time.time()
        self.agent_name = config.AGENT_NAME

        self.temperature = 0.7
        self.top_p = 0.8
        self.max_tokens = 1024

    async def initialize(self):
        """Initialize with system prompt"""
        system_prompt = config.SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": system_prompt}]

    async def _check_session_timeout(self):
        """Check if conversation should be reset due to timeout."""
        timeout = config.CONVERSATION_TIMEOUT
        if self.last_query_time and time.time() - self.last_query_time > timeout:
            logger.debug(f"{timeout}s without a message, resetting conversation")

            if self.messages and len(self.messages) > 1:
                try:
                    metadata = {
                        'reset_reason': f'timeout_{timeout}_seconds',
                        'message_count': len(self.messages),
                        'last_query_time': self.last_query_time,
                        'agent_name': self.agent_name,
                    }
                    await conversation_db.store_conversation_history(
                        messages=self.messages,
                        metadata=metadata,
                    )
                    logger.info(f"Stored conversation history with {len(self.messages)} messages before reset")
                except Exception as e:
                    logger.error(f"Failed to store conversation history before reset: {e}")

            await self.initialize()

    async def run(self, user_message: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Process a user message, yielding events as the agent works.

        Events:
        - THINKING: Agent is calling the LLM
        - TOOL_CALL: Agent is calling a tool
        - TOOL_RESULT: Tool returned a result
        - RESPONSE_CHUNK: Part of the final response (for streaming)
        - DONE: Final response complete
        - ERROR: An error occurred
        """
        wrapped_message = f"""
### System Context
- Current date and time: {datetime.now(ZoneInfo(config.CURRENT_TIMEZONE)).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}

### User Message
{user_message}
"""
        unique_id = f"query_{int(time.time())}"

        await self._check_session_timeout()
        self.last_query_time = time.time()

        try:
            self.messages.append({"role": "user", "content": wrapped_message})
            logger.info(f"Query: {user_message}")

            iteration = 0

            while iteration < MAX_TOOL_ITERATIONS:
                iteration += 1
                logger.debug(f"Iteration {iteration} of tool calling loop")

                yield AgentEvent(type=EventType.THINKING, data={"iteration": iteration})

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )

                assistant_message = response.choices[0].message
                logger.debug(f"Assistant response: {assistant_message}")

                # Handle models that embed tool calls in content tags
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
                                    arguments=json.dumps(tool_data["arguments"]),
                                ),
                            )
                            formatted_tool_calls.append(tool_call)
                        assistant_message.tool_calls = formatted_tool_calls
                        assistant_message.content = None

                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls == []:
                    assistant_message.tool_calls = None
                self.messages.append(assistant_message.model_dump())

                # Execute tool calls
                if assistant_message.tool_calls:
                    logger.debug(f"Model requested {len(assistant_message.tool_calls)} tool calls")

                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        yield AgentEvent(
                            type=EventType.TOOL_CALL,
                            data={"tool": function_name, "args": function_args, "id": tool_call.id},
                        )

                        result = await self._execute_tool_call(tool_call)

                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={"tool": function_name, "result": result, "id": tool_call.id},
                        )

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })

                    continue

                # Final text response
                if assistant_message.content:
                    logger.info(f"Got final response after {iteration} iteration(s)")
                    yield AgentEvent(
                        type=EventType.DONE,
                        data={"content": assistant_message.content},
                    )
                    return

                logger.warning("Response had neither tool calls nor content")
                break

            if iteration >= MAX_TOOL_ITERATIONS:
                error_msg = "ERROR: Maximum tool calling iterations reached. The model may be stuck in a loop."
                logger.error(f"Hit maximum iterations ({MAX_TOOL_ITERATIONS}) in tool calling loop")
                yield AgentEvent(type=EventType.ERROR, data={"error": error_msg})
                return

            yield AgentEvent(type=EventType.ERROR, data={"error": "ERROR: No valid response generated"})

        except Exception as e:
            logger.error(f"Error in query: {e}\n{traceback.format_exc()}")
            yield AgentEvent(type=EventType.ERROR, data={"error": f"ERROR: {str(e)}"})

    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a single tool call via MCP"""
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.debug(f"Executing tool: {function_name} with args: {function_args}")

            result = await self.mcp_manager.execute_tool(function_name, function_args)
            return truncate_tool_result(str(result))

        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function.name}: {e}")
            return f"ERROR executing tool: {str(e)}"

    @staticmethod
    def _extract_tool_calls_from_content(content: str) -> Optional[list]:
        """Extract tool calls from content wrapped in <tool_call> tags."""
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


async def collect_response(orchestrator: AgentOrchestrator, user_message: str) -> str:
    """
    Helper to run the orchestrator and collect the final text response.
    Used by non-streaming endpoints for backward compatibility.
    """
    final_content = ""
    async for event in orchestrator.run(user_message):
        if event.type == EventType.DONE:
            final_content = event.data.get("content", "")
        elif event.type == EventType.ERROR:
            final_content = event.data.get("error", "ERROR: Unknown error")
    return final_content
