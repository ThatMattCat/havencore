"""
MCP Client Manager for HavenCore
Manages connections to MCP servers and provides tool discovery and execution
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback
import sys
import os

from mcp import StdioServerParameters, ClientSession, Tool
from mcp.client.stdio import stdio_client

from selene_agent.utils import logger as custom_logger
from selene_agent.utils import config

logger = custom_logger.get_logger('loki')


# --- Transport-failure classification -------------------------------------
#
# A tool that ran and failed is NOT the same thing as a dead subprocess.
# Tearing a connection down because one tool legitimately errored would take
# every other tool on that server with it, so the classifier below is
# deliberately narrow: only stream/pipe/process-level failures count.

try:  # anyio ships with mcp; the guard keeps this module importable regardless
    import anyio as _anyio

    _TRANSPORT_EXC_TYPES: Tuple[type, ...] = (
        _anyio.BrokenResourceError,
        _anyio.ClosedResourceError,
        _anyio.EndOfStream,
    )
except Exception:  # pragma: no cover - anyio is a hard dep of mcp
    _TRANSPORT_EXC_TYPES = ()

_TRANSPORT_EXC_TYPES = _TRANSPORT_EXC_TYPES + (
    BrokenPipeError,
    ConnectionError,  # covers ConnectionReset/Aborted/Refused
    EOFError,
    ProcessLookupError,
)

# Name-based fallback so a differently-imported anyio (or a future rename)
# still classifies correctly.
_TRANSPORT_EXC_NAMES = frozenset({
    "BrokenResourceError",
    "ClosedResourceError",
    "EndOfStream",
    "BrokenPipeError",
    "ConnectionResetError",
    "ProcessLookupError",
    "IncompleteRead",
})

try:
    from mcp.shared.exceptions import McpError as _McpError
except Exception:  # pragma: no cover
    _McpError = None  # type: ignore[assignment]


def _iter_exception_chain(exc: BaseException, _depth: int = 0):
    """Yield `exc` plus its explicit causes and any ExceptionGroup members.

    MCP runs its session over anyio task groups, so a dead pipe often reaches
    us wrapped in an ExceptionGroup or chained behind another exception.
    """
    if exc is None or _depth > 8:
        return
    yield exc
    for member in getattr(exc, "exceptions", ()) or ():
        yield from _iter_exception_chain(member, _depth + 1)
    # Only `__cause__` (explicit `raise ... from ...`), never `__context__`:
    # implicit context is set for *any* exception raised inside an except
    # block, which would let an unrelated tool error inherit a transport
    # verdict it did not earn.
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        yield from _iter_exception_chain(cause, _depth + 1)


def is_transport_error(exc: BaseException) -> bool:
    """True only when `exc` means the MCP stdio transport/subprocess is gone.

    Explicitly False for `McpError`: that is a JSON-RPC error the server
    *answered* with (or a client-side read timeout), which proves the
    subprocess is alive. Ordinary tool exceptions fall through to False too —
    only broken/closed streams, EOF and dead-process errors return True.
    """
    if _McpError is not None and isinstance(exc, _McpError):
        return False
    for item in _iter_exception_chain(exc):
        if _McpError is not None and isinstance(item, _McpError):
            continue
        if isinstance(item, _TRANSPORT_EXC_TYPES):
            return True
        if type(item).__name__ in _TRANSPORT_EXC_NAMES:
            return True
    return False


class ToolSource(Enum):
    """Enum to identify the source of a tool"""
    LEGACY = "legacy"
    MCP = "mcp"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def to_stdio_params(self) -> StdioServerParameters:
        """Convert to StdioServerParameters for MCP client"""
        return StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )


@dataclass
class UnifiedTool:
    """Unified representation of a tool from either source"""
    name: str
    description: str
    parameters: Dict[str, Any]
    source: ToolSource
    server_name: Optional[str] = None  # For MCP tools
    original_tool: Optional[Any] = None  # Store original MCP Tool object
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class MCPServerConnection:
    """Manages a single MCP server connection"""
    
    def __init__(self, server_name: str, mcp_config: MCPServerConfig):
        self.server_name = server_name
        self.mcp_config = mcp_config
        self.client_session: Optional[ClientSession] = None
        self.read_stream = None
        self.write_stream = None
        self._context_manager = None
        # Transport liveness. The ClientSession object stays truthy long after
        # its subprocess dies, so health is tracked separately.
        self._alive = True
        self.failure_reason: Optional[str] = None

    async def connect(self):
        """Connect to the MCP server"""
        try:
            # Merge any per-server env overrides (from the MCP_SERVERS config)
            # OVER the inherited environment, rather than discarding them.
            self.mcp_config.env = {**os.environ, **(self.mcp_config.env or {})}
            stdio_params = self.mcp_config.to_stdio_params()
            
            self._context_manager = stdio_client(stdio_params)
            
            streams = await self._context_manager.__aenter__()
            self.read_stream, self.write_stream = streams
            
            self.client_session = ClientSession(self.read_stream, self.write_stream)
            self.client_session = await self.client_session.__aenter__()
            
            await self.client_session.initialize()
            
            logger.info(f"Connected to MCP server: {self.server_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_name}: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.client_session:
            try:
                await self.client_session.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error exiting client session: {e}")
        
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
                logger.debug(f"Disconnected from MCP server: {self.server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {self.server_name}: {e}")
    
    def mark_dead(self, reason: str):
        """Flag this connection's transport as gone. Idempotent."""
        if self._alive:
            self._alive = False
            self.failure_reason = reason
            logger.error(
                f"MCP server {self.server_name} transport is dead: {reason}"
            )

    async def close_transport(self):
        """Best-effort teardown of a dead transport, safe from any task.

        Deliberately does NOT call `__aexit__` on the stdio/session context
        managers: those were entered inside the per-server connect task, and
        anyio refuses to exit a cancel scope from a different task (see issue
        #68). Closing the memory streams is task-agnostic and is enough to let
        the reader/writer tasks wind down; the contexts themselves are still
        unwound by `MCPClientManager.cleanup()` at lifespan shutdown, exactly
        as they are today.
        """
        self._alive = False
        for stream in (self.write_stream, self.read_stream):
            if stream is None:
                continue
            try:
                await stream.aclose()
            except Exception as e:
                logger.debug(
                    f"Error closing stream for {self.server_name}: {e}"
                )

    def is_connected(self) -> bool:
        """Check if connected.

        A non-None `client_session` is not proof of a live subprocess — it
        stays truthy forever after `connect()`. Transport failures observed
        during tool execution flip `_alive`, and that is reflected here so
        `/mcp/status` stops reporting dead servers as connected.
        """
        return self.client_session is not None and self._alive


class MCPClientManager:
    """Manages MCP client connections and tool discovery"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.connections: Dict[str, MCPServerConnection] = {}
        self.mcp_tools: Dict[str, UnifiedTool] = {}  # tool_name -> UnifiedTool
        self.server_tools: Dict[str, List[str]] = {}  # server_name -> [tool_names]
        self.failed_servers: Dict[str, str] = {}  # server_name -> error message
        self._initialized = False
        # Bounded reconnect state for servers whose subprocess died mid-uptime.
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}
        self._reconnect_attempts: Dict[str, int] = {}  # per-outage attempt count
        # Connections replaced by a reconnect. Kept so `cleanup()` can still
        # unwind their MCP contexts at shutdown (see close_transport docstring).
        self._stale_connections: List[MCPServerConnection] = []
        self.reconnect_max_attempts = config.MCP_RECONNECT_MAX_ATTEMPTS
        self.reconnect_backoff_seconds = config.MCP_RECONNECT_BACKOFF_SECONDS
        self.reconnect_timeout_seconds = config.MCP_RECONNECT_TIMEOUT_SECONDS

    def add_server(self, mcp_config: MCPServerConfig):
        """Add an MCP server configuration"""
        if mcp_config.enabled:
            self.servers[mcp_config.name] = mcp_config
            logger.info(f"Added MCP server configuration: {mcp_config.name}")
    
    async def initialize(self, connection_timeout: float = 30.0):
        """Initialize connections to all configured MCP servers"""
        if self._initialized:
            logger.debug("MCP Client Manager already initialized")
            return
            
        logger.info("Initializing MCP Client Manager...")
        
        # Create connection tasks for all servers
        connection_tasks = []
        server_names = []
        
        for server_name, server_config in self.servers.items():
            logger.info(f"Preparing connection to MCP server: {server_name}")
            task = asyncio.create_task(
                self._connect_to_server_with_timeout(server_name, server_config, connection_timeout),
                name=f"connect_{server_name}"
            )
            connection_tasks.append(task)
            server_names.append(server_name)
        
        # Execute all connections concurrently
        if connection_tasks:
            logger.info(f"Connecting to {len(connection_tasks)} MCP servers concurrently...")
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            # Process results
            successful_connections = 0
            for i, (server_name, result) in enumerate(zip(server_names, results)):
                if isinstance(result, Exception):
                    error_msg = str(result)
                    logger.error(f"Failed to connect to MCP server {server_name}: {error_msg}")
                    self.failed_servers[server_name] = error_msg
                else:
                    successful_connections += 1
                    logger.info(f"Successfully connected to MCP server: {server_name}")
        
        self._initialized = True
        logger.info(f"MCP Client Manager initialized with {len(self.connections)} active servers")
    
    async def _connect_to_server_with_timeout(self, server_name: str, config: MCPServerConfig, timeout: float):
        """Connect to a single MCP server with timeout protection"""
        try:
            await asyncio.wait_for(
                self._connect_to_server(server_name, config),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            error_msg = f"Connection to MCP server {server_name} timed out after {timeout}s"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_name}: {e}")
            raise

    async def _connect_to_server(self, server_name: str, config: MCPServerConfig):
        """Connect to a single MCP server"""
        try:
            # Create a connection object
            connection = MCPServerConnection(server_name, config)
            logger.info(f"Created connection object for MCP server: {server_name}")
            # Connect to the server
            await connection.connect()
            logger.info(f"Connected to MCP server: {server_name}")
            # Store the connection
            self.connections[server_name] = connection
            
            # Discover tools from this server
            await self._discover_server_tools(server_name, connection.client_session)
            
            logger.info(f"Successfully connected to MCP server: {server_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_name}: {e}")
            raise
    
    def _schedule_reconnect(self, server_name: str):
        """Kick a bounded background reconnect for a server that dropped.

        At most one reconnect task per server, and at most
        `reconnect_max_attempts` attempts per outage — a broken server module
        must not turn every subsequent tool call into a respawn storm.
        """
        task = self._reconnect_tasks.get(server_name)
        if task is not None and not task.done():
            return  # already in flight
        if self._reconnect_attempts.get(server_name, 0) >= self.reconnect_max_attempts:
            logger.debug(
                f"Not reconnecting to MCP server {server_name}: "
                f"{self.reconnect_max_attempts} attempts already exhausted"
            )
            return
        try:
            self._reconnect_tasks[server_name] = asyncio.create_task(
                self._reconnect_server(server_name),
                name=f"mcp_reconnect_{server_name}",
            )
        except RuntimeError as e:  # no running loop
            logger.error(
                f"Could not schedule MCP reconnect for {server_name}: {e}"
            )

    async def _reconnect_server(self, server_name: str):
        """Tear down a dead connection and re-run `_connect_to_server`.

        Bounded: `reconnect_max_attempts` tries with exponential backoff, then
        the server is recorded in `failed_servers` and left alone until the
        next successful reconnect resets the budget.
        """
        server_config = self.servers.get(server_name)
        if server_config is None:
            logger.error(f"Cannot reconnect unknown MCP server: {server_name}")
            return

        # Retire the dead connection before respawning so tool calls fail fast
        # while the reconnect is in flight.
        old = self.connections.pop(server_name, None)
        if old is not None:
            await old.close_transport()
            self._stale_connections.append(old)

        while self._reconnect_attempts.get(server_name, 0) < self.reconnect_max_attempts:
            attempt = self._reconnect_attempts.get(server_name, 0) + 1
            self._reconnect_attempts[server_name] = attempt

            if attempt > 1 and self.reconnect_backoff_seconds > 0:
                delay = self.reconnect_backoff_seconds * (2 ** (attempt - 2))
                logger.info(
                    f"Waiting {delay:.1f}s before MCP reconnect attempt "
                    f"{attempt} for {server_name}"
                )
                await asyncio.sleep(delay)

            logger.info(
                f"Reconnecting to MCP server {server_name} "
                f"(attempt {attempt}/{self.reconnect_max_attempts})"
            )
            try:
                await asyncio.wait_for(
                    self._connect_to_server(server_name, server_config),
                    timeout=self.reconnect_timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    f"MCP reconnect attempt {attempt} for {server_name} failed: {e}"
                )
                continue

            self._reconnect_attempts[server_name] = 0
            self.failed_servers.pop(server_name, None)
            logger.info(f"Reconnected to MCP server: {server_name}")
            return

        reason = (
            f"Reconnect abandoned after {self.reconnect_max_attempts} attempts"
        )
        self.failed_servers[server_name] = reason
        logger.error(
            f"MCP server {server_name}: {reason}. Its tools stay unavailable "
            "until the agent is restarted."
        )

    async def _discover_server_tools(self, server_name: str, session: ClientSession):
        """Discover and register tools from an MCP server"""
        try:
            # List available tools
            tools_response = await session.list_tools()
            tools = tools_response.tools if tools_response.tools else []
            
            tool_names = []
            for tool in tools:
                # Convert MCP tool to UnifiedTool
                unified_tool = self._convert_mcp_tool(tool, server_name)
                
                # Register the tool
                self.mcp_tools[unified_tool.name] = unified_tool
                tool_names.append(unified_tool.name)
                
                logger.debug(f"Discovered MCP tool: {unified_tool.name} from {server_name}")
            
            # Track which tools belong to which server
            self.server_tools[server_name] = tool_names
            
            logger.info(f"Discovered {len(tool_names)} tools from MCP server {server_name}")
            
        except Exception as e:
            logger.error(f"Error discovering tools from {server_name}: {e}")
    
    def _convert_mcp_tool(self, mcp_tool: Tool, server_name: str) -> UnifiedTool:
        """Convert an MCP Tool to UnifiedTool format"""
        # Parse the input schema to OpenAI format
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        if mcp_tool.inputSchema:
            # MCP uses JSON Schema, which is compatible with OpenAI format
            if isinstance(mcp_tool.inputSchema, dict):
                parameters = mcp_tool.inputSchema
            else:
                # If it's a string, try to parse it
                try:
                    parameters = json.loads(mcp_tool.inputSchema)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Could not parse input schema for tool {mcp_tool.name}: {e}"
                    )
        
        return UnifiedTool(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            parameters=parameters,
            source=ToolSource.MCP,
            server_name=server_name,
            original_tool=mcp_tool
        )
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool"""
        if tool_name not in self.mcp_tools:
            raise ValueError(f"MCP tool '{tool_name}' not found")
        
        tool = self.mcp_tools[tool_name]
        server_name = tool.server_name
        
        # Fail fast on a server we already know is down: calling into a dead
        # pipe would otherwise burn the full MCP_TOOL_TIMEOUT_SECONDS per call.
        connection = self.connections.get(server_name)
        if connection is None or not connection.is_connected():
            self._schedule_reconnect(server_name)
            raise ValueError(f"MCP server '{server_name}' is not connected")

        client_session = connection.client_session
        
        timeout = config.MCP_TOOL_TIMEOUT_SECONDS
        try:
            logger.debug(f"Executing MCP tool {tool_name} with args: {arguments}")

            # Call the tool through MCP, with a hard cap so a wedged server
            # cannot block the orchestrator indefinitely.
            result = await asyncio.wait_for(
                client_session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Handle the result based on its type (CallToolResult)
            if hasattr(result, 'isError') and result.isError:
                # Error case - extract text from content
                error_parts = []
                if hasattr(result, 'content') and isinstance(result.content, list):
                    for item in result.content:
                        if hasattr(item, 'text'):
                            error_parts.append(item.text)
                        else:
                            error_parts.append(str(item))
                error_msg = f"MCP tool error: {' '.join(error_parts) if error_parts else 'Unknown error'}"
                logger.error(error_msg)
                return error_msg
            
            # Extract content from successful result
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    # Handle multiple content items
                    content_parts = []
                    for item in result.content:
                        # Check for different content types based on MCP spec
                        if hasattr(item, 'text'):
                            # TextContent
                            content_parts.append(item.text)
                        elif hasattr(item, 'data') and hasattr(item, 'mimeType'):
                            # ImageContent
                            content_parts.append(f"[Image: {item.mimeType}]")
                        elif hasattr(item, 'resource'):
                            # EmbeddedResource
                            content_parts.append(f"[Resource: {item.resource}]")
                        else:
                            content_parts.append(str(item))
                    return '\n'.join(content_parts)
                else:
                    return str(result.content)
            else:
                return str(result)
                
        except asyncio.TimeoutError:
            # A slow or wedged tool is not proof of a dead transport, so the
            # connection is deliberately left alone here.
            error_msg = (
                f"MCP tool '{tool_name}' on server '{server_name}' did not "
                f"return within {timeout:.0f}s and was cancelled. The tool may be "
                "hung or the request was too large; you can retry with simpler inputs."
            )
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            if is_transport_error(e):
                detail = str(e) or type(e).__name__
                connection.mark_dead(detail)
                self._schedule_reconnect(server_name)
                error_msg = (
                    f"MCP server '{server_name}' connection lost while executing "
                    f"'{tool_name}': {detail}. Reconnecting in the background; "
                    "retry this tool shortly."
                )
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                return error_msg

            # An ordinary tool failure: the server is still alive, so the
            # connection (and every other tool on it) must survive.
            error_msg = f"Error executing MCP tool {tool_name}: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return error_msg
    
    def get_all_mcp_tools(self) -> List[UnifiedTool]:
        """Get all discovered MCP tools"""
        return list(self.mcp_tools.values())
    
    def get_tool_by_name(self, tool_name: str) -> Optional[UnifiedTool]:
        """Get a specific MCP tool by name"""
        return self.mcp_tools.get(tool_name)
    
    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool is an MCP tool"""
        return tool_name in self.mcp_tools
    
    async def cleanup(self):
        """Clean up all MCP connections"""
        logger.info("Cleaning up MCP connections...")

        # Stop any in-flight reconnect before tearing connections down.
        pending = [t for t in self._reconnect_tasks.values() if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._reconnect_tasks.clear()
        self._reconnect_attempts.clear()

        # Connections retired by a reconnect still hold un-exited MCP contexts.
        for connection in list(self.connections.values()) + self._stale_connections:
            try:
                await connection.disconnect()
            except Exception as e:
                logger.error(
                    f"Error closing MCP server {connection.server_name}: {e}"
                )

        self._stale_connections.clear()
        self.connections.clear()
        self.mcp_tools.clear()
        self.server_tools.clear()
        self._initialized = False
        
        logger.info("MCP cleanup complete")
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status information about MCP servers.

        `connected_servers` reflects real transport health, so a server whose
        subprocess died drops out of it instead of being reported as healthy.
        """
        return {
            "configured_servers": list(self.servers.keys()),
            "connected_servers": [
                name for name, conn in self.connections.items()
                if conn.is_connected()
            ],
            "disconnected_servers": {
                name: (conn.failure_reason or "connection lost")
                for name, conn in self.connections.items()
                if not conn.is_connected()
            },
            "reconnecting_servers": [
                name for name, task in self._reconnect_tasks.items()
                if not task.done()
            ],
            "failed_servers": self.failed_servers,
            "total_mcp_tools": len(self.mcp_tools),
            "tools_by_server": {
                server: len(tools)
                for server, tools in self.server_tools.items()
            },
        }