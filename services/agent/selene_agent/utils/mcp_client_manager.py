"""
MCP Client Manager for HavenCore
Manages connections to MCP servers and provides tool discovery and execution
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import traceback
import sys
import os

from mcp import StdioServerParameters, ClientSession, Tool
from mcp.client.stdio import stdio_client  # Fixed import

logger = logging.getLogger(__name__)


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
        
    async def connect(self):
        """Connect to the MCP server"""
        try:
            # if "python" in self.mcp_config.command.lower() or "sys.executable" in self.mcp_config.command.lower():
            #     self.mcp_config.command = sys.executable
            # set "self.config.env" to the current environment variables
            self.mcp_config.env = dict(os.environ)
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
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.client_session is not None


class MCPClientManager:
    """Manages MCP client connections and tool discovery"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.connections: Dict[str, MCPServerConnection] = {}
        self.mcp_tools: Dict[str, UnifiedTool] = {}  # tool_name -> UnifiedTool
        self.server_tools: Dict[str, List[str]] = {}  # server_name -> [tool_names]
        self._initialized = False
        
    def add_server(self, mcp_config: MCPServerConfig):
        """Add an MCP server configuration"""
        if mcp_config.enabled:
            self.servers[mcp_config.name] = mcp_config
            logger.info(f"Added MCP server configuration: {mcp_config.name}")
    
    async def initialize(self, connection_timeout: float = 600.0):
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
                    logger.error(f"Failed to connect to MCP server {server_name}: {result}")
                    logger.debug(traceback.format_exc())
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
                except:
                    logger.warning(f"Could not parse input schema for tool {mcp_tool.name}")
        
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
        
        if server_name not in self.connections:
            raise ValueError(f"MCP server '{server_name}' not connected")
        
        connection = self.connections[server_name]
        if not connection.is_connected():
            raise ValueError(f"MCP server '{server_name}' is not connected")
        
        client_session = connection.client_session
        
        try:
            logger.debug(f"Executing MCP tool {tool_name} with args: {arguments}")
            
            # Call the tool through MCP
            result = await client_session.call_tool(tool_name, arguments)
            
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
                
        except Exception as e:
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
        
        for server_name, connection in self.connections.items():
            try:
                await connection.disconnect()
            except Exception as e:
                logger.error(f"Error closing MCP server {server_name}: {e}")
        
        self.connections.clear()
        self.mcp_tools.clear()
        self.server_tools.clear()
        self._initialized = False
        
        logger.info("MCP cleanup complete")
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status information about MCP servers"""
        return {
            "configured_servers": list(self.servers.keys()),
            "connected_servers": [
                name for name, conn in self.connections.items() 
                if conn.is_connected()
            ],
            "total_mcp_tools": len(self.mcp_tools),
            "tools_by_server": {
                server: len(tools) 
                for server, tools in self.server_tools.items()
            }
        }