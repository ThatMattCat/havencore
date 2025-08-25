"""
Unified Tool Registry for HavenCore
Manages both legacy and MCP tools in a single registry
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
import logging
import inspect

from utils.mcp_client_manager import UnifiedTool, ToolSource, MCPClientManager

import shared.scripts.logger as logger_module
logger = logger_module.get_logger('loki')


class UnifiedToolRegistry:
    """Registry that manages both legacy and MCP tools"""
    
    def __init__(self, mcp_manager: Optional[MCPClientManager] = None):
        self.mcp_manager = mcp_manager
        self.legacy_tools: Dict[str, UnifiedTool] = {}
        self.legacy_functions: Dict[str, Callable] = {}
        self.tool_priority = {}  # tool_name -> priority (lower is higher priority)
        
        # Configuration for tool source preference
        self.prefer_mcp = False  # By default, prefer legacy tools
        
    def register_legacy_tool(self, 
                           tool_def: Dict[str, Any], 
                           function: Callable,
                           priority: int = 100):
        """Register a legacy tool with its implementation"""
        func_def = tool_def.get("function", {})
        tool_name = func_def.get("name")
        
        if not tool_name:
            logger.error(f"Invalid tool definition: missing name")
            return
        
        unified_tool = UnifiedTool(
            name=tool_name,
            description=func_def.get("description", ""),
            parameters=func_def.get("parameters", {}),
            source=ToolSource.LEGACY,
            server_name=None
        )
        
        self.legacy_tools[tool_name] = unified_tool
        self.legacy_functions[tool_name] = function
        self.tool_priority[tool_name] = priority
        
        logger.debug(f"Registered legacy tool: {tool_name}")
    
    def register_legacy_tools_bulk(self, 
                                 tool_defs: List[Dict[str, Any]], 
                                 functions: Dict[str, Callable]):
        """Register multiple legacy tools at once"""
        for tool_def in tool_defs:
            func_name = tool_def.get("function", {}).get("name")
            if func_name and func_name in functions:
                self.register_legacy_tool(tool_def, functions[func_name])
    
    async def get_all_tools(self) -> List[UnifiedTool]:
        """Get all available tools from both sources"""
        all_tools = []
        
        all_tools.extend(self.legacy_tools.values())
        
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_all_mcp_tools()
            
            # Handle conflicts based on preference
            for mcp_tool in mcp_tools:
                if mcp_tool.name in self.legacy_tools:
                    # Conflict detected
                    if self.prefer_mcp:
                        all_tools = [t for t in all_tools if t.name != mcp_tool.name]
                        all_tools.append(mcp_tool)
                        logger.debug(f"Tool conflict for '{mcp_tool.name}': using MCP version")
                    else:
                        logger.debug(f"Tool conflict for '{mcp_tool.name}': using legacy version")
                else:
                    # No conflict, add MCP tool
                    all_tools.append(mcp_tool)
        
        logger.info(f"Total available tools: {len(all_tools)} "
                   f"(Legacy: {len(self.legacy_tools)}, "
                   f"MCP: {len(self.mcp_manager.get_all_mcp_tools()) if self.mcp_manager else 0})")
        
        return all_tools
    
    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI format for LLM"""
        all_tools = await self.get_all_tools()
        return [tool.to_openai_format() for tool in all_tools]
    
    def get_tool_by_name(self, tool_name: str) -> Optional[UnifiedTool]:
        """Get a specific tool by name, respecting preferences"""
        # Check if we have both versions
        has_legacy = tool_name in self.legacy_tools
        has_mcp = self.mcp_manager and self.mcp_manager.is_mcp_tool(tool_name)
        
        if has_legacy and has_mcp:
            # Conflict - use preference
            if self.prefer_mcp:
                return self.mcp_manager.get_tool_by_name(tool_name)
            else:
                return self.legacy_tools[tool_name]
        elif has_legacy:
            return self.legacy_tools[tool_name]
        elif has_mcp:
            return self.mcp_manager.get_tool_by_name(tool_name)
        else:
            return None
    
    def get_tool_source(self, tool_name: str) -> Optional[ToolSource]:
        """Get the source of a tool"""
        tool = self.get_tool_by_name(tool_name)
        return tool.source if tool else None
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name, routing to appropriate handler"""
        tool = self.get_tool_by_name(tool_name)
        
        if not tool:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.error(error_msg)
            return error_msg
        
        logger.debug(f"Executing tool '{tool_name}' from source: {tool.source.value}")
        
        if tool.source == ToolSource.LEGACY:
            return await self._execute_legacy_tool(tool_name, arguments)
        elif tool.source == ToolSource.MCP:
            return await self._execute_mcp_tool(tool_name, arguments)
        else:
            return f"Unknown tool source: {tool.source}"
    
    async def _execute_legacy_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a legacy tool"""
        if tool_name not in self.legacy_functions:
            return f"Legacy tool '{tool_name}' has no implementation"
        
        try:
            func = self.legacy_functions[tool_name]
            
            # Check if the function is async
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**arguments))
            
            return str(result)
            
        except Exception as e:
            error_msg = f"Error executing legacy tool {tool_name}: {e}"
            logger.error(error_msg)
            return error_msg
    
    async def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute an MCP tool"""
        if not self.mcp_manager:
            return "MCP manager not available"
        
        return await self.mcp_manager.execute_tool(tool_name, arguments)
    
    def set_tool_preference(self, prefer_mcp: bool):
        """Set whether to prefer MCP tools over legacy when both exist"""
        self.prefer_mcp = prefer_mcp
        logger.info(f"Tool preference set to: {'MCP' if prefer_mcp else 'Legacy'}")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status information about the registry"""
        status = {
            "legacy_tools": list(self.legacy_tools.keys()),
            "legacy_tool_count": len(self.legacy_tools),
            "prefer_mcp": self.prefer_mcp,
            "has_mcp_manager": self.mcp_manager is not None
        }
        
        if self.mcp_manager:
            mcp_status = self.mcp_manager.get_server_status()
            status.update({
                "mcp_tools": list(self.mcp_manager.mcp_tools.keys()),
                "mcp_tool_count": mcp_status["total_mcp_tools"],
                "mcp_servers": mcp_status
            })
        
        # Find conflicts
        if self.mcp_manager:
            conflicts = []
            for tool_name in self.legacy_tools:
                if self.mcp_manager.is_mcp_tool(tool_name):
                    conflicts.append(tool_name)
            status["tool_conflicts"] = conflicts
            status["conflict_count"] = len(conflicts)
        
        return status