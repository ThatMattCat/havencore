"""
Tool Migration Helper for HavenCore
Assists in migrating legacy tools to MCP format and managing dual implementations
"""

import json
import asyncio
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import logging

import shared.scripts.logger as logger_module
logger = logger_module.get_logger('loki')


class MigrationStatus(Enum):
    """Status of a tool migration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    MIGRATED = "migrated"
    DEPRECATED = "deprecated"


@dataclass
class ToolMigration:
    """Represents a tool migration"""
    tool_name: str
    legacy_available: bool
    mcp_available: bool
    status: MigrationStatus
    mcp_server: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "legacy_available": self.legacy_available,
            "mcp_available": self.mcp_available,
            "status": self.status.value,
            "mcp_server": self.mcp_server,
            "notes": self.notes
        }


class ToolMigrationHelper:
    """Helper class for migrating tools from legacy to MCP"""
    
    def __init__(self, registry=None):
        self.registry = registry
        self.migrations: Dict[str, ToolMigration] = {}
        self.feature_flags: Dict[str, bool] = {}  # tool_name -> use_mcp
        
        # Load migration config from environment or file
        self._load_migration_config()
    
    def _load_migration_config(self):
        """Load migration configuration"""
        import os
        
        # Example: TOOL_MIGRATION_WEATHER=mcp
        # Example: TOOL_MIGRATION_BRAVE_SEARCH=legacy
        for key, value in os.environ.items():
            if key.startswith("TOOL_MIGRATION_"):
                tool_name = key.replace("TOOL_MIGRATION_", "").lower()
                use_mcp = value.lower() == "mcp"
                self.feature_flags[tool_name] = use_mcp
                logger.info(f"Tool migration config: {tool_name} -> {'MCP' if use_mcp else 'Legacy'}")
    
    def register_migration(self, tool_name: str, 
                          legacy_available: bool,
                          mcp_available: bool,
                          mcp_server: Optional[str] = None,
                          status: MigrationStatus = MigrationStatus.NOT_STARTED):
        """Register a tool migration"""
        
        migration = ToolMigration(
            tool_name=tool_name,
            legacy_available=legacy_available,
            mcp_available=mcp_available,
            status=status,
            mcp_server=mcp_server
        )
        
        self.migrations[tool_name] = migration
        logger.info(f"Registered migration for tool '{tool_name}': {migration.status.value}")
    
    def auto_detect_migrations(self):
        """Automatically detect migration opportunities from the registry"""
        if not self.registry:
            logger.warning("No registry available for auto-detection")
            return
        
        # Get registry status
        status = self.registry.get_registry_status()
        
        # Check all legacy tools
        for tool_name in status.get("legacy_tools", []):
            legacy_available = True
            mcp_available = tool_name in status.get("mcp_tools", [])
            
            # Determine migration status
            if mcp_available:
                # Tool exists in both - it's being migrated
                status_val = MigrationStatus.TESTING
                
                # Check if we're preferring MCP
                if self.feature_flags.get(tool_name, False):
                    status_val = MigrationStatus.MIGRATED
            else:
                # Only legacy exists
                status_val = MigrationStatus.NOT_STARTED
            
            # Find MCP server if available
            mcp_server = None
            if mcp_available and hasattr(self.registry, 'mcp_manager'):
                tool = self.registry.mcp_manager.get_tool_by_name(tool_name)
                if tool:
                    mcp_server = tool.server_name
            
            # Register the migration
            self.register_migration(
                tool_name=tool_name,
                legacy_available=legacy_available,
                mcp_available=mcp_available,
                mcp_server=mcp_server,
                status=status_val
            )
        
        # Check for MCP-only tools (new capabilities)
        for tool_name in status.get("mcp_tools", []):
            if tool_name not in status.get("legacy_tools", []):
                # MCP-only tool
                self.register_migration(
                    tool_name=tool_name,
                    legacy_available=False,
                    mcp_available=True,
                    mcp_server=self._find_mcp_server(tool_name),
                    status=MigrationStatus.MIGRATED
                )
    
    def _find_mcp_server(self, tool_name: str) -> Optional[str]:
        """Find which MCP server provides a tool"""
        if not self.registry or not hasattr(self.registry, 'mcp_manager'):
            return None
        
        tool = self.registry.mcp_manager.get_tool_by_name(tool_name)
        return tool.server_name if tool else None
    
    def set_tool_preference(self, tool_name: str, use_mcp: bool):
        """Set preference for a specific tool"""
        self.feature_flags[tool_name] = use_mcp
        
        # Update registry if available
        if self.registry:
            # This is a bit hacky but works for individual tools
            if tool_name in self.migrations:
                migration = self.migrations[tool_name]
                if use_mcp and migration.mcp_available:
                    migration.status = MigrationStatus.MIGRATED
                elif not use_mcp and migration.legacy_available:
                    migration.status = MigrationStatus.TESTING
                
                logger.info(f"Updated tool preference: {tool_name} -> {'MCP' if use_mcp else 'Legacy'}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get overall migration status"""
        
        # Count tools by status
        status_counts = {status: 0 for status in MigrationStatus}
        for migration in self.migrations.values():
            status_counts[migration.status] += 1
        
        # Calculate percentages
        total = len(self.migrations)
        percentages = {}
        if total > 0:
            percentages = {
                status.value: (count / total) * 100 
                for status, count in status_counts.items()
            }
        
        return {
            "total_tools": total,
            "status_counts": {s.value: c for s, c in status_counts.items()},
            "percentages": percentages,
            "migrations": [m.to_dict() for m in self.migrations.values()],
            "feature_flags": self.feature_flags
        }
    
    def get_tool_migration(self, tool_name: str) -> Optional[ToolMigration]:
        """Get migration status for a specific tool"""
        return self.migrations.get(tool_name)
    
    def should_use_mcp(self, tool_name: str) -> bool:
        """Determine if MCP version should be used for a tool"""
        
        # Check explicit feature flag first
        if tool_name in self.feature_flags:
            return self.feature_flags[tool_name]
        
        # Check migration status
        migration = self.migrations.get(tool_name)
        if migration:
            return migration.status == MigrationStatus.MIGRATED
        
        # Default to legacy
        return False
    
    def generate_migration_report(self) -> str:
        """Generate a human-readable migration report"""
        
        status = self.get_migration_status()
        
        report = []
        report.append("=" * 60)
        report.append("Tool Migration Report")
        report.append("=" * 60)
        report.append(f"\nTotal Tools: {status['total_tools']}")
        report.append("\nStatus Summary:")
        
        for status_name, count in status['status_counts'].items():
            percentage = status['percentages'].get(status_name, 0)
            report.append(f"  {status_name}: {count} ({percentage:.1f}%)")
        
        report.append("\nDetailed Status:")
        report.append("-" * 40)
        
        # Group by status
        by_status = {}
        for migration in self.migrations.values():
            if migration.status not in by_status:
                by_status[migration.status] = []
            by_status[migration.status].append(migration)
        
        for status in MigrationStatus:
            if status in by_status:
                report.append(f"\n{status.value.upper()}:")
                for migration in by_status[status]:
                    sources = []
                    if migration.legacy_available:
                        sources.append("Legacy")
                    if migration.mcp_available:
                        sources.append(f"MCP ({migration.mcp_server})")
                    
                    report.append(f"  • {migration.tool_name}")
                    report.append(f"    Sources: {', '.join(sources)}")
                    if migration.notes:
                        report.append(f"    Notes: {migration.notes}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def create_migration_plan(self, tool_name: str) -> Dict[str, Any]:
        """Create a migration plan for a specific tool"""
        
        migration = self.migrations.get(tool_name)
        if not migration:
            return {"error": f"No migration registered for tool '{tool_name}'"}
        
        plan = {
            "tool_name": tool_name,
            "current_status": migration.status.value,
            "steps": []
        }
        
        if migration.status == MigrationStatus.NOT_STARTED:
            plan["steps"] = [
                "1. Create MCP implementation in appropriate server",
                "2. Test MCP implementation independently",
                "3. Register tool in MCP server",
                "4. Update status to TESTING",
                "5. Run parallel testing",
                "6. Compare outputs between legacy and MCP",
                "7. Update status to MIGRATED when ready",
                "8. Deprecate legacy version after validation period"
            ]
        elif migration.status == MigrationStatus.TESTING:
            plan["steps"] = [
                "1. Continue parallel testing",
                "2. Monitor for discrepancies",
                "3. Validate edge cases",
                "4. Update feature flag to prefer MCP",
                "5. Monitor in production",
                "6. Update status to MIGRATED when stable"
            ]
        elif migration.status == MigrationStatus.MIGRATED:
            plan["steps"] = [
                "1. Monitor for any issues",
                "2. Consider deprecating legacy version",
                "3. Update documentation",
                "4. Remove legacy code in next major version"
            ]
        
        return plan


def generate_mcp_tool_wrapper(legacy_tool_def: Dict[str, Any], 
                              implementation: Callable) -> str:
    """Generate MCP server code for a legacy tool"""
    
    func_def = legacy_tool_def.get("function", {})
    tool_name = func_def.get("name", "unknown")
    description = func_def.get("description", "")
    parameters = func_def.get("parameters", {})
    
    # Generate the tool definition with proper imports and handler registration
    mcp_code = f'''\
# MCP Implementation for {tool_name}
from mcp import Tool, register_handler
from typing import Dict, Any

{tool_name}_tool = Tool(
    name="{tool_name}_mcp",
    description="{description}",
    inputSchema={json.dumps(parameters, indent=8)}
)

async def handle_{tool_name}(arguments: Dict[str, Any]) -> str:
    """MCP handler for {tool_name}"""
    # TODO: Implement the tool logic here
    # Original implementation can be referenced from the legacy version
    pass

# Register the handler with the MCP server
register_handler({tool_name}_tool, handle_{tool_name})
'''
    
    return mcp_code