# Home Assistant MCP Server

This document describes the new unified Home Assistant MCP server that consolidates all Home Assistant functionality into a single MCP server.

## Overview

The Home Assistant MCP server (`havencore_homeassistant_mcp_server.py`) provides a unified interface to all Home Assistant tools through the Model Context Protocol (MCP). This replaces the previous legacy tool registration system with a more modular and maintainable approach.

## Tools Provided

### Basic Home Assistant API Tools

1. **ha_get_domain_entity_states**
   - Get current states of all entities in a Home Assistant domain
   - Parameters: `domain` (string) - e.g., 'light', 'switch', 'sensor'
   - Returns: JSON formatted entity states

2. **ha_get_domain_services**
   - Get available services for a Home Assistant domain
   - Parameters: `domain` (string) - domain to get services for
   - Returns: JSON formatted service descriptions

3. **ha_execute_service**
   - Execute a Home Assistant service on an entity
   - Parameters: 
     - `entity_id` (string) - e.g., 'light.living_room'
     - `service` (string) - e.g., 'turn_on', 'turn_off', 'toggle'
   - Returns: Execution result message

### Media Player Control Tools

4. **ha_control_media_player**
   - Control media player playback, volume, power, and sources
   - Parameters:
     - `action` (string) - One of: play, pause, stop, toggle, next, previous, seek, shuffle, repeat, volume_set, volume_up, volume_down, mute, unmute, turn_on, turn_off, select_source
     - `device` (string, optional) - Device name or entity_id
     - `value` (number/string/boolean, optional) - Action-specific value
   - Returns: JSON formatted operation result

5. **ha_get_media_player_statuses**
   - Get status information about Home Assistant media players
   - Parameters: None
   - Returns: JSON formatted player statuses

6. **ha_play_media**
   - Play a specific media item on a device from the media library
   - Parameters:
     - `media_item_id` (string) - Media ID or title to play
     - `playback_device_id` (string, optional) - Target device
   - Returns: JSON formatted playback result

7. **ha_find_media_items**
   - Search for media items in the Home Assistant media library
   - Parameters:
     - `query` (string) - Search query
     - `query_type` (string, optional) - 'title', 'genre', or 'year'
     - `media_type` (string, optional) - 'video', 'audio', 'image', or 'playlist'
     - `limit` (integer, optional) - Max results (default: 5)
   - Returns: Formatted search results

## Configuration

The MCP server is configured through the `MCP_SERVERS` environment variable in the `.env` file:

```json
{
  "name": "homeassistant",
  "command": "/usr/bin/python",
  "args": ["./utils/havencore_homeassistant_mcp_server.py"],
  "enabled": true
}
```

## Migration from Legacy Tools

When `MCP_PREFER_OVER_LEGACY` is set to `true`, the agent will:

1. Skip registration of legacy Home Assistant tools
2. Use the MCP server tools instead
3. Maintain the same functionality through the unified tool registry

Legacy tools are still available as fallback when:
- `MCP_ENABLED` is `false`
- `MCP_PREFER_OVER_LEGACY` is `false`
- MCP server initialization fails

## Error Handling

The MCP server includes robust error handling:

- **Graceful Degradation**: Falls back to mock implementations if dependencies are missing
- **Exception Handling**: All tool executions are wrapped in try-catch blocks
- **Connection Resilience**: Handles Home Assistant connection failures gracefully

## Mock Mode

When Home Assistant dependencies are not available, the server operates in mock mode:

- Provides simulated responses for all tools
- Allows testing without a real Home Assistant instance
- Maintains the same tool interfaces

## Dependencies

Required dependencies:
- `mcp` - Model Context Protocol library
- `homeassistant_api` - Home Assistant API client (optional, uses mock if missing)
- `aiohttp` - For WebSocket connections (optional)

The server gracefully handles missing dependencies and provides mock implementations for testing.

## Testing

Test the MCP server directly:

```bash
cd services/agent/app
echo '{"method": "tools/list", "id": 1}' | python utils/havencore_homeassistant_mcp_server.py
```

Test through the agent system by enabling MCP in your `.env` file and using the unified tool registry.