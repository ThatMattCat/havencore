# Tool Development Guide

This guide covers creating tools and integrations for HavenCore. Every agent tool is an MCP (Model Context Protocol) server — there is no in-process "legacy" tool registry.

## Overview

All of the agent's tools are provided by MCP servers: separate processes that advertise their tools over stdio and are wired into the LLM's function-calling interface via the `UnifiedTool` abstraction. To add a tool, add (or copy) an in-tree MCP module — see [Adding an MCP tool to HavenCore (in-tree)](#adding-an-mcp-tool-to-havencore-in-tree) below.

## Adding an MCP tool to HavenCore (in-tree)

The fastest path to a new tool is to copy one of the existing in-tree MCP
modules under `services/agent/selene_agent/modules/` and register it in
`.env`. The agent spawns each module as a subprocess at startup and discovers
its tools over stdio — no agent code changes needed.

Every in-tree module has the same three-file layout. The server file is named `mcp_server.py` by convention in the template modules, but it can have any name as long as `__main__.py` imports its `main()` — several in-tree modules use a different name (e.g. `face_mcp_server.py`, `github_mcp_server.py`, `qdrant_mcp_server.py`, `server.py`):

```
services/agent/selene_agent/modules/mcp_<your_tool>/
├── __init__.py          # empty is fine
├── __main__.py          # entry point — runs mcp_server.main()
└── mcp_server.py        # the server: tool list + handler
```

### 1. Scaffold the module

```bash
cd services/agent/selene_agent/modules
cp -r mcp_general_tools mcp_hello          # pick any existing module as your template
# prune the template's tools so you start empty
```

`__main__.py` is boilerplate — keep it identical:

```python
import asyncio

if __name__ == "__main__":
    from .mcp_server import main
    asyncio.run(main())
```

### 2. Define your tools in `mcp_server.py`

Minimal working server — one tool, no external deps:

```python
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

# mcp SDK 2.x dropped the low-level Server's @list_tools()/@call_tool()
# decorators; this shim restores the v1 decorator surface until the
# modules are rewritten onto the MCPServer decorator API.
from selene_agent.modules._mcp_compat import Server

server = Server("havencore-hello")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="say_hello",
            description="Greets the user by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"},
                },
                "required": ["name"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "say_hello":
        return [TextContent(type="text", text=f"Hello, {arguments['name']}!")]
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="havencore-hello",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )
```

The agent reads `TextContent.text` back out and hands it to the LLM as the
tool result. Return an error message the same way — the orchestrator will
show it in the dashboard's tool-call card.

### 3. Register the server in `.env`

Append an entry to the `MCP_SERVERS` JSON array:

```json
{
  "name": "hello",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_hello"],
  "enabled": true
}
```

All in-tree modules use the `python -m selene_agent.modules.<name>` form.
Keep it consistent — relative script paths break when the agent's working
directory changes.

### 4. Restart and smoke-test

```bash
docker compose restart agent

# From the host: confirm the server loaded and its tools are listed
curl http://localhost:6002/api/mcp/status | jq '.connected_servers, .tools_by_server'
```

Then run the stdio handshake from the **MCP server testing** section below
to exercise your new tool without going through the LLM.

### Tips

- **Read the agent's logs, not stdout.** Subprocess stderr is captured by
  `MCPClientManager`; write diagnostics with a logger rather than
  `print()` so they land in `docker compose logs agent`.
- **Wrap external calls in timeouts.** The orchestrator already caps
  `call_tool()` at `MCP_TOOL_TIMEOUT_SECONDS` (default 120s), but a tool
  that blocks on a hung socket burns that entire window.
- **A crashed server recovers itself.** Transport-level failures (broken or
  closed stream, EOF, dead process) mark the connection dead and trigger a
  bounded reconnect — `MCP_RECONNECT_MAX_ATTEMPTS` (default 3) tries with
  exponential backoff seeded by `MCP_RECONNECT_BACKOFF_SECONDS` (default 2s).
  Calls made while the server is down fail fast instead of waiting out the
  tool timeout. An ordinary tool error (a raised exception, or a result with
  `isError`) is *not* treated as a dead transport, so one failing tool never
  takes its sibling tools offline.
- **Return actionable errors.** The LLM recovers well from
  `"Error: city 'Foo' not found — try one of: [Rome, Foothill, ...]"`
  and poorly from `"Error: 404"`.

---

## MCP Server Development

### What is MCP?

Model Context Protocol (MCP) is a standard for connecting AI assistants to external tools and data sources. MCP servers run as separate processes and communicate with HavenCore through a standardized protocol.

**Benefits of MCP**:
- **Language agnostic**: Write servers in any language
- **Process isolation**: Servers run independently
- **Standardized interface**: Consistent tool interface

### Creating an MCP Server

#### 1. Setup MCP Server Project

```bash
# Create new MCP server directory
mkdir -p mcp-servers/file-manager
cd mcp-servers/file-manager

# Initialize Node.js project
npm init -y

# Install MCP SDK
npm install @modelcontextprotocol/sdk

# Create server file
touch server.js
```

#### 2. Implement MCP Server

```javascript
// server.js
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs/promises';
import path from 'path';

class FileManagerServer {
  constructor() {
    this.server = new Server(
      {
        name: "file-manager",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
  }

  setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: "list_files",
            description: "List files and directories in a given path",
            inputSchema: {
              type: "object",
              properties: {
                path: {
                  type: "string",
                  description: "Directory path to list (default: current directory)",
                  default: "."
                },
                show_hidden: {
                  type: "boolean", 
                  description: "Include hidden files (starting with .)",
                  default: false
                }
              }
            }
          },
          {
            name: "read_file",
            description: "Read contents of a text file",
            inputSchema: {
              type: "object",
              properties: {
                file_path: {
                  type: "string",
                  description: "Path to the file to read"
                }
              },
              required: ["file_path"]
            }
          },
          {
            name: "write_file",
            description: "Write content to a file",
            inputSchema: {
              type: "object",
              properties: {
                file_path: {
                  type: "string",
                  description: "Path to the file to write"
                },
                content: {
                  type: "string",
                  description: "Content to write to the file"
                },
                append: {
                  type: "boolean",
                  description: "Append to file instead of overwriting",
                  default: false
                }
              },
              required: ["file_path", "content"]
            }
          }
        ]
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case "list_files":
            return await this.listFiles(args.path || ".", args.show_hidden || false);
          
          case "read_file":
            return await this.readFile(args.file_path);
          
          case "write_file":
            return await this.writeFile(args.file_path, args.content, args.append || false);
          
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Error: ${error.message}`
            }
          ],
          isError: true
        };
      }
    });
  }

  async listFiles(dirPath, showHidden) {
    try {
      const files = await fs.readdir(dirPath, { withFileTypes: true });
      
      const fileList = [];
      for (const file of files) {
        if (!showHidden && file.name.startsWith('.')) {
          continue;
        }
        
        const filePath = path.join(dirPath, file.name);
        const stats = await fs.stat(filePath);
        
        fileList.push({
          name: file.name,
          type: file.isDirectory() ? 'directory' : 'file',
          size: stats.size,
          modified: stats.mtime.toISOString(),
          path: filePath
        });
      }
      
      return {
        content: [
          {
            type: "text",
            text: `Found ${fileList.length} items in ${dirPath}:\n\n` +
                  fileList.map(f => 
                    `${f.type === 'directory' ? '📁' : '📄'} ${f.name} (${f.size} bytes)`
                  ).join('\n')
          }
        ]
      };
      
    } catch (error) {
      throw new Error(`Failed to list files: ${error.message}`);
    }
  }

  async readFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      
      return {
        content: [
          {
            type: "text",
            text: `Content of ${filePath}:\n\n${content}`
          }
        ]
      };
      
    } catch (error) {
      throw new Error(`Failed to read file: ${error.message}`);
    }
  }

  async writeFile(filePath, content, append) {
    try {
      if (append) {
        await fs.appendFile(filePath, content);
      } else {
        await fs.writeFile(filePath, content);
      }
      
      return {
        content: [
          {
            type: "text",
            text: `Successfully ${append ? 'appended to' : 'wrote'} ${filePath}`
          }
        ]
      };
      
    } catch (error) {
      throw new Error(`Failed to write file: ${error.message}`);
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("File Manager MCP Server running on stdio");
  }
}

// Start the server
const server = new FileManagerServer();
server.run().catch(console.error);
```

#### 3. Configure MCP Server in HavenCore

Add to your `.env` file:

```bash
# Configure MCP servers (MCP is enabled by having entries here — no separate flag)
MCP_SERVERS='[
  {
    "name": "file-manager",
    "command": "node",
    "args": ["mcp-servers/file-manager/server.js"],
    "enabled": true
  }
]'
```

#### 4. Test MCP Server

```bash
# Restart HavenCore to load MCP server
docker compose restart agent

# Check MCP status
curl http://localhost:6002/mcp/status

# Test file operations via chat
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "List the files in the current directory"}
    ]
  }'
```

### Advanced MCP Patterns

#### Python MCP Server

> **Note:** The hand-rolled stdin/JSON loop below is illustrative only and is **not** how HavenCore connects to MCP servers. HavenCore uses the official `mcp` SDK `ClientSession`, which performs the full MCP handshake (`initialize` → `notifications/initialized`) before `tools/list`; a bare loop like this never completes that handshake and would fail to register. For a working Python server, use the `mcp.server.Server` + `stdio_server` pattern shown earlier under "Define your tools in `mcp_server.py`" and in every in-tree module.

```python
# python_mcp_server.py
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]

class PythonMCPServer:
    def __init__(self):
        self.tools = [
            Tool(
                name="python_calculator",
                description="Execute safe Python calculations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Python expression to evaluate (math operations only)"
                        }
                    },
                    "required": ["expression"]
                }
            )
        ]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get("method")
        params = request.get("params", {})

        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    }
                    for tool in self.tools
                ]
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "python_calculator":
                return await self.python_calculator(arguments.get("expression"))

            else:
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True
                }

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown method: {method}"}],
                "isError": True
            }

    async def python_calculator(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate Python mathematical expressions."""
        try:
            # Whitelist of safe operations
            allowed_names = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow,
                "int": int, "float": float, "str": str,
            }
            
            # Import safe math functions
            import math
            for name in dir(math):
                if not name.startswith('_'):
                    allowed_names[name] = getattr(math, name)

            # Evaluate expression safely
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Calculation: {expression} = {result}"
                    }
                ]
            }

        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": f"Calculation error: {str(e)}"
                    }
                ],
                "isError": True
            }

    async def run(self):
        """Run the MCP server using stdio transport."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                print(json.dumps(response))
                sys.stdout.flush()

            except Exception as e:
                error_response = {
                    "content": [{"type": "text", "text": f"Server error: {str(e)}"}],
                    "isError": True
                }
                print(json.dumps(error_response))
                sys.stdout.flush()

if __name__ == "__main__":
    server = PythonMCPServer()
    asyncio.run(server.run())
```

## Tool Registry Management

### Registered tools

Every agent tool is an MCP tool. List the current registry, grouped by MCP server, over HTTP:

```bash
# List all registered tools
curl http://localhost:6002/api/tools

# Response:
{
  "tools_by_server": {
    "homeassistant": [
      {
        "name": "ha_get_domain_entity_states",
        "description": "Get states for all entities in a domain",
        "parameters": {}
      }
    ],
    "general_tools": [
      {
        "name": "brave_search",
        "description": "Search the web with Brave",
        "parameters": {}
      }
    ]
  },
  "total": 68
}
```

Because there is no legacy tool source, there is no name-conflict resolution: tool names simply have to be unique across servers.

### Reloading MCP servers (requires restart)

MCP servers are loaded once from `MCP_SERVERS` at agent startup — adding or removing a server requires restarting the agent to pick up the new configuration:

```bash
# Update MCP_SERVERS in .env
MCP_SERVERS='[
  {
    "name": "file-manager",
    "command": "node",
    "args": ["mcp-servers/file-manager/server.js"],
    "enabled": true
  },
  {
    "name": "new-server",
    "command": "python",
    "args": ["mcp-servers/new-server/server.py"],
    "enabled": true
  }
]'

# Restart agent to pick up new configuration
docker compose restart agent
```

## Testing and Debugging Tools

### Unit tests

The agent's Python test suite lives in `services/agent/tests/` (pytest + pytest-asyncio, `asyncio_mode=auto`). Run it inside the agent container so imports and env resolve:

```bash
docker compose exec -T agent pytest
```

### MCP Server Testing

HavenCore's MCP servers run as subprocesses of the agent (see the `MCP_SERVERS`
JSON in `.env`). Each one is a Python module launched with
`python -m selene_agent.modules.<name>` and talks JSON-RPC over stdio. To
exercise one manually, run the same command inside the **agent container** —
that's where the Python deps, the loaded `.env`, and `shared_config` live.

Requires the agent container to be running (`docker compose up -d agent`).

#### Quick sanity check — does the server start and list tools?

Run the full MCP handshake (`initialize` → `notifications/initialized` → `tools/list`)
against the server:

```bash
docker compose exec -T agent python -m selene_agent.modules.mcp_homeassistant_tools <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-test","version":"0.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
EOF
```

Expect two JSON-RPC responses on stdout: the `initialize` result (protocol
version + server info) and the `tools/list` result (array of tool definitions
with `name`, `description`, `inputSchema`). Server logs go to stderr —
`docker compose exec` interleaves both, so pipe to `grep -E '^\{'` if you only
want the JSON.

Swap the module for any of the others:

| Server | Module |
|---|---|
| general_tools | `selene_agent.modules.mcp_general_tools` |
| homeassistant | `selene_agent.modules.mcp_homeassistant_tools` |
| plex | `selene_agent.modules.mcp_plex_tools` |
| music_assistant | `selene_agent.modules.mcp_music_assistant_tools` |
| mcp_server_qdrant | `selene_agent.modules.mcp_qdrant_tools` |
| mqtt | `selene_agent.modules.mcp_mqtt_tools` |
| face | `selene_agent.modules.mcp_face_tools` |
| vision | `selene_agent.modules.mcp_vision_tools` |
| github | `selene_agent.modules.mcp_github_tools` |
| reminder | `selene_agent.modules.mcp_reminder_tools` |
| device_action | `selene_agent.modules.mcp_device_action_tools` |

#### Exercising a specific tool

Add a `tools/call` message with the tool name and arguments. Important: if the
tool makes an external call (HA REST/WS, web API, vector DB), stdin EOF from a
heredoc can race the response and kill the subprocess mid-reply. Wrap the
input so stdin stays open until the tool finishes:

```bash
(cat <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-test","version":"0.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ha_list_services","arguments":{"domain":"light"}}}
EOF
sleep 3) | docker compose exec -T agent python -m selene_agent.modules.mcp_homeassistant_tools 2>&1 | grep -E '^\{'
```

The response wraps the tool's output in `result.content` (an array of
`{type, text}` blocks). `isError: false` means the transport succeeded — a
tool reporting its own failure will still come back with `isError: false` and
the error text inside `content`. Bump `sleep` higher for slow tools.

#### End-to-end via the chat API

To test a tool the way the LLM actually invokes it — including tool
selection, argument synthesis, and the follow-up response — POST a prompt
that forces the tool to `/v1/chat/completions` and inspect the tool-call
handling in the response.

### Debugging Tools

```python
# Add debugging to tools
import logging

logger = logging.getLogger(__name__)

def debug_tool_wrapper(func):
    """Decorator to add debugging to tools."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Tool {func.__name__} called with args: {args}, kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Tool {func.__name__} result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool {func.__name__} error: {e}")
            raise
    
    return wrapper

@debug_tool_wrapper
def my_custom_tool(param: str) -> Dict[str, Any]:
    # Tool implementation
    return {"result": param, "success": True}
```

---

**Next Steps**:
- [Agent Tools overview](README.md) - The MCP servers that make up the agent's tool surface
- [Home Assistant Integration](../../../integrations/home-assistant.md) - Smart home tool development
- [API Reference](../../../api-reference.md) - Building applications on HavenCore