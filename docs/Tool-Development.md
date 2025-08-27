# Tool Development Guide

This guide covers creating custom tools and integrations for HavenCore, including both legacy tools and MCP (Model Context Protocol) servers.

## Overview

HavenCore supports two types of tools:
- **Legacy Tools**: Python functions directly integrated into the agent
- **MCP Tools**: External servers using Model Context Protocol for communication

Both types can coexist and are managed through the Unified Tool Registry.

## Legacy Tool Development

### Tool Architecture

Legacy tools consist of:
1. **Function Implementation**: Python function that performs the actual work
2. **Tool Definition**: JSON schema describing the tool's interface
3. **Registration**: Adding the tool to the agent's tool registry

### Creating a Basic Tool

#### 1. Implement the Function

Create or edit `services/agent/app/utils/custom_tools.py`:

```python
import requests
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

def get_cryptocurrency_price(
    symbol: str, 
    currency: str = "USD",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Get current cryptocurrency price and market data.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH', 'ADA')
        currency: Target currency for price (default: 'USD')
        api_key: Optional API key for rate limiting
        
    Returns:
        Dict containing price data and market information
    """
    try:
        # Use CoinGecko API (free tier)
        base_url = "https://api.coingecko.com/api/v3"
        endpoint = f"{base_url}/simple/price"
        
        params = {
            "ids": symbol.lower(),
            "vs_currencies": currency.lower(),
            "include_market_cap": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return {"error": f"Cryptocurrency '{symbol}' not found"}
            
        coin_data = list(data.values())[0]
        
        return {
            "symbol": symbol.upper(),
            "currency": currency.upper(),
            "price": coin_data.get(f"{currency.lower()}"),
            "market_cap": coin_data.get(f"{currency.lower()}_market_cap"),
            "24h_change": coin_data.get(f"{currency.lower()}_24h_change"),
            "last_updated": datetime.fromtimestamp(
                coin_data.get("last_updated_at", 0)
            ).isoformat(),
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {"error": f"Failed to fetch cryptocurrency data: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in get_cryptocurrency_price: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


def calculate_tip(
    bill_amount: float, 
    tip_percentage: float = 15.0,
    split_ways: int = 1
) -> Dict[str, Any]:
    """Calculate tip and split bill among multiple people.
    
    Args:
        bill_amount: Total bill amount
        tip_percentage: Tip percentage (default: 15%)
        split_ways: Number of people to split bill (default: 1)
        
    Returns:
        Dict containing tip calculation details
    """
    try:
        if bill_amount <= 0:
            return {"error": "Bill amount must be greater than 0"}
            
        if tip_percentage < 0:
            return {"error": "Tip percentage cannot be negative"}
            
        if split_ways < 1:
            return {"error": "Must split among at least 1 person"}
            
        tip_amount = bill_amount * (tip_percentage / 100)
        total_amount = bill_amount + tip_amount
        per_person = total_amount / split_ways
        tip_per_person = tip_amount / split_ways
        
        return {
            "bill_amount": round(bill_amount, 2),
            "tip_percentage": tip_percentage,
            "tip_amount": round(tip_amount, 2),
            "total_amount": round(total_amount, 2),
            "split_ways": split_ways,
            "per_person_total": round(per_person, 2),
            "per_person_tip": round(tip_per_person, 2),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in calculate_tip: {e}")
        return {"error": f"Calculation error: {str(e)}"}
```

#### 2. Define Tool Schema

Add tool definitions to `services/agent/app/utils/general_tools_defs.py`:

```python
# Cryptocurrency price tool definition
cryptocurrency_price_tool = {
    "type": "function",
    "function": {
        "name": "get_cryptocurrency_price",
        "description": "Get current price and market data for cryptocurrencies like Bitcoin, Ethereum, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Cryptocurrency symbol (e.g., BTC, ETH, ADA, DOGE)"
                },
                "currency": {
                    "type": "string",
                    "description": "Target currency for price (e.g., USD, EUR, GBP)",
                    "default": "USD"
                }
            },
            "required": ["symbol"]
        }
    }
}

# Tip calculator tool definition
tip_calculator_tool = {
    "type": "function", 
    "function": {
        "name": "calculate_tip",
        "description": "Calculate tip amount and split bill among multiple people",
        "parameters": {
            "type": "object",
            "properties": {
                "bill_amount": {
                    "type": "number",
                    "description": "Total bill amount in dollars"
                },
                "tip_percentage": {
                    "type": "number",
                    "description": "Tip percentage (e.g., 15 for 15%)",
                    "default": 15.0
                },
                "split_ways": {
                    "type": "integer",
                    "description": "Number of people to split the bill among",
                    "default": 1
                }
            },
            "required": ["bill_amount"]
        }
    }
}
```

#### 3. Register Tools with Agent

Update `services/agent/app/selene_agent.py`:

```python
from utils.custom_tools import get_cryptocurrency_price, calculate_tip
from utils.general_tools_defs import cryptocurrency_price_tool, tip_calculator_tool

class SeleneAgent:
    def _setup_tool_functions(self) -> Dict[str, callable]:
        """Map tool names to their implementation functions"""
        return {
            # Existing tools...
            'home_assistant.get_domain_entity_states': self.haos.get_domain_entity_states,
            'brave_search': self.brave_search,
            'wolfram_alpha': self.wolfram_alpha,
            
            # New custom tools
            'get_cryptocurrency_price': get_cryptocurrency_price,
            'calculate_tip': calculate_tip,
        }
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tool definitions"""
        return [
            # Existing tools...
            haos_get_domain_entity_states_tool,
            brave_search_tool,
            wolfram_alpha_tool,
            
            # New custom tools
            cryptocurrency_price_tool,
            tip_calculator_tool,
        ]
```

#### 4. Add Environment Configuration

Add any required API keys to `.env.tmpl`:

```bash
# Cryptocurrency API Configuration (optional)
CRYPTO_API_KEY=""  # CoinGecko Pro API key for higher rate limits
```

#### 5. Test the Tools

```bash
# Restart agent to load new tools
docker compose restart agent

# Test via API
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "What is the current price of Bitcoin?"}
    ]
  }'

# Test tip calculator
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo", 
    "messages": [
      {"role": "user", "content": "Calculate a 20% tip on a $85 bill split 4 ways"}
    ]
  }'
```

### Advanced Tool Patterns

#### Asynchronous Tools

For I/O intensive operations, create async tools:

```python
import asyncio
import aiohttp
from typing import Dict, Any

async def async_web_scraper(url: str, selector: str = None) -> Dict[str, Any]:
    """Asynchronously scrape content from a web page.
    
    Args:
        url: URL to scrape
        selector: Optional CSS selector for specific content
        
    Returns:
        Dict containing scraped content
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return {"error": f"HTTP {response.status}: {response.reason}"}
                
                html = await response.text()
                
                # Use BeautifulSoup for parsing
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                if selector:
                    elements = soup.select(selector)
                    content = [elem.get_text().strip() for elem in elements]
                else:
                    # Extract title and meta description
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No title"
                    
                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    desc_text = meta_desc.get('content', '') if meta_desc else ""
                    
                    content = {
                        "title": title_text,
                        "description": desc_text
                    }
                
                return {
                    "url": url,
                    "content": content,
                    "success": True
                }
                
    except Exception as e:
        logger.error(f"Web scraping error: {e}")
        return {"error": f"Scraping failed: {str(e)}"}

# Register async tool (requires special handling in agent)
def web_scraper_sync_wrapper(url: str, selector: str = None) -> Dict[str, Any]:
    """Synchronous wrapper for async web scraper tool."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_web_scraper(url, selector))
    finally:
        loop.close()
```

#### Stateful Tools

Tools that maintain state between calls:

```python
class TaskManager:
    """Tool for managing a simple task list."""
    
    def __init__(self):
        self.tasks = []
        self.next_id = 1
    
    def add_task(self, description: str, priority: str = "medium") -> Dict[str, Any]:
        """Add a new task to the list."""
        task = {
            "id": self.next_id,
            "description": description,
            "priority": priority,
            "completed": False,
            "created_at": datetime.now().isoformat()
        }
        
        self.tasks.append(task)
        self.next_id += 1
        
        return {
            "task": task,
            "total_tasks": len(self.tasks),
            "success": True
        }
    
    def complete_task(self, task_id: int) -> Dict[str, Any]:
        """Mark a task as completed."""
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = datetime.now().isoformat()
                return {"task": task, "success": True}
        
        return {"error": f"Task with ID {task_id} not found"}
    
    def list_tasks(self, show_completed: bool = True) -> Dict[str, Any]:
        """List all tasks, optionally filtering out completed ones."""
        if show_completed:
            filtered_tasks = self.tasks
        else:
            filtered_tasks = [t for t in self.tasks if not t["completed"]]
        
        return {
            "tasks": filtered_tasks,
            "total": len(filtered_tasks),
            "success": True
        }

# Create global instance
task_manager = TaskManager()

# Tool functions that use the stateful manager
def add_task(description: str, priority: str = "medium") -> Dict[str, Any]:
    return task_manager.add_task(description, priority)

def complete_task(task_id: int) -> Dict[str, Any]:
    return task_manager.complete_task(task_id)

def list_tasks(show_completed: bool = True) -> Dict[str, Any]:
    return task_manager.list_tasks(show_completed)
```

#### Database-Integrated Tools

Tools that interact with the PostgreSQL database:

```python
import asyncpg
from shared.configs.shared_config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, 
    POSTGRES_USER, POSTGRES_PASSWORD
)

class DatabaseTool:
    """Tool for interacting with the conversation database."""
    
    def __init__(self):
        self.connection_string = (
            f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
            f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations."""
        try:
            conn = await asyncpg.connect(self.connection_string)
            
            # Total conversations
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM conversation_histories"
            )
            
            # Recent conversations (last 7 days)
            recent = await conn.fetchval(
                "SELECT COUNT(*) FROM conversation_histories "
                "WHERE created_at > NOW() - INTERVAL '7 days'"
            )
            
            # Average messages per conversation
            avg_messages = await conn.fetchval(
                "SELECT AVG((metadata->>'message_count')::int) "
                "FROM conversation_histories "
                "WHERE metadata->>'message_count' IS NOT NULL"
            )
            
            await conn.close()
            
            return {
                "total_conversations": total,
                "recent_conversations": recent,
                "average_messages": round(float(avg_messages or 0), 1),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return {"error": f"Database error: {str(e)}"}

# Create tool instance
db_tool = DatabaseTool()

def get_conversation_statistics() -> Dict[str, Any]:
    """Get conversation statistics from the database."""
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(db_tool.get_conversation_stats())
    finally:
        loop.close()
```

### Error Handling Best Practices

#### Comprehensive Error Handling

```python
import traceback
from typing import Dict, Any, Optional

def robust_api_tool(
    endpoint: str, 
    api_key: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """Template for robust API tool with comprehensive error handling."""
    
    try:
        # Input validation
        if not endpoint:
            return {"error": "Endpoint URL is required"}
        
        if not endpoint.startswith(('http://', 'https://')):
            return {"error": "Invalid URL format"}
        
        # API call with proper error handling
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.get(
            endpoint, 
            headers=headers, 
            timeout=timeout
        )
        
        # Handle different HTTP status codes
        if response.status_code == 200:
            try:
                data = response.json()
                return {"data": data, "success": True}
            except ValueError as e:
                return {"error": f"Invalid JSON response: {str(e)}"}
                
        elif response.status_code == 401:
            return {"error": "Authentication failed - check API key"}
            
        elif response.status_code == 403:
            return {"error": "Access forbidden - insufficient permissions"}
            
        elif response.status_code == 404:
            return {"error": "Endpoint not found"}
            
        elif response.status_code == 429:
            return {"error": "Rate limit exceeded - try again later"}
            
        else:
            return {
                "error": f"HTTP {response.status_code}: {response.reason}"
            }
            
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed - check network connectivity"}
        
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {timeout} seconds"}
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
        
    except Exception as e:
        # Log full traceback for debugging
        logger.error(f"Unexpected error in robust_api_tool: {traceback.format_exc()}")
        return {"error": f"Unexpected error: {str(e)}"}
```

#### Logging and Debugging

```python
import logging
from functools import wraps

def log_tool_execution(func):
    """Decorator to log tool execution details."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool args: {args}, kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            
            if isinstance(result, dict) and result.get("success"):
                logger.info(f"Tool {tool_name} completed successfully")
            elif isinstance(result, dict) and "error" in result:
                logger.warning(f"Tool {tool_name} failed: {result['error']}")
            
            logger.debug(f"Tool {tool_name} result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Tool {tool_name} crashed: {str(e)}")
            return {"error": f"Tool execution failed: {str(e)}"}
    
    return wrapper

# Usage
@log_tool_execution
def my_custom_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """Custom tool with automatic logging."""
    # Tool implementation here
    return {"result": "success", "success": True}
```

---

## MCP Server Development

### What is MCP?

Model Context Protocol (MCP) is a standard for connecting AI assistants to external tools and data sources. MCP servers run as separate processes and communicate with HavenCore through a standardized protocol.

**Benefits of MCP**:
- **Language agnostic**: Write servers in any language
- **Process isolation**: Servers run independently
- **Standardized interface**: Consistent tool interface
- **Hot reloading**: Add/remove servers without restarting HavenCore

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
# Enable MCP support
MCP_ENABLED=true

# Configure MCP servers
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

### Unified Tool Registry

The Unified Tool Registry manages both legacy and MCP tools:

```python
# Example of checking tool status
curl http://localhost:6002/tools/status

# Response:
{
  "total_tools": 15,
  "legacy_tools": 8,
  "mcp_tools": 7,
  "conflicts": ["weather_tool"],  # Tools with same name from both sources
  "tool_preference": "legacy",    # Which version is used for conflicts
  "tools": [
    {
      "name": "get_weather_forecast",
      "source": "legacy",
      "description": "Get weather forecast for a location"
    },
    {
      "name": "list_files", 
      "source": "mcp",
      "server_name": "file-manager",
      "description": "List files and directories"
    }
  ]
}
```

### Managing Tool Conflicts

When both legacy and MCP tools have the same name:

```bash
# Set preference for MCP tools over legacy
curl -X POST http://localhost:6002/tools/preference \
  -H "Content-Type: application/json" \
  -d '{"prefer_mcp": true}'

# Set preference for legacy tools (default)
curl -X POST http://localhost:6002/tools/preference \
  -H "Content-Type: application/json" \
  -d '{"prefer_mcp": false}'
```

### Dynamic Tool Loading

MCP servers can be added/removed without restarting HavenCore:

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

### Unit Testing Tools

```python
# test_custom_tools.py
import pytest
from unittest.mock import Mock, patch
from utils.custom_tools import get_cryptocurrency_price, calculate_tip

class TestCryptocurrencyTool:
    @patch('requests.get')
    def test_successful_price_fetch(self, mock_get):
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "bitcoin": {
                "usd": 45000.00,
                "usd_market_cap": 850000000000,
                "usd_24h_change": 2.5,
                "last_updated_at": 1640995200
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_cryptocurrency_price("BTC")

        assert result["success"] is True
        assert result["symbol"] == "BTC"
        assert result["price"] == 45000.00
        assert result["24h_change"] == 2.5

    @patch('requests.get')
    def test_api_error_handling(self, mock_get):
        # Mock API error
        mock_get.side_effect = Exception("Network error")

        result = get_cryptocurrency_price("BTC")

        assert "error" in result
        assert "Network error" in result["error"]

class TestTipCalculator:
    def test_basic_calculation(self):
        result = calculate_tip(100.0, 15.0, 1)
        
        assert result["success"] is True
        assert result["bill_amount"] == 100.0
        assert result["tip_amount"] == 15.0
        assert result["total_amount"] == 115.0
        assert result["per_person_total"] == 115.0

    def test_split_bill(self):
        result = calculate_tip(100.0, 20.0, 4)
        
        assert result["success"] is True
        assert result["total_amount"] == 120.0
        assert result["per_person_total"] == 30.0
        assert result["per_person_tip"] == 5.0

    def test_invalid_inputs(self):
        # Test negative bill amount
        result = calculate_tip(-10.0)
        assert "error" in result

        # Test zero split
        result = calculate_tip(100.0, 15.0, 0)
        assert "error" in result

# Run tests
# docker compose exec agent python -m pytest test_custom_tools.py -v
```

### Integration Testing

```python
# test_tool_integration.py
import requests
import json

class TestToolIntegration:
    def setup_method(self):
        self.base_url = "http://localhost"
        self.headers = {
            "Authorization": "Bearer your_api_key",
            "Content-Type": "application/json"
        }

    def test_cryptocurrency_tool_integration(self):
        """Test cryptocurrency tool through chat API."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "What is the current price of Ethereum?"}
            ]
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        
        # Check that tool was called and price information is in response
        assistant_message = data["choices"][0]["message"]["content"]
        assert "ethereum" in assistant_message.lower() or "eth" in assistant_message.lower()

    def test_tip_calculator_integration(self):
        """Test tip calculator through chat API.""" 
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Calculate a 18% tip on a $75 bill for 3 people"}
            ]
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", 
            headers=self.headers,
            json=payload
        )

        assert response.status_code == 200
        data = response.json()
        
        assistant_message = data["choices"][0]["message"]["content"]
        # Should contain calculation results
        assert "75" in assistant_message  # Original bill
        assert "18" in assistant_message  # Tip percentage
        assert "3" in assistant_message   # Number of people
```

### MCP Server Testing

```bash
# Test MCP server directly
echo '{"method": "tools/list", "params": {}}' | node mcp-servers/file-manager/server.js

# Test tool call
echo '{"method": "tools/call", "params": {"name": "list_files", "arguments": {"path": "."}}}' | node mcp-servers/file-manager/server.js
```

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
- [MCP Integration](MCP-Integration.md) - Deep dive into Model Context Protocol
- [Home Assistant Integration](Home-Assistant-Integration.md) - Smart home tool development
- [API Development](API-Development.md) - Building applications on HavenCore