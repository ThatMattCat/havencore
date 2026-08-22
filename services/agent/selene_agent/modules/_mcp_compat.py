"""v1-style decorator surface for the mcp SDK 2.x low-level Server.

mcp 2.0 removed the ``@server.list_tools()`` / ``@server.call_tool()``
decorators from the low-level ``mcp.server.Server`` in favor of
constructor-registered handlers with ``(ctx, params)`` signatures and
wrapped result types (``ListToolsResult`` / ``CallToolResult``). The
in-tree tool modules all use the v1 decorator surface; this subclass
restores exactly that surface — including v1.1.2's behavior of turning a
handler exception into ``CallToolResult(is_error=True)`` with
``str(e)`` as the text — so the modules run unchanged under SDK 2.x
over stdio.

TEMPORARY bridge for the SDK migration: delete this module when the tool
modules are rewritten onto the ``mcp.server.MCPServer`` decorator API in a
later migration phase (the Streamable HTTP host serves the modules through
this same shim). New code must not import it.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Sequence

import mcp.types as types
from mcp.server import Server as _V2Server


class Server(_V2Server):
    """mcp 2.x low-level Server with the v1 tools decorators restored."""

    def list_tools(self):
        """v1-compatible ``@server.list_tools()`` decorator.

        The decorated coroutine takes no arguments and returns a bare
        ``list[types.Tool]``; the shim wraps it into ``ListToolsResult``.
        """

        def decorator(func: Callable[[], Awaitable[list[types.Tool]]]):
            async def handler(
                _ctx: Any, _params: types.PaginatedRequestParams
            ) -> types.ListToolsResult:
                tools = await func()
                return types.ListToolsResult(tools=list(tools))

            self.add_request_handler(
                "tools/list", types.PaginatedRequestParams, handler
            )
            return func

        return decorator

    def call_tool(self):
        """v1-compatible ``@server.call_tool()`` decorator.

        The decorated coroutine takes ``(name, arguments)`` and returns a
        bare content sequence; the shim wraps it into ``CallToolResult``.
        Exceptions become ``CallToolResult(is_error=True)`` exactly as in
        v1.1.2 (v2 would otherwise surface them as JSON-RPC errors).
        """

        def decorator(
            func: Callable[
                ...,
                Awaitable[
                    Sequence[
                        types.TextContent
                        | types.ImageContent
                        | types.EmbeddedResource
                    ]
                ],
            ],
        ):
            async def handler(
                _ctx: Any, params: types.CallToolRequestParams
            ) -> types.CallToolResult:
                try:
                    results = await func(params.name, (params.arguments or {}))
                    return types.CallToolResult(
                        content=list(results), is_error=False
                    )
                except Exception as e:
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=str(e))],
                        is_error=True,
                    )

            self.add_request_handler(
                "tools/call", types.CallToolRequestParams, handler
            )
            return func

        return decorator
