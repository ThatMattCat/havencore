"""Shared helpers for typed MCP tool signatures (mcp 2.0 ``MCPServer`` modules).

The decorator API generates each tool's ``inputSchema`` from the function
signature via pydantic. Optional parameters are declared as the bare type
with a ``None`` default::

    body: Annotated[str, NULL_OK, Field(description="...")] = None

so the generated property keeps the plain ``{"type": "string"}`` shape of the
hand-written-schema era instead of pydantic's ``anyOf [T, null]`` union.
``NULL_OK`` then keeps validation from rejecting an explicit JSON ``null``,
which LLMs routinely send for parameters they mean to omit: the null passes
straight through to the tool function, which treats it as "not provided" —
exactly like the old dict-``get`` dispatch did. It contributes nothing to the
generated schema.
"""
from typing import Any

from pydantic import WrapValidator


def _none_passthrough(value: Any, handler: Any) -> Any:
    """Skip a field's type validation entirely for an explicit null."""
    return None if value is None else handler(value)


#: Annotated-metadata marker for optional tool params: accept explicit null.
NULL_OK = WrapValidator(_none_passthrough)
