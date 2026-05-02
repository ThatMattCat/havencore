"""
Entry point for running as module: python -m selene_agent.modules.mcp_vision_tools
"""
import asyncio
import sys

if __name__ == "__main__":
    from .server import main

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
