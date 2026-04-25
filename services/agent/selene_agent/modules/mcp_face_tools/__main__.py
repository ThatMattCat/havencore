"""
Entry point for running as module: python -m selene_agent.modules.mcp_face_tools
"""
import asyncio
import sys

if __name__ == "__main__":
    from .face_mcp_server import main

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)
