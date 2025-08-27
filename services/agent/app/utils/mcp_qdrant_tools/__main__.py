"""
Entry point for running as module: python -m utils.mcp_qdrant_tools.qdrant_mcp_server
"""
import asyncio
import sys

if __name__ == "__main__":
    # Import here to avoid circular imports and duplicate module warnings
    from .qdrant_mcp_server import main
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)