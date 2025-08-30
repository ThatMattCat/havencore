import asyncio

if __name__ == "__main__":
    from .mcp_server import main  # Import here to avoid circular imports
    asyncio.run(main())