if __name__ == "__main__":
    from .mcp_server import main  # Import here to avoid circular imports

    main()  # synchronous: runs the async init/serve wrapper via anyio
