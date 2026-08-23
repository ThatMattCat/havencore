if __name__ == "__main__":
    from .mcp_server import main
    main()  # synchronous: runs the async init/serve/disconnect wrapper via anyio
