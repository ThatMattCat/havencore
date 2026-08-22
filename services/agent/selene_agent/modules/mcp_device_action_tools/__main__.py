if __name__ == "__main__":
    from .mcp_server import main  # Import here to avoid circular imports
    main()  # MCPServer.run("stdio") is synchronous (it owns the event loop)
