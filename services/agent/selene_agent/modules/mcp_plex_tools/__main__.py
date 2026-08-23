if __name__ == "__main__":
    from .mcp_server import main
    main()  # MCPServer.run("stdio") is synchronous (it owns the event loop)
