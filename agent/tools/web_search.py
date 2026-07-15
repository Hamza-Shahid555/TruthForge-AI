import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_SCRIPT = Path(__file__).resolve().parent.parent.parent / "mcp_server" / "search_server.py"


async def get_mcp_tools():
    client = MultiServerMCPClient(
        {
            "web_search": {
                "command": "python",
                "args": [str(SERVER_SCRIPT)],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


if __name__ == "__main__":
    tools = asyncio.run(get_mcp_tools())
    print(f"Discovered {len(tools)} tool(s): {[t.name for t in tools]}")

    result = asyncio.run(tools[0].ainvoke({"query": "latest coffee health research 2026"}))
    print("\nSample result:")
    print(result)
