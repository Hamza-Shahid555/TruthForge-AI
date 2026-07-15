from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web-search")


@mcp.tool()
def web_search(query: str) -> str:
    """Search the live web for the given query and return a few short results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
