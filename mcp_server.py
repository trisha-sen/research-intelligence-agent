from mcp.server.fastmcp import FastMCP
from agent_tools import (
    search_abstracts,
    search_abstracts_by_topic,
    classify_topic,
    topic_trend,
)

mcp = FastMCP("research-agent")


@mcp.tool()
async def search(query: str, limit: int = 5) -> list[dict]:
    """Keyword search over 21K ML paper abstracts stored in PostgreSQL.
    Searches title and abstract (case-insensitive). Returns doi, title,
    abstract snippet, year, and NMF topic IDs."""
    return await search_abstracts(query=query, limit=limit)


@mcp.tool()
async def search_by_topic(topic_id: int, limit: int = 10) -> list[dict]:
    """Find papers by NMF topic ID (0-19), ranked by topic relevance then
    citation count. Use after classify to fetch papers in the top cluster."""
    return await search_abstracts_by_topic(topic_id=topic_id, limit=limit)


@mcp.tool()
def classify(text: str) -> dict:
    """Classify text into one of 20 NMF research topic clusters.
    Returns top_topic_id (0-19), top_topic_label, confidence, and all_weights."""
    return classify_topic(text=text)


@mcp.tool()
async def trend(topic_id: int) -> dict:
    """Annual paper fraction for an NMF topic cluster (topic_id 0-19).
    Returns topic_label and trend {year: fraction_of_all_papers_that_year}."""
    return await topic_trend(topic_id=topic_id)


if __name__ == "__main__":
    mcp.run()  # stdio transport - connect via Claude Desktop or any MCP client
