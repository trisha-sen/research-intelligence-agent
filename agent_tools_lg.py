import asyncio
from langchain_core.tools import tool

from agent_tools import (
    topic_trend       as _topic_trend,
    search_by_topic   as _search_by_topic,
    search_by_content as _search_by_content,
)

topic_trend       = tool(_topic_trend)
search_by_topic   = tool(_search_by_topic)
search_by_content = tool(_search_by_content)

TOOLS = [search_by_topic, search_by_content]


if __name__ == "__main__":
    async def _smoke_test():
        print("--- search_by_topic ---")
        results = await search_by_topic.ainvoke({"query": "federated learning privacy"})
        for r in results[:2]:
            print(r["title"], r["year"])

        print("\n--- search_by_content (hybrid) ---")
        results = await search_by_content.ainvoke({"query": "attention mechanism transformer"})
        for r in results[:2]:
            print(r["title"], r["year"])

        print("\n--- search_by_content (semantic) ---")
        results = await search_by_content.ainvoke({"query": "graph convolution node classification", "mode": "semantic"})
        for r in results[:2]:
            print(r["title"], r["year"])

    asyncio.run(_smoke_test())
