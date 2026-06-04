import asyncio
from langchain_core.tools import tool

from agent_tools import (
    search_abstracts          as _search_abstracts,
    search_abstracts_by_topic as _search_abstracts_by_topic,
    classify_topic            as _classify_topic,
    topic_trend               as _topic_trend,
)

search_abstracts          = tool(_search_abstracts)
search_abstracts_by_topic = tool(_search_abstracts_by_topic)
classify_topic            = tool(_classify_topic)
topic_trend               = tool(_topic_trend)

TOOLS = [
    search_abstracts, search_abstracts_by_topic,
    classify_topic, topic_trend
]


if __name__ == "__main__":
    async def _smoke_test():
        print("--- classify_topic ---")
        print(classify_topic.invoke({"text": "federated learning privacy"}))

        print("\n--- search_abstracts ---")
        results = await search_abstracts.ainvoke({"query": "attention mechanism", "limit": 2})
        for r in results:
            print(r["title"], r["year"])

        print("\n--- search_abstracts_by_topic ---")
        results = await search_abstracts_by_topic.ainvoke({"topic_id": 12, "limit": 2})
        for r in results:
            print(r["title"], r["year"])

        print("\n--- topic_trend ---")
        trend = await topic_trend.ainvoke({"topic_id": 12})
        print(trend["topic_label"], trend["trend"])

    asyncio.run(_smoke_test())
