import os
from dotenv import load_dotenv
import asyncio
from typing import TypedDict
from anthropic import AsyncAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from agent_tools import search_abstracts_by_topic, search_abstracts_hybrid, classify_topic

load_dotenv()

client = AsyncAnthropic()
CHAT_MODEL = "claude-haiku-4-5"

# -- State Definition -----------------------------------------------------------
class ResearchState(TypedDict):
    question:        str          # the user's research question, set at start
    topic_id:        int          # filled by classify node
    topic_label:     str          # filled by classify node
    topic_results:   list[dict]   # filled by search_by_topic_node
    keyword_results: list[dict]   # filled by search_by_keyword_node
    answer:          str          # filled by summarise node


async def classify_node(state: ResearchState) -> dict:
    """Classify the research question into one of 20 NMF topic clusters."""
    print(f"  [classify] '{state['question'][:60]}...'")

    result = classify_topic(state["question"])

    print(f"  [classify] --> {result['top_topic_label']} (confidence: {result['confidence']})")

    return {
        "topic_id":    result["top_topic_id"],
        "topic_label": result["top_topic_label"],
    }


async def search_by_topic_node(state: ResearchState) -> dict:
    """Retrieve papers whose NMF topic list contains the classified topic ID."""
    print(f"  [search_by_topic] topic: {state['topic_label']} (id: {state['topic_id']})")

    papers = await search_abstracts_by_topic(topic_id=state["topic_id"], limit=8)

    print(f"  [search_by_topic] --> {len(papers)} papers retrieved")
    return {"topic_results": papers}


async def search_hybrid_node(state: ResearchState) -> dict:
    """Hybrid search: keyword + vector similarity combined in one SQL query."""
    print(f"  [search_hybrid] query: '{state['question'][:60]}...'")

    papers = await search_abstracts_hybrid(query=state["question"], limit=8)

    print(f"  [search_hybrid] --> {len(papers)} papers retrieved")
    return {"keyword_results": papers}

async def summarise_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Synthesise retrieved papers into a cited answer using the LLM."""
    # Merge and deduplicate by DOI; topic results first (higher relevance)
    seen = set()
    papers = []
    for p in state.get("topic_results", []) + state.get("keyword_results", []):
        if p["doi"] not in seen:
            seen.add(p["doi"])
            papers.append(p)

    print(f"  [summarise] synthesising {len(papers)} papers ({len(state.get('topic_results', []))} topic + {len(state.get('keyword_results', []))} keyword, {len(papers)} unique)...")

    papers_text = "\n\n".join([
        f"Title: {p['title']} ({p['year']})\n"
        f"DOI: {p['doi']}\n"
        f"Abstract: {p['abstract']}"
        for p in papers
    ])

    chunks: list[str] = []

    async with client.messages.stream(
        model=CHAT_MODEL,
        max_tokens=2048,
        system=(
            "You are a research assistant. Synthesise the provided paper abstracts "
            "into a clear, structured answer to the research question. "
            "Cite papers inline as (Author et al., YEAR) where the author is inferred "
            "from the title. End with a References section listing: Title · DOI · Year. "
            "Be concise - 3 to 5 paragraphs."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research question: {state['question']}\n\n"
                    f"Topic area: {state['topic_label']}\n\n"
                    f"Retrieved papers:\n{papers_text}"
                )
            }
        ],
        temperature=0.3,
    ) as stream:
        async for text in stream.text_stream:
            await adispatch_custom_event("token", text, config=config)
            chunks.append(text)

    answer = "".join(chunks)
    print(f"  [summarise] --> done ({len(answer)} chars)")

    return {"answer": answer}

# -- 3. Build the graph ---------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("classify",      classify_node)
    graph.add_node("search_topic",  search_by_topic_node)
    graph.add_node("search_hybrid", search_hybrid_node)
    graph.add_node("summarise",     summarise_node)

    # classify fans out to both searches in parallel, then merge into summarise
    graph.add_edge(START,           "classify")
    graph.add_edge("classify",      "search_topic")
    graph.add_edge("classify",      "search_hybrid")
    graph.add_edge("search_topic",  "summarise")
    graph.add_edge("search_hybrid", "summarise")
    graph.add_edge("summarise",     END)

    return graph.compile()


# -- 4. Run it ------------------------------------------------------------------

research_graph = build_graph()

# async def ask(question: str) -> str:
#     """Entry point - takes a question, returns a cited answer."""
#     final_state = await research_graph.ainvoke(
#         {"question": question}
#     )
#     return final_state["answer"]


async def ask_stream(question: str):
    """Streaming entry point - drives the full graph and yields LLM token chunks."""
    async for event in research_graph.astream_events(
        {"question": question},
        version="v2",
    ):
        if event["event"] == "on_custom_event" and event["name"] == "token":
            yield event["data"]


if __name__ == "__main__":
    async def main():
        question = "What are the main privacy approaches in federated learning?"
        print(f"Question: {question}\n" + "="*60)
        async for chunk in ask_stream(question):
            print(chunk, end="", flush=True)
        print()

    asyncio.run(main())