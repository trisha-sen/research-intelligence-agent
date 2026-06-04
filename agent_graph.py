import os
from dotenv import load_dotenv
import asyncio
from typing import TypedDict
from anthropic import AsyncAnthropic
from langgraph.graph import StateGraph, START, END
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from agent_tools import search_by_topic, search_by_content

load_dotenv()

client = AsyncAnthropic()
CHAT_MODEL = "claude-haiku-4-5"

# -- State Definition -----------------------------------------------------------
class ResearchState(TypedDict):
    question:       str
    search_results: list[dict]
    answer:         str


# -- Tool schemas (Anthropic format) -------------------------------------------
SEARCH_TOOL_SCHEMAS = [
    {
        "name": "search_by_topic",
        "description": (
            "Classify the query using NMF and retrieve papers assigned to that "
            "topic cluster. Best for broad thematic questions like 'what research "
            "exists on federated learning' or 'papers about graph neural networks'. "
            "Uses pre-computed NMF topic assignments — fast and topic-coherent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The research question or topic"},
                "limit": {"type": "integer", "description": "Number of papers to return (default 8)", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_by_content",
        "description": (
            "Search abstract text directly using keyword matching, pgvector semantic "
            "similarity, or a hybrid of both. Best for specific method names, authors, "
            "conceptual questions that cross topic boundaries, or queries where exact "
            "wording matters. Mode: 'keyword', 'semantic', or 'hybrid' (default)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "mode":  {"type": "string", "enum": ["keyword", "semantic", "hybrid"], "default": "hybrid",
                          "description": "Search mode: keyword=exact match, semantic=vector similarity, hybrid=both"},
                "limit": {"type": "integer", "description": "Number of papers to return (default 8)", "default": 8},
            },
            "required": ["query"],
        },
    },
]

TOOL_DISPATCH = {
    "search_by_topic":   search_by_topic,
    "search_by_content": search_by_content,
}


# -- Nodes ---------------------------------------------------------------------

async def search_node(state: ResearchState) -> dict:
    """Let the LLM choose which search tool to use, then execute it."""
    print(f"  [search] '{state['question'][:60]}...'")

    response = await client.messages.create(
        model=CHAT_MODEL,
        max_tokens=256,
        tools=SEARCH_TOOL_SCHEMAS,
        messages=[{"role": "user", "content": state["question"]}],
    )

    seen: set[str] = set()
    papers: list[dict] = []

    for block in response.content:
        if block.type == "tool_use":
            print(f"  [search] --> {block.name}({block.input})")
            results = await TOOL_DISPATCH[block.name](**block.input)
            for p in results:
                if p["doi"] not in seen:
                    seen.add(p["doi"])
                    papers.append(p)

    print(f"  [search] --> {len(papers)} papers retrieved")
    return {"search_results": papers}


async def summarise_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Synthesise retrieved papers into a cited answer using the LLM."""
    papers = state.get("search_results", [])
    print(f"  [summarise] synthesising {len(papers)} papers...")

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


# -- Graph ---------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search",    search_node)
    graph.add_node("summarise", summarise_node)

    graph.add_edge(START,      "search")
    graph.add_edge("search",   "summarise")
    graph.add_edge("summarise", END)

    return graph.compile()


research_graph = build_graph()


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
