from dotenv import load_dotenv
import asyncio
import time
from langgraph.graph import StateGraph, START, END
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from anthropic import AsyncAnthropic

from agent_config import (
    ResearchState, CHAT_MODEL,
    SUMMARISE_SYSTEM,
    search_by_topic, search_by_content,
    SEARCH_TOKEN, SUMMARISE_TOKEN,
    SUMMARISE_TEMPERATURE
)
from mlflow_tracking import log_research_run

load_dotenv()

client = AsyncAnthropic()

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
    start_time = time.time()
    print(f"  [search] '{state['question'][:60]}...'")

    response = await client.messages.create(
        model=CHAT_MODEL,
        max_tokens=SEARCH_TOKEN,
        tools=SEARCH_TOOL_SCHEMAS,
        messages=[{"role": "user", "content": state["question"]}],
    )

    seen: set[str] = set()
    papers: list[dict] = []
    search_method = "unknown"
    search_params: dict = {}

    for block in response.content:
        if block.type == "tool_use":
            if search_method == "unknown":
                search_method = block.name
                search_params = dict(block.input)
            print(f"  [search] --> {block.name}({block.input})")
            results = await TOOL_DISPATCH[block.name](**block.input)
            for p in results:
                if p["doi"] not in seen:
                    seen.add(p["doi"])
                    papers.append(p)

    print(f"  [search] --> {len(papers)} papers retrieved")
    return {
        "search_results":   papers,
        "search_method":    search_method,
        "search_params":    search_params,
        "search_tokens_in": response.usage.input_tokens,
        "search_tokens_out":response.usage.output_tokens,
        "run_start_time":   start_time,
    }


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
        max_tokens=SUMMARISE_TOKEN,
        system=SUMMARISE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Research question: {state['question']}\n\n"
                    f"Retrieved papers:\n{papers_text}"
                )
            }
        ],
        temperature=SUMMARISE_TEMPERATURE,
    ) as stream:
        async for text in stream.text_stream:
            await adispatch_custom_event("token", text, config=config)
            chunks.append(text)
        final_msg = await stream.get_final_message()

    answer = "".join(chunks)
    print(f"  [summarise] --> done ({len(answer)} chars)")

    await asyncio.to_thread(
        log_research_run,
        {
            **state,
            "answer":               answer,
            "summarise_tokens_in":  final_msg.usage.input_tokens,
            "summarise_tokens_out": final_msg.usage.output_tokens,
        },
        time.time(),
    )

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
