from typing import TypedDict
from agent_tools import classify_topic, search_abstracts_by_topic, search_abstracts_hybrid

CHAT_MODEL = "claude-haiku-4-5"
SEARCH_TOKEN = 2048
SUMMARISE_TOKEN = 1024
SUMMARISE_TEMPERATURE = 0.3
SUMMARISE_SYSTEM = (
    "You are a research assistant. Synthesise the provided paper abstracts "
    "into a clear, structured answer to the research question. "
    "Cite papers inline as (Author et al., YEAR) where the author is inferred "
    "from the title. End with a References section listing: Title · DOI · Year. "
    "Be concise - 3 to 5 paragraphs."
)

# -- State Definition -----------------------------------------------------------
class _ResearchStateRequired(TypedDict):
    question:       str
    search_results: list[dict]
    answer:         str

class ResearchState(_ResearchStateRequired, total=False):
    run_start_time:   float
    search_method:    str
    search_tokens_in: int
    search_tokens_out:int

# -- consolidated agent-facing wrappers -----------------------------------------

async def search_by_topic(query: str, limit: int = 8) -> list[dict]:
    """Classify the query using NMF and retrieve papers assigned to that
    topic cluster. Best for broad thematic questions like 'what research
    exists on federated learning' or 'papers about graph neural networks'.
    Uses pre-computed NMF topic assignments - fast and topic-coherent."""
    result = classify_topic(query)
    return await search_abstracts_by_topic(topic_id=result["top_topic_id"], limit=limit)


async def search_by_content(query: str, mode: str = "hybrid", limit: int = 8) -> list[dict]:
    """Search abstract text using keyword matching, pgvector semantic similarity,
    or a hybrid of both.
    Semantic mode (mode='semantic') embeds the query and finds nearest neighbours
    in vector space - it understands paraphrases and conceptual intent, so it works
    even when the query uses different vocabulary than the abstracts.
    Keyword mode (mode='keyword') is best for exact method names, author names, or
    acronyms where the literal string must appear.
    Hybrid mode (mode='hybrid') combines both signals and is a safe default when
    the query mixes specific terms with broader context.
    Prefer this tool over search_by_topic when the question crosses multiple topic
    areas, when precision on a specific term matters, or when semantic similarity
    to the query text is more important than topic-cluster coherence."""
    alpha_map = {"keyword": 1.0, "semantic": 0.0, "hybrid": 0.5}
    alpha = alpha_map.get(mode, 0.5)
    return await search_abstracts_hybrid(query=query, limit=limit, alpha=alpha)
