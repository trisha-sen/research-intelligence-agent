# tool_schemas.py

TOOL_SCHEMAS = [
    {
        "name": "search_abstracts",
        "description": (
            "Keyword search over 33K ML journal abstracts stored in Postgres. "
            "Use this when you need to find papers on a specific topic or method. "
            "Returns doi, title, abstract snippet, year, and NMF topic IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term - e.g. 'attention mechanism', 'federated learning privacy'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "classify_topic",
        "description": (
            "Run NMF topic inference on any text string. Returns the top topic label "
            "from 20 research clusters (e.g. 'Federated Learning & Data Privacy', "
            "'Graph Neural Networks'). Use this to classify a research question or "
            "abstract into its primary topic area."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to classify - a question, abstract, or any research text"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "search_abstracts_by_topic",
        "description": (
            "Find papers by NMF topic ID. Use this when you already know a topic ID "
            "(from classify_topic or the user) and want papers strongly associated with it. "
            "Results are ranked by topic position (primary topic first), then citation count, "
            "then recency. Returns title, abstract snippet, year, topic list, topic position, and citations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": "integer",
                    "description": "NMF topic ID (0–19)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (default 10, max 20)",
                    "default": 10
                }
            },
            "required": ["topic_id"]
        }
    }
]
