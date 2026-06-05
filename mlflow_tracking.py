import json
import os
import mlflow

from agent_config import CHAT_MODEL

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
EXPERIMENT_NAME = "research_agent"

# Claude Haiku 4.5 pricing
_INPUT_COST_PER_TOKEN  = 0.80 / 1_000_000   # $0.80 per million input tokens
_OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000   # $4.00 per million output tokens


def log_research_run(state: dict, end_time: float) -> None:
    """Log one /research query as a single MLflow run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    s_in  = state.get("search_tokens_in", 0)
    s_out = state.get("search_tokens_out", 0)
    u_in  = state.get("summarise_tokens_in", 0)
    u_out = state.get("summarise_tokens_out", 0)
    total_in  = s_in + u_in
    total_out = s_out + u_out
    cost      = total_in * _INPUT_COST_PER_TOKEN + total_out * _OUTPUT_COST_PER_TOKEN
    latency   = end_time - state.get("run_start_time", end_time)

    papers  = state.get("search_results", [])
    answer  = state.get("answer", "")

    with mlflow.start_run():
        mlflow.log_params({
            "question":      state.get("question", "")[:250],
            "model":         CHAT_MODEL,
            "search_method": state.get("search_method", "unknown"),
            "search_params": json.dumps(state.get("search_params", {})),
        })

        mlflow.log_metrics({
            "papers_retrieved":    len(papers),
            "search_tokens_in":    s_in,
            "search_tokens_out":   s_out,
            "summarise_tokens_in": u_in,
            "summarise_tokens_out":u_out,
            "total_tokens_in":     total_in,
            "total_tokens_out":    total_out,
            "cost_usd":            round(cost, 6),
            "latency_s":           round(latency, 2),
            "answer_length_chars": len(answer),
        })

        mlflow.log_text(answer, "answer.txt")
        mlflow.log_text(
            json.dumps(
                [{"doi": p["doi"], "title": p["title"], "year": p["year"]}
                 for p in papers],
                indent=2,
            ),
            "retrieved_papers.json",
        )
