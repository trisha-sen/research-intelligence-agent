# agent.py
import json
import asyncio
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from agent_tools import search_abstracts_by_topic, search_abstracts, classify_topic
from agent_tool_schemas import TOOL_SCHEMAS

load_dotenv()

client = AsyncAnthropic()

CHAT_MODEL = "claude-haiku-4-5"
SYSTEM_PROMPT = """You are a research assistant with access to a corpus of 33K ML
journal abstracts. When answering research questions:
1. First classify the question's topic using classify_topic to get a topic ID.
2. Search for relevant papers using both:
   - search_abstracts_by_topic(topic_id) - finds papers where that NMF topic is
     dominant; prefer this for broad topic overviews.
   - search_abstracts(query) - keyword search; prefer this for specific methods,
     terms, or when the topic search returns insufficient results.
   Use both when the question has both a clear topic and specific keywords.
3. Synthesise the results into a clear answer with paper citations (title + year).

Always ground your answer in the retrieved papers. If search returns nothing
relevant, say so rather than answering from general knowledge."""


# -- Tool dispatcher ------------------------------------------------------------

async def dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    if name=="search_abstracts_by_topic":
        result = await search_abstracts_by_topic(**args)
    elif name == "search_abstracts":
        result = await search_abstracts(**args)
    elif name == "classify_topic":
        result = classify_topic(**args)        # sync - NMF is fast
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# -- The loop -------------------------------------------------------------------

async def run_agent(question: str, max_steps: int = 10) -> str:
    """
    Run the ReAct loop for a research question.
    Returns the final answer string.
    """
    messages = [
        {"role": "user", "content": question},
    ]

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    for step in range(max_steps):
        print(f"\n-- Step {step + 1} --")

        response = await client.messages.create(
            model=CHAT_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        stop_reason = response.stop_reason

        # -- Case 1: model wants to call tools ---------------------------------
        if stop_reason == "tool_use":
            # Append the assistant's response (including tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  --> calling {block.name}({block.input})")
                    result = await dispatch_tool(block.name, block.input)
                    print(f"  <-- {result[:200]}...")

                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })

            # Feed all results back in a single user message
            messages.append({"role": "user", "content": tool_results})

        # -- Case 2: model is done - final answer ------------------------------
        elif stop_reason == "end_turn":
            answer = next(
                (block.text for block in response.content if block.type == "text"),
                ""
            )
            print(f"\nFinal answer:\n{answer}")
            return answer

        # -- Case 3: something unexpected --------------------------------------
        else:
            print(f"  Unexpected stop_reason: {stop_reason}")
            break

    return "Agent reached max steps without a final answer."


# -- Run it --------------------------------------------------------------------

if __name__ == "__main__":
    answer = asyncio.run(
        run_agent("What are the main approaches to privacy in federated learning?")
    )
