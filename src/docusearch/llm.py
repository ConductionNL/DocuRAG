import os
import re
import logging
from fireworks import LLM
import yaml
from pathlib import Path
from pydantic import BaseModel
import json
from dotenv import load_dotenv

load_dotenv()

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

HISTORY_LENGTH = 5
# Default model name to avoid runtime errors when env is missing
# See memory: default should be 'llama-v3p3-70b-instruct'
LLM_NAME = os.getenv("LLM_NAME") or "llama-v3p3-70b-instruct"
logger.info(f"LLM_NAME: {LLM_NAME}")


class Source(BaseModel):
    id: int
    title: str
    date: str
    reason: str


class Result(BaseModel):
    answer: str
    sources: list[Source]


class FollowUpDecision(BaseModel):
    is_follow_up: bool
    reason: str


_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"
with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
    _prompts = yaml.safe_load(f)
SYSTEM_PROMPT = _prompts["system_woo"]["text"]


def load_llm_message(
    docs: list[dict], query: str, history: list[dict] | None = None
) -> str:
    """Construct the user message for the LLM from retrieved docs, optional history, and query.

    @param docs: List of document dicts, each with keys like 'id', 'title', 'date', 'text'.
    @param query: Natural-language question from the user.
    @param history: Prior turns as a list of {"user": str, "assistant": str}.
    @return: Concatenated message string for the LLM.
    """
    context_blocks = "\n\n".join(
        [
            f"Doc {i+1}: Doc id: {d['id']}; Title: {d['title']}\nDate: {d['date']}\nDocument: {d['text']}"
            for i, d in enumerate(docs)
        ]
    )
    history_str = ""
    if history:
        turns = "\n".join(
            [
                f"User: {t['user']}\nAssistant: {t['assistant']}"
                for t in history[-HISTORY_LENGTH:]
            ]
        )
        history_str = f"\n\nConversation so far:\n{turns}"
    message = f"{context_blocks}{history_str}\n\nQuestion: {query}\n"
    return message


def classify_follow_up(
    current_query: str,
    last_query: str | None,
    prior_sources: list[dict] | None,
    last_answer: str | None = None,
    fireworks_api_key: str | None = None,
) -> bool:
    """Use the LLM to classify if the current query is a follow-up.

    The classifier decides if the user question depends on or continues the
    previous turn's topic/sources versus starting a new topic. Returns True for
    follow-up, False for new topic.
    """
    llm = LLM(
        model=LLM_NAME,
        deployment_type="serverless",
        api_key=fireworks_api_key,
    )

    prev_q = last_query or ""
    prev_ans = (last_answer or "")[:500]
    src_lines: list[str] = []
    for i, d in enumerate(prior_sources or []):
        # use key fields robustly
        ident = d.get("id", d.get("doc_id", ""))
        title = d.get("title", "")
        date = d.get("date", "")
        src_lines.append(f"Source {i+1}: id={ident}; title={title}; date={date}")
    sources_block = "\n".join(src_lines) if src_lines else "(none)"

    classifier_system = (
        "You are a strict JSON-only classifier. Decide if the current user query "
        "is a follow-up that depends on or meaningfully continues the previous "
        "turn's topic or the listed sources, versus a new unrelated request. "
        "Return only JSON with fields 'is_follow_up' (boolean) and 'reason' (string)."
    )

    classifier_user = (
        "Previous query:\n"
        + prev_q
        + ("\n\nPrevious answer (truncated):\n" + prev_ans if prev_ans else "")
        + "\n\nPrevious sources (ids, titles):\n"
        + sources_block
        + "\n\nCurrent query:\n"
        + current_query
        + "\n\nGuidelines:\n"
        "- Return true when the user references the prior docs (e.g., 'doc 2', 'that section'), "
        "asks for more detail, comparison, or refinement on the same topic, or otherwise depends on the previous context.\n"
        "- Return false when the query introduces a new subject or can be answered independently without the previous context."
    )

    response = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": classifier_system},
            {"role": "user", "content": classifier_user},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "FollowUpDecision",
                "schema": FollowUpDecision.model_json_schema(),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        return bool(data.get("is_follow_up", False))
    except Exception:
        logger.warning("Follow-up classifier returned non-JSON; defaulting to False")
        return False


def get_llm_response(user_message: str, fireworks_api_key: str | None = None) -> str:
    """Call the LLM with system prompt and user message and return raw content.

    @param user_message: The message string to send as the user content.
    @return: Raw content string from the model's top choice.
    """
    llm = LLM(
        model=LLM_NAME,
        deployment_type="serverless",
        api_key=fireworks_api_key,
        perf_metrics_in_response=True,
    )
    response = llm.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Result",
                "schema": Result.model_json_schema(),
                "strict": True,
            },
        },
    )
    # log metrics to pinpoint where time is spent
    perf = getattr(response, "perf_metrics", None)
    if perf:
        logger.info(f"fireworks perf: {perf}")
    content = response.choices[0].message.content or ""
    return content


def stream_llm_response(
    user_message: str, fireworks_api_key: str | None = None, print_stream: bool = True
) -> str:
    """Stream the LLM response with the same JSON schema and optionally print.

    The function streams `delta.content` chunks as they arrive, printing them
    when `print_stream` is True, while accumulating the full content to return.

    @param user_message: The message string to send as the user content.
    @param print_stream: If True, print streamed tokens to stdout.
    @return: The full accumulated content string from the top choice.
    """
    llm = LLM(
        model=LLM_NAME,
        deployment_type="serverless",
        api_key=fireworks_api_key,
    )
    response_generator = llm.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Result",
                "schema": Result.model_json_schema(),
                "strict": True,
            },
        },
        stream=True,
    )

    content_parts: list[str] = []
    for chunk in response_generator:
        try:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
        except Exception:
            piece = None
        if piece:
            if print_stream:
                print(piece, end="", flush=True)
            content_parts.append(piece)

    if print_stream:
        print()
    return "".join(content_parts)


async def astream_llm_response(
    user_message: str, fireworks_api_key: str | None = None, print_stream: bool = True
):
    """Async generator that yields NDJSON lines with incremental content under 'delta'."""
    llm = LLM(
        model=LLM_NAME,
        deployment_type="serverless",
        api_key=fireworks_api_key,
    )
    stream = await llm.chat.completions.acreate(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Result",
                "schema": Result.model_json_schema(),
                "strict": True,
            },
        },
        stream=True,
    )

    async for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
        except Exception:
            piece = None
        if piece:
            if print_stream:
                print(piece, end="", flush=True)
            yield json.dumps({"delta": piece}) + "\n"

    if print_stream:
        print()


def parse_llm_response(llm_response: str) -> Result:
    """Parse the raw LLM response into a Result object.

    The model is instructed to return JSON; this function extracts the first
    JSON object from the string and parses it.

    @param llm_response: Raw content string returned by the LLM.
    @return: Parsed Result pydantic model (or dict if validation not applied).
    """
    match = re.search(r"\{.*\}", llm_response, flags=re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        first = llm_response.find("{")
        last = llm_response.rfind("}")
        json_str = (
            llm_response[first : last + 1] if first != -1 and last != -1 else "{}"
        )
    parsed = json.loads(json_str)
    return parsed
