import os
import re
import logging
from fireworks import LLM
import yaml
from pydantic import BaseModel
import json


# Configure module-level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class Source(BaseModel):
    id: int
    title: str
    date: str
    reason: str


class Result(BaseModel):
    answer: str
    sources: list[Source]


with open("prompts.yaml", "r", encoding="utf-8") as f:
    _prompts = yaml.safe_load(f)
SYSTEM_PROMPT = _prompts["system_woo"]["text"]

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")


def load_llm_message(docs: list[dict], query: str) -> str:
    """Construct the user message for the LLM from retrieved docs and query.

    @param docs: List of document dicts, each with keys like 'id', 'title', 'date', 'body'.
    @param query: Natural-language question from the user.
    @return: Concatenated message string for the LLM.
    """
    context_blocks = "\n\n".join(
        [
            f"Doc {i+1}: Doc id: {d['id']}; Title: {d['title']}\nDate: {d['date']}\nDocument: {d['text']}"
            for i, d in enumerate(docs)
        ]
    )
    message = f"{context_blocks}\n\n" f"Question: {query}\n"
    return message


def get_llm_response(user_message: str) -> str:
    """Call the LLM with system prompt and user message and return raw content.

    @param user_message: The message string to send as the user content.
    @return: Raw content string from the model's top choice.
    """
    llm = LLM(
        model="llama-v3p3-70b-instruct",
        deployment_type="serverless",
        api_key=os.getenv("FIREWORKS_API_KEY"),
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


def stream_llm_response(user_message: str, print_stream: bool = True) -> str:
    """Stream the LLM response with the same JSON schema and optionally print.

    The function streams `delta.content` chunks as they arrive, printing them
    when `print_stream` is True, while accumulating the full content to return.

    @param user_message: The message string to send as the user content.
    @param print_stream: If True, print streamed tokens to stdout.
    @return: The full accumulated content string from the top choice.
    """
    llm = LLM(
        model="llama-v3p3-70b-instruct",
        deployment_type="serverless",
        api_key=os.getenv("FIREWORKS_API_KEY"),
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


async def astream_llm_response(user_message: str, print_stream: bool = True) -> str:
    """Async variant of streaming with the same JSON schema and optional printing."""
    llm = LLM(
        model="llama-v3p3-70b-instruct",
        deployment_type="serverless",
        api_key=os.getenv("FIREWORKS_API_KEY"),
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

    content_parts: list[str] = []
    async for chunk in stream:
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


async def astream_llm_ndjson(user_message: str):
    """Async generator that yields NDJSON lines with incremental LLM output.

    Each yielded line is a JSON object followed by a newline. The primary event
    is {"delta": "..."} for incremental content. A final {"event": "done"}
    line signals completion.
    """
    llm = LLM(
        model="llama-v3p3-70b-instruct",
        deployment_type="serverless",
        api_key=os.getenv("FIREWORKS_API_KEY"),
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

    try:
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
            except Exception:
                piece = None
            if piece:
                yield json.dumps({"delta": piece}) + "\n"
    except Exception as e:
        logger.exception("Streaming LLM failed: %s", e)
        yield json.dumps({"event": "error", "message": str(e)}) + "\n"
    finally:
        # Signal completion to the client
        yield json.dumps({"event": "done"}) + "\n"


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
