import os
import re
import logging
from fireworks import LLM
import yaml
from pathlib import Path
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import time
import random
import asyncio
import httpx
import socket
import concurrent.futures as futures

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


# Timeouts and retry configuration (env-overridable)
def _get_float_env(name: str, default_value: float) -> float:
    try:
        return float(os.getenv(name, str(default_value)))
    except Exception:
        return default_value


def _get_int_env(name: str, default_value: int) -> int:
    try:
        return int(os.getenv(name, str(default_value)))
    except Exception:
        return default_value


LLM_TIMEOUT_S: float = _get_float_env("LLM_TIMEOUT_S", 60.0)
LLM_MAX_RETRIES: int = _get_int_env("LLM_MAX_RETRIES", 2)
LLM_BACKOFF_BASE_S: float = _get_float_env("LLM_BACKOFF_BASE_S", 0.75)
LLM_STREAM_TIMEOUT_S: float = _get_float_env("LLM_STREAM_TIMEOUT_S", 90.0)
LLM_STREAM_IDLE_TIMEOUT_S: float = _get_float_env("LLM_STREAM_IDLE_TIMEOUT_S", 20.0)


def _is_retryable(exc: Exception) -> bool:
    # Common transient classes from httpx and network layers
    retryable_httpx = (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.WriteError,
        httpx.TransportError,
    )
    if isinstance(exc, retryable_httpx):
        return True
    # Generic timeout types
    if isinstance(
        exc, (TimeoutError, asyncio.TimeoutError, futures.TimeoutError, socket.timeout)
    ):
        return True
    # If the exception exposes an HTTP status code, decide based on it
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 408 or status == 429 or status >= 500):
        return True
    # Heuristic check by message text
    msg = str(exc).lower()
    for token in [
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "server error",
        "too many requests",
        "rate limit",
    ]:
        if token in msg:
            return True
    return False


def _backoff_delay_s(
    attempt_index_one_based: int,
    base_s: float = LLM_BACKOFF_BASE_S,
    cap_s: float = 10.0,
) -> float:
    # Exponential backoff with jitter
    expo = base_s * (2 ** (attempt_index_one_based - 1))
    jitter = expo * (0.5 + random.random())
    return min(cap_s, jitter)


def _run_with_retries(callable_fn):
    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            logger.info("llm request attempt %d/%d", attempt + 1, LLM_MAX_RETRIES + 1)
            return callable_fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            should = _is_retryable(exc)
            if attempt >= LLM_MAX_RETRIES or not should:
                break
            delay = _backoff_delay_s(attempt + 1)
            logger.warning(
                "LLM call failed (attempt %s/%s): %s; retrying in %.2fs",
                attempt + 1,
                LLM_MAX_RETRIES + 1,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _call_with_deadline_sync(callable_fn, timeout_s: float):
    """Run a blocking callable with a wall-clock deadline using a thread.

    We avoid blocking on shutdown if the work exceeds the deadline to ensure the caller
    does not wait longer than timeout_s. The underlying task may still finish in the
    background; we proactively shutdown the executor without waiting.
    """
    executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-call")
    future = executor.submit(callable_fn)
    try:
        return future.result(timeout=timeout_s)
    except futures.TimeoutError:
        # Do not block; allow thread to finish in background.
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        raise
    except Exception:
        # Ensure executor is torn down quickly on other errors too.
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        raise
    finally:
        # Fast shutdown if already completed.
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


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


_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "prompts.yaml"
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

    response = _run_with_retries(
        lambda: _call_with_deadline_sync(
            lambda: llm.chat.completions.create(
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
            ),
            LLM_TIMEOUT_S,
        )
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
    t0 = time.monotonic()
    response = _run_with_retries(
        lambda: _call_with_deadline_sync(
            lambda: llm.chat.completions.create(
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
            ),
            LLM_TIMEOUT_S,
        )
    )
    # log metrics to pinpoint where time is spent
    perf = getattr(response, "perf_metrics", None)
    if perf:
        logger.info(f"fireworks perf: {perf}")
    logger.info("llm non-streaming duration: %.2fs", time.monotonic() - t0)
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
    """Async generator that yields NDJSON lines with incremental content under 'delta',
    with overall and idle timeouts plus pre-start retries.
    """
    llm = LLM(
        model=LLM_NAME,
        deployment_type="serverless",
        api_key=fireworks_api_key,
    )

    prestart_retries = LLM_MAX_RETRIES

    for attempt in range(prestart_retries + 1):
        logger.info(
            "llm stream request attempt %d/%d",
            attempt + 1,
            prestart_retries + 1,
        )
        try:
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
        except Exception as exc:  # noqa: BLE001
            if attempt < prestart_retries and _is_retryable(exc):
                delay = _backoff_delay_s(attempt + 1)
                logger.warning(
                    "LLM stream creation failed (attempt %s/%s): %s; retrying in %.2fs",
                    attempt + 1,
                    prestart_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            yield json.dumps({"event": "error", "message": str(exc)}) + "\n"
            return

        start_time = time.monotonic()
        got_any_chunk = False

        while True:
            elapsed = time.monotonic() - start_time
            remaining_overall = max(0.0, LLM_STREAM_TIMEOUT_S - elapsed)
            if remaining_overall <= 0.0:
                yield json.dumps(
                    {
                        "event": "timeout",
                        "type": "overall",
                        "message": "LLM stream exceeded overall deadline",
                    }
                ) + "\n"
                return

            next_timeout = min(LLM_STREAM_IDLE_TIMEOUT_S, remaining_overall)
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=next_timeout)
            except asyncio.TimeoutError:
                if not got_any_chunk and attempt < prestart_retries:
                    # Consider as pre-start stall and retry establishing the stream
                    delay = _backoff_delay_s(attempt + 1)
                    logger.warning(
                        "LLM stream idle before first token (attempt %s/%s); retrying in %.2fs",
                        attempt + 1,
                        prestart_retries + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    break
                # Idle timeout after streaming started (or no retries left)
                yield json.dumps(
                    {
                        "event": "timeout",
                        "type": "idle",
                        "message": "No tokens received within idle timeout",
                    }
                ) + "\n"
                return
            except StopAsyncIteration:
                # Proper end of stream
                if print_stream:
                    print()
                yield json.dumps({"event": "done"}) + "\n"
                return
            except Exception as exc:  # noqa: BLE001
                if (
                    not got_any_chunk
                    and attempt < prestart_retries
                    and _is_retryable(exc)
                ):
                    delay = _backoff_delay_s(attempt + 1)
                    logger.warning(
                        "LLM stream failed before first token (attempt %s/%s): %s; retrying in %.2fs",
                        attempt + 1,
                        prestart_retries + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    break
                yield json.dumps({"event": "error", "message": str(exc)}) + "\n"
                return

            # Normal token chunk received
            try:
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
            except Exception:
                piece = None
            if piece:
                if print_stream:
                    print(piece, end="", flush=True)
                got_any_chunk = True
                yield json.dumps({"delta": piece}) + "\n"
        # If we broke out of inner loop to retry, continue to next attempt


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
