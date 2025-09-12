import os
import re
from fireworks import LLM
import yaml
from pydantic import BaseModel
import json

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
    context_blocks = "\n\n".join([
        f"Doc {i+1}: Doc id: {d['id']}; Title: {d['title']}\nDate: {d['date']}\nDocument: {d['text']}"
        for i, d in enumerate(docs)
    ])
    message = (
        f"{context_blocks}\n\n"
        f"Question: {query}\n"
    )
    return message

def get_llm_response(user_message: str) -> str:
    """Call the LLM with system prompt and user message and return raw content.

    @param user_message: The message string to send as the user content.
    @return: Raw content string from the model's top choice.
    """
    llm = LLM(
        model="llama-v3p3-70b-instruct",
        deployment_type="serverless",
        api_key=os.getenv("FIREWORKS_API_KEY")
        )
    response = llm.chat.completions.create(
    messages=[{
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": user_message,
    }],
    temperature=0,
    response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Result",
                "schema": Result.model_json_schema(),
                "strict": True
            }
        },
    )
    content = response.choices[0].message.content or ""
    return content

def parse_llm_response(llm_response: str) -> Result:
    """Parse the raw LLM response into a Result object.

    The model is instructed to return JSON; this function extracts the first
    JSON object from the string and parses it.

    @param llm_response: Raw content string returned by the LLM.
    @return: Parsed Result pydantic model (or dict if validation not applied).
    """
    match = re.search(r'\{.*\}', llm_response, flags=re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        first = llm_response.find("{")
        last = llm_response.rfind("}")
        json_str = llm_response[first:last+1] if first != -1 and last != -1 else "{}"
    parsed = json.loads(json_str)
    return parsed