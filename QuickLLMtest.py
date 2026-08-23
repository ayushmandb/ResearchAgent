import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

load_dotenv()

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

MODEL = "gemma-4-31b"


def call_llm(prompt: str, system: str = "You are a helpful research assistant.", max_tokens: int = 2000) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_llm_structured(prompt: str, pydantic_model, system: str = "You are a helpful research assistant.") -> object:
    schema_hint = pydantic_model.model_json_schema()
    full_prompt = (
        f"{prompt}\n\n"
        f"Respond ONLY with valid JSON matching this schema, no other text:\n{schema_hint}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": full_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return pydantic_model.model_validate_json(raw)