"""
Single shared Groq client. Planner, workers, and writer all import call_llm
from here instead of each setting up their own client.

TODO:
- initialize Groq client using GROQ_API_KEY from env (see utils via .env)
- call_llm(prompt: str, structured_output_model: type[BaseModel] | None = None)
    -> if structured_output_model is given, use it to constrain/parse the response
    -> otherwise return plain text
- pick a Groq free-tier model (e.g. llama-3.3-70b-versatile) and keep it as
  a single constant here so it's easy to swap later
"""
import os
from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI  # Gemini's compatibility endpoint uses the OpenAI SDK shape
load_dotenv()  # Load environment variables from .env file

############## M1 groq ###############3


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL="openai/gpt-oss-120b"  # Free-tier model for Groq API

############ M2 gemini #########################
# client = OpenAI(
#     api_key=os.environ["GEMINI_API_KEY"],
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# MODEL = "gemini-3.5-flash-lite"


######### m3 ##########

# import os
# from dotenv import load_dotenv
# from cerebras.cloud.sdk import Cerebras

# load_dotenv()

# client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

# MODEL = "gemma-4-31b"


# def call_llm(prompt: str, system: str = "You are a helpful research assistant.", max_tokens: int = 2000) -> str:
#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": prompt},
#         ],
#         max_completion_tokens=max_tokens,
#     )
#     return response.choices[0].message.content


# def call_llm_structured(prompt: str, pydantic_model, system: str = "You are a helpful research assistant.") -> object:
#     schema_hint = pydantic_model.model_json_schema()
#     full_prompt = (
#         f"{prompt}\n\n"
#         f"Respond ONLY with valid JSON matching this schema, no other text:\n{schema_hint}"
#     )
#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": full_prompt},
#         ],
#         response_format={"type": "json_object"},
#     )
#     raw = response.choices[0].message.content
#     return pydantic_model.model_validate_json(raw)




############# 
import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

PROVIDERS = [
    {
        "name": "groq",
        "client": Groq(api_key=os.environ["GROQ_API_KEY"]),
        "model": "openai/gpt-oss-120b",
    },
    {
        "name": "gemini",
        "client": OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "model": "gemini-3.5-flash-lite",
    },
]


def call_llm(prompt: str, system: str = "You are a helpful research assistant.", max_tokens: int = 2000) -> str:
    last_error = None
    for provider in PROVIDERS:
        try:
            response = provider["client"].chat.completions.create(
                model=provider["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[llm fallback] {provider['name']} failed: {e}")
            last_error = e
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def call_llm_structured(prompt: str, pydantic_model, system: str = "You are a helpful research assistant.") -> object:
    schema_hint = pydantic_model.model_json_schema()
    full_prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema, no other text:\n{schema_hint}"

    last_error = None
    for provider in PROVIDERS:
        try:
            response = provider["client"].chat.completions.create(
                model=provider["model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return pydantic_model.model_validate_json(raw)
        except Exception as e:
            print(f"[llm fallback] {provider['name']} failed: {e}")
            last_error = e
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")