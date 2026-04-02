import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    input: str

def call_llm(system_prompt: str, user_input: str, max_tokens: int = 700):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não encontrada.")

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    return data["choices"][0]["message"]["content"]

PRODUCTBUILDER_PROMPT = "És um criador de produtos."

@app.post("/productbuilder")
def productbuilder(req: Request):
    out = call_llm(PRODUCTBUILDER_PROMPT, req.input, max_tokens=700)
    return {"role": "ProductBuilderAI", "response": out}

@app.get("/")
def root():
    return {"status": "online"}

@app.get("/debug-env")
def debug_env():
    return {"GROQ_API_KEY": os.getenv("GROQ_API_KEY")}
