import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    input: str

def call_llm(system_prompt: str, user_input: str, max_tokens: int = 700):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Erro: OPENAI_API_KEY não encontrada no ambiente."

    url = "https://api.openai.com/v1/responses"

    payload = {
        "model": "gpt-4o-mini",
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "max_output_tokens": max_tokens,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    try:
        data = response.json()
    except Exception:
        return f"Erro: resposta inválida da API OpenAI. Status: {response.status_code}"

    if "error" in data:
        return f"Erro da API OpenAI: {data['error']}"

    try:
        return data["output"][0]["content"][0]["text"]
    except:
        return f"Erro inesperado: {data}"
    

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
    return {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
