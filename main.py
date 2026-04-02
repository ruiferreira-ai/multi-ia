import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Request(BaseModel):
    input: str

# -----------------------------
# Função LLM via API HTTP Groq
# -----------------------------
def call_llm(system_prompt: str, user_input: str, max_tokens: int = 700):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Erro: GROQ_API_KEY não encontrada no ambiente."

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
    "model": "llama-3.1-70b-versatile",
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

    # Tenta converter para JSON
    try:
        data = response.json()
    except Exception:
        return f"Erro: resposta inválida da API Groq. Status: {response.status_code}"

    # Se a API devolveu erro
    if "error" in data:
        return f"Erro da API Groq: {data['error']}"

    # Se não há choices
    if "choices" not in data:
        return f"Erro: resposta inesperada da API Groq: {data}"

    # Tudo OK → devolve o texto
    return data["choices"][0]["message"]["content"]

# -----------------------------
# PROMPT DO PRODUCT BUILDER
# -----------------------------
PRODUCTBUILDER_PROMPT = "És um criador de produtos."

# -----------------------------
# ENDPOINT PRINCIPAL
# -----------------------------
@app.post("/productbuilder")
def productbuilder(req: Request):
    out = call_llm(PRODUCTBUILDER_PROMPT, req.input, max_tokens=700)
    return {"role": "ProductBuilderAI", "response": out}

# -----------------------------
# ENDPOINTS DE TESTE
# -----------------------------
@app.get("/")
def root():
    return {"status": "online"}

@app.get("/debug-env")
def debug_env():
    return {"GROQ_API_KEY": os.getenv("GROQ_API_KEY")}
