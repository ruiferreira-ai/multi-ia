print(">>> ESTA É A VERSÃO NOVA <<<")

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

class Request(BaseModel):
    input: str

def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não encontrada no ambiente.")
    return Groq(api_key=api_key)

def call_llm(system_prompt: str, user_input: str, max_tokens: int = 700):
    client = get_client()
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content

# ---------- PROMPTS MESTRES ----------

MASTERMINDAI_PROMPT = """..."""
PRODUCTBUILDER_PROMPT = """..."""
IMAGEFORGE_PROMPT = """..."""
PROMOMASTER_PROMPT = """..."""
INSIGHTAI_PROMPT = """..."""

# ---------- ENDPOINTS ----------

@app.post("/mastermind")
def mastermind(req: Request):
    out = call_llm(MASTERMINDAI_PROMPT, req.input, max_tokens=600)
    return {"role": "MasterMindAI", "response": out}

@app.post("/productbuilder")
def productbuilder(req: Request):
    out = call_llm(PRODUCTBUILDER_PROMPT, req.input, max_tokens=700)
    return {"role": "ProductBuilderAI", "response": out}

@app.post("/imageforge")
def imageforge(req: Request):
    out = call_llm(IMAGEFORGE_PROMPT, req.input, max_tokens=500)
    return {"role": "ImageForgeAI", "response": out}

@app.post("/promomaster")
def promomaster(req: Request):
    out = call_llm(PROMOMASTER_PROMPT, req.input, max_tokens=700)
    return {"role": "PromoMasterAI", "response": out}

@app.post("/insightai")
def insightai(req: Request):
    out = call_llm(INSIGHTAI_PROMPT, req.input, max_tokens=800)
    return {"role": "InsightAI", "response": out}

@app.get("/")
def root():
    return {"status": "online", "message": "Multi-IA (MasterMind + 4 IAs) ativa."}

@app.get("/debug-env")
def debug_env():
    import os
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY")
    }


