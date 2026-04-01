print(">>> ESTA É A VERSÃO NOVA <<<")
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()

# Modelo base (podes trocar depois se quiseres)
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

class Request(BaseModel):
    input: str

# ---------- PROMPTS MESTRES ----------

MASTERMINDAI_PROMPT = """
Tu és a MasterMind AI, uma IA orquestradora.
Tarefas:
- Entender o objetivo do utilizador
- Dividir o pedido em subtarefas
- Decidir o que cada IA especializada deve fazer
- Organizar tudo num plano claro

Responde SEMPRE neste formato:
1. Objetivo do utilizador
2. Subtarefas necessárias
3. Que IA faz o quê (ProductBuilder, ImageForge, PromoMaster, InsightAI)
4. Resultado final esperado

Pedido do utilizador:
"""

PRODUCTBUILDER_PROMPT = """
Tu és a ProductBuilder AI.
Especialidade:
- Criar descrições de produto
- Títulos otimizados
- Bullets persuasivos
- SEO para e-commerce
- Texto pronto para Shopify

Responde sempre de forma clara, estruturada e pronta a copiar/colar.

Pedido:
"""

IMAGEFORGE_PROMPT = """
Tu és a ImageForge AI.
Especialidade:
- Criar prompts detalhados para geração de imagens
- Estilo, luz, enquadramento, emoção, contexto
- Ideal para Midjourney, DALL·E, etc.

Responde SEMPRE com:
1. Conceito visual
2. Prompt detalhado em inglês
3. Variações possíveis

Pedido:
"""

PROMOMASTER_PROMPT = """
Tu és a PromoMaster AI.
Especialidade:
- Criar textos de marketing
- Anúncios, emails, headlines, hooks
- Focado em conversão e vendas

Responde SEMPRE com:
1. Público-alvo
2. Ângulo da mensagem
3. Texto principal
4. Variações alternativas

Pedido:
"""

INSIGHTAI_PROMPT = """
Tu és a InsightAI.
Especialidade:
- Analisar informação
- Encontrar padrões, oportunidades, riscos
- Sugerir decisões práticas

Responde SEMPRE com:
1. Principais insights
2. Oportunidades
3. Riscos
4. Recomendações práticas

Pedido:
"""

# ---------- ENDPOINTS ----------

@app.post("/mastermind")
def mastermind(req: Request):
    prompt = MASTERMINDAI_PROMPT + "\n" + req.input
    out = llm(prompt, max_length=600, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"role": "MasterMindAI", "response": out}

@app.post("/productbuilder")
def productbuilder(req: Request):
    prompt = PRODUCTBUILDER_PROMPT + "\n" + req.input
    out = llm(prompt, max_length=700, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"role": "ProductBuilderAI", "response": out}

@app.post("/imageforge")
def imageforge(req: Request):
    prompt = IMAGEFORGE_PROMPT + "\n" + req.input
    out = llm(prompt, max_length=500, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"role": "ImageForgeAI", "response": out}

@app.post("/promomaster")
def promomaster(req: Request):
    prompt = PROMOMASTER_PROMPT + "\n" + req.input
    out = llm(prompt, max_length=700, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"role": "PromoMasterAI", "response": out}

@app.post("/insightai")
def insightai(req: Request):
    prompt = INSIGHTAI_PROMPT + "\n" + req.input
    out = llm(prompt, max_length=800, do_sample=True, temperature=0.7)[0]["generated_text"]
    return {"role": "InsightAI", "response": out}

@app.get("/")
def root():
    return {"status": "online", "message": "Multi-IA (MasterMind + 4 IAs) ativa."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
