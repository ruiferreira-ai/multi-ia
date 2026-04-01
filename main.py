print(">>> ESTA É A VERSÃO NOVA <<<")

from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

# Cliente Groq (usa a variável de ambiente GROQ_API_KEY no Railway)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Request(BaseModel):
    input: str

def call_llm(system_prompt: str, user_input: str, max_tokens: int = 700):
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
