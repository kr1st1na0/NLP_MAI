import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:0.5b"

class PromptRequest(BaseModel):
    """
    Модель запроса к LLM.
    """
    prompt: str

@app.post("/generate")
def generate(req: PromptRequest):
    """
    Отправляет текстовый запрос к модели Ollama и возвращает ответ.
    
    Этот эндпоинт принимает JSON с полем prompt, отправляет запрос
    к запущенному экземпляру Ollama и возвращает сгенерированный ответ.
    """
    payload = {"model": MODEL, "prompt": req.prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/health")
def health():
    return {"ok": True}