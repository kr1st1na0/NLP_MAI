
"""
FastAPI сервис для проксирования запросов к Ollama LLM.

Этот модуль предоставляет REST API для взаимодействия с моделью Qwen2.5:0.5B,
запущенной через Ollama. Сервис принимает текстовые запросы и возвращает
ответы от языковой модели.
"""

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Инициализация FastAPI приложения
app = FastAPI()

# Конфигурация Ollama сервера
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:0.5b"

class PromptRequest(BaseModel):
    """
    Модель запроса к LLM.

    Attributes:
        prompt (str): Текст запроса для отправки языковой модели
    """
    prompt: str

@app.post("/generate")
def generate(req: PromptRequest):
    """
    Отправляет текстовый запрос к модели Ollama и возвращает ответ.
    
    Этот эндпоинт принимает JSON с полем prompt, отправляет запрос
    к запущенному экземпляру Ollama и возвращает сгенерированный ответ.

    Args:
        req (PromptRequest): Объект запроса, содержащий текстовый промпт
    
    Returns:
        dict: Ответ от Ollama в формате JSON, содержащий сгенерированный текст
    """
    payload = {"model": MODEL, "prompt": req.prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/health")
def health():
    """
    Проверяет работоспособность сервиса.

    Эндпоинт для health check, позволяющий убедиться,
    что FastAPI сервис запущен и работает корректно.

    Returns:
    dict: Словарь с ключом "ok": True, подтверждающий работоспособность
    """
    return {"ok": True}