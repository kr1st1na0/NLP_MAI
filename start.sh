#!/bin/bash

echo "Starting Ollama..."
ollama serve &

sleep 8

echo "Pulling model..."
ollama pull qwen2.5:0.5b

echo "Starting FastAPI on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000