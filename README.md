# NLP_MAI

## Лабораторная работа №2 по киберфизическим системам

### МАИ | 4 курс | 8 семестр

**Выполнил студент:** Былькова Кристина Алексеевна

**Группа:** М8О-408Б-22

## Структура проекта:
```
- app
--- main.py
- dataset
--- SMSSpamCollection
- results
--- comparison.md
--- cot_few_shot_report.md
--- cot_report.md
--- few_shot_report.md
--- report_fast_api.md
--- report.md
--- zero_shot_report.md
- docker-compose.yml
- Dockerfile
- evaluate.py
- requirements.txt
- start.sh
- test_requests.py
```
где:
- ```main.py``` - FastAPI сервер, который работает внутри Docker контейнера. Предоставляет эндпоинт /generate для отправки запросов к модели Qwen2.5:0.5B, запущенной через Ollama. Служит прослойкой между клиентом и LLM;
- ```dataset``` - папка с данными для обучения и тестирования. Содержит файл SMSSpamCollection - классический датасет SMS сообщений;
- ```results``` - все результаты запусков (отчёты по каждой технике промптинга, сравнительный анализ техник, результаты тестирования через FastAPI и прямой Ollama)
- ```evaluate.py``` - основной скрипт для оценки качества различных техник промптинга. Загружает датасет, отправляет запросы к LLM, рассчитывает метрики (accuracy, precision, recall, f1) и сохраняет отчёты.
- ```test_requests.py``` - клиентский скрипт для тестирования двух режимов работы: прямое обращение к Ollama и через FastAPI. Позволяет сравнивать производительность и ответы модели.

## Запуск:

1. Запуск Docker контейнера
```
docker-compose up -d --build
```
После запуска контейнер доступен по адресам:
- FastAPI сервер: ```http://localhost:8000```
- Ollama сервер: ```http://localhost:11434```

Проверка работоспособности:
```
curl.exe http://localhost:8000/health
```

2. Тестирование режимов работы:

- Режим 1: Через FastAPI
```
python test_requests.py --mode fastapi
```

- Режим 2: Прямой доступ к Ollama
```
python test_requests.py --mode ollama
```

3. Оценка техник промптинга
- Запуск всех техник на 100 сообщениях:

```
python evaluate.py --technique all --samples 100
```

- Запуск отдельных техник:
```
python evaluate.py --technique zero_shot --samples 100
python evaluate.py --technique cot --samples 100
python evaluate.py --technique few_shot --samples 100
python evaluate.py --technique cot_few_shot --samples 100
```

## Результаты:

Все результаты сохранены в папку results:
- Тестирование через Ollama: [report.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/report.md)
- Тестирование через FastAPI: [report_fast_api.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/report_fast_api.md)
- Zero-shot: [zero_shot_report.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/zero_shot_report.md)
- CoT: [cot_report.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/cot_report.md)
- Few-shot: [few_shot_report.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/few_shot_report.md)
- CoT + few-shot: [cot_few_shot_report.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/cot_few_shot_report.md)
- Сравнительный анализ техник: [comparison.md](https://github.com/kr1st1na0/NLP_MAI/blob/main/results/comparison.md)
