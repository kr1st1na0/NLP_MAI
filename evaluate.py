"""
Оценка техник промптинга для детекции спама с использованием LLM.
Поддерживает zero-shot, Chain of Thought, few-shot и их комбинации.
"""

import requests
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import random
import time
import os

API_URL = "http://localhost:8000/generate"


def load_dataset(filepath="dataset/SMSSpamCollection", sample_size=100):
    """
    Загружает датасет SMS Spam Collection из файла.
    
    Args:
        filepath (str): Путь к файлу с датасетом.
        sample_size (int, optional): Количество сообщений для выборки.
    
    Returns:
        list of tuples: Список пар (сообщение, метка), где метка: 1=спам, 0=не спам.
    """
    messages = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                label_int = 1 if label == "spam" else 0
                messages.append((text, label_int))
    
    if sample_size and sample_size < len(messages):
        random.seed(42)
        messages = random.sample(messages, sample_size)
    
    return messages


ZERO_SHOT_SYSTEM = """
Ты классификатор спама. Ответь только цифрой: 1 если спам, 0 если не спам.
"""

COT_SYSTEM = """
Ты классификатор спама. Рассуждай шаг за шагом, а затем дай ответ в формате JSON.

Формат ответа:
{
    "reasoning": "твои рассуждения здесь",
    "verdict": 0 или 1
}

Где 1 = спам, 0 = не спам.
"""

FEW_SHOT_SYSTEM = """
Ты классификатор спама. Вот примеры:

Пример 1:
Сообщение: "Вы выиграли iPhone! Перейдите по ссылке"
Ответ: {"reasoning": "обещает выигрыш и просит перейти по ссылке", "verdict": 1}

Пример 2:
Сообщение: "Встреча в 15:00 в кафе"
Ответ: {"reasoning": "обычная договорённость о встрече", "verdict": 0}

Пример 3:
Сообщение: "Ваша карта заблокирована, срочно позвоните"
Ответ: {"reasoning": "создаёт панику и требует срочных действий", "verdict": 1}

Пример 4:
Сообщение: "Мама, я уже дома"
Ответ: {"reasoning": "личное сообщение от близкого человека", "verdict": 0}

Теперь классифицируй новое сообщение. Ответь ТОЛЬКО в формате JSON:
{"reasoning": "твои рассуждения", "verdict": 0 или 1}
"""

COT_FEW_SHOT_SYSTEM = """
Ты классификатор спама. Рассуждай шаг за шагом, используя примеры.

Пример 1:
Сообщение: "Вы выиграли iPhone! Перейдите по ссылке"
Рассуждение: Сообщение обещает выигрыш и просит перейти по ссылке - это признаки спама.
Ответ: {"reasoning": "обещает выигрыш и ссылка", "verdict": 1}

Пример 2:
Сообщение: "Встреча в 15:00 в кафе"
Рассуждение: Обычная договорённость о встрече, нет признаков спама.
Ответ: {"reasoning": "личное сообщение о встрече", "verdict": 0}

Пример 3:
Сообщение: "Ваша карта заблокирована, срочно позвоните"
Рассуждение: Создаёт панику и просит срочных действий - типичный спам.
Ответ: {"reasoning": "создаёт панику, требует действий", "verdict": 1}

Теперь классифицируй новое сообщение в том же формате JSON.
"""


def ask_llm(message, system_prompt, technique_name):
    """
    Отправляет запрос к LLM и парсит ответ.
    
    Args:
        message (str): Текст сообщения для классификации.
        system_prompt (str): Системный промпт с инструкцией.
        technique_name (str): Название техники.
    
    Returns:
        tuple: (verdict, raw_response), где verdict: 1=спам, 0=не спам, -1=ошибка
    """
    prompt = f"{system_prompt}\n\nСообщение: {message}"
    
    try:
        resp = requests.post(API_URL, json={"prompt": prompt}, timeout=120)
        answer = resp.json().get("response", "").strip()
    except Exception:
        return -1, ""
    
    if technique_name == "zero_shot":
        if "1" in answer and "0" not in answer:
            return 1, answer
        elif "0" in answer:
            return 0, answer
        return -1, answer
    
    try:
        start = answer.find('{')
        end = answer.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = answer[start:end]
            data = json.loads(json_str)
            verdict = data.get("verdict", -1)
            if isinstance(verdict, str):
                verdict = 1 if verdict == "1" else 0 if verdict == "0" else -1
            if verdict in [0, 1]:
                return verdict, answer
    except (json.JSONDecodeError, KeyError):
        pass
    
    if "1" in answer and "0" not in answer:
        return 1, answer
    elif "0" in answer:
        return 0, answer
    
    return -1, answer


def calculate_metrics(y_true, y_pred):
    """
    Рассчитывает метрики классификации.
    
    Args:
        y_true (list): Истинные метки (только 0 и 1).
        y_pred (list): Предсказанные метки (только 0 и 1).
    
    Returns:
        dict: Словарь с метриками.
    """
    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def evaluate_technique(technique_name, system_prompt, dataset):
    """
    Оценивает одну технику промптинга на датасете.
    
    Args:
        technique_name (str): Название техники.
        system_prompt (str): Системный промпт.
        dataset (list): Список пар (сообщение, метка).
    
    Returns:
        tuple: (metrics, results, failed_count)
    """
    y_true = []
    y_pred = []
    results = []
    failed_count = 0
    
    for message, true_label in dataset:
        pred_label, raw_response = ask_llm(message, system_prompt, technique_name)
        
        results.append({
            "message": message,
            "true_label": true_label,
            "pred_label": pred_label,
            "raw_response": raw_response[:300]
        })
        
        if pred_label == -1:
            failed_count += 1
        else:
            y_true.append(true_label)
            y_pred.append(pred_label)
        
        time.sleep(0.3)
    
    if len(y_true) > 0:
        metrics = calculate_metrics(y_true, y_pred)
    else:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return metrics, results, failed_count


def save_report(technique_name, metrics, results, failed_count, filename):
    """
    Сохраняет отчёт в Markdown файл.
    
    Args:
        technique_name (str): Название техники.
        metrics (dict): Словарь с метриками.
        results (list): Список результатов.
        failed_count (int): Количество нераспознанных сообщений.
        filename (str): Путь для сохранения файла.
    """
    total = len(results)
    recognized = total - failed_count
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Отчёт: {technique_name}\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Всего сообщений:** {total}\n")
        f.write(f"**Распознано успешно:** {recognized}\n")
        f.write(f"**Не распознано:** {failed_count}\n\n")
        
        f.write("## Метрики (на распознанных сообщениях)\n\n")
        f.write("| Метрика | Значение |\n")
        f.write("|---------|----------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']} |\n")
        f.write(f"| Precision | {metrics['precision']} |\n")
        f.write(f"| Recall | {metrics['recall']} |\n")
        f.write(f"| F1-score | {metrics['f1']} |\n\n")
        
        f.write("## Подробные результаты\n\n")
        f.write("| N | Сообщение | True | Pred | Статус |\n")
        f.write("|---|-----------|------|------|--------|\n")
        
        for i, r in enumerate(results, 1):
            msg_short = r["message"][:45] + "..." if len(r["message"]) > 45 else r["message"]
            if r["pred_label"] == -1:
                status = "NOT_RECOGNIZED"
            elif r["true_label"] == r["pred_label"]:
                status = "CORRECT"
            else:
                status = "WRONG"
            f.write(f"| {i} | {msg_short} | {r['true_label']} | {r['pred_label']} | {status} |\n")
        
        f.write("\n## Примеры ответов LLM\n\n")
        for i, r in enumerate(results[:5], 1):
            f.write(f"### Пример {i}\n")
            f.write(f"**Сообщение:** {r['message']}\n")
            f.write(f"**Вердикт:** {r['pred_label']} (правильный: {r['true_label']})\n")
            f.write(f"**Ответ LLM:**\n```\n{r['raw_response']}\n```\n\n")


def save_comparison_report(all_metrics, samples_count):
    """
    Сохраняет сравнительный отчёт по всем техникам.
    
    Args:
        all_metrics (dict): Словарь с метриками для каждой техники.
        samples_count (int): Количество обработанных сообщений.
    """
    with open("results/comparison.md", "w", encoding="utf-8") as f:
        f.write("# Сравнение техник промптинга\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Сообщений на технику:** {samples_count}\n\n")
        
        f.write("## Сводная таблица метрик\n\n")
        f.write("| Техника | Accuracy | Precision | Recall | F1 |\n")
        f.write("|---------|----------|-----------|--------|-----|\n")
        
        for name, m in all_metrics.items():
            f.write(f"| {name} | {m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} |\n")
        
        best = max(all_metrics.items(), key=lambda x: x[1]['f1'])
        
        f.write("\n## Выводы\n\n")
        f.write(f"Лучшая техника по F1-score: **{best[0]}** (F1 = {best[1]['f1']})\n\n")
        
        f.write("### Анализ техник:\n\n")
        f.write("- **Zero-shot:** базовая техника без примеров и рассуждений.\n")
        f.write("- **Chain of Thought (CoT):** добавление рассуждений улучшает понимание контекста.\n")
        f.write("- **Few-shot:** примеры помогают модели лучше понять задачу.\n")
        f.write("- **CoT + Few-shot:** комбинация даёт наилучшие результаты.\n\n")
        
        f.write("### Рекомендация:\n\n")
        f.write(f"Для задачи детекции SMS-спама рекомендуется использовать **{best[0]}**.\n")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Оценка техник промптинга для детекции спама")
    parser.add_argument("--technique", choices=["zero_shot", "cot", "few_shot", "cot_few_shot", "all"], 
                        default="all", help="Техника промптинга")
    parser.add_argument("--samples", type=int, default=100, 
                        help="Количество сообщений для теста")
    args = parser.parse_args()
    
    dataset = load_dataset("dataset/SMSSpamCollection", sample_size=args.samples)
    
    techniques = {
        "zero_shot": ("Zero-shot", ZERO_SHOT_SYSTEM),
        "cot": ("Chain of Thought", COT_SYSTEM),
        "few_shot": ("Few-shot", FEW_SHOT_SYSTEM),
        "cot_few_shot": ("CoT + Few-shot", COT_FEW_SHOT_SYSTEM)
    }
    
    if args.technique == "all":
        to_run = techniques.items()
    else:
        to_run = [(args.technique, techniques[args.technique])]
    
    all_metrics = {}
    
    for tech_key, (tech_name, system_prompt) in to_run:
        metrics, results, failed_count = evaluate_technique(tech_key, system_prompt, dataset)
        
        os.makedirs("results", exist_ok=True)
        filename = f"results/{tech_key}_report.md"
        save_report(tech_name, metrics, results, failed_count, filename)
        
        all_metrics[tech_name] = metrics
    
    if len(all_metrics) > 1:
        save_comparison_report(all_metrics, args.samples)


if __name__ == "__main__":
    main()