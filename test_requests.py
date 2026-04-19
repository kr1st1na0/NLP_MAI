import requests
import argparse

API_URL = "http://localhost:8000"
OLLAMA_DIRECT = "http://localhost:11434/api/generate"

SMS_LIST = [
    "Вы выиграли миллион рублей! Перейдите по ссылке",
    "Мы идем завтра в кино?",
    "Встреча на выходных в силе?",
    "Мы рады сообщим Вам, что вы стали победителем в нашем розыгрыше бесплатного MacBook!",
    "Вас взломали! Срочно пришлите код, чтобы мы восстановили ваш аккаунт!",
    "Когда будешь дома?",
    "Не мог бы ты зайти в магазин и купить молока?",
    "Предлагаем вам оформить кредит под 5%",
    "Я ушла в театр. В холодильнике суп и макароны",
    "У нас для вас уникальное предложение! Инвестируйте и получайте по 100 тысяч рублей в неделю",
    "Вам заказали доставку цветов! Сообщите нам код из смс, чтобы мы смогли их Вам доставить"
]

def ask_llm(prompt, use_fastapi=True):
    """
    Отправляет запрос к LLM и возвращает ответ.
    """
    if use_fastapi:
        resp = requests.post(f"{API_URL}/generate", json={"prompt": prompt})
        return resp.json().get("response", "")
    else:
        resp = requests.post(OLLAMA_DIRECT, json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        })
        return resp.json().get("response", "")

def save_to_md(results, mode, filename="results/report.md"):
    """
    Сохраняет результаты тестирования в Markdown файл.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Отчет проверки спама\n\n")
        f.write(f"**Режим:** `{mode}`\n\n")
        f.write(f"**Всего сообщений:** `{len(results)}`\n\n")
        
        f.write("## Результаты\n\n")
        f.write("| № | Сообщение | Вердикт |\n")
        f.write("|---|-----------|---------|\n")
        
        for i, (sms, answer) in enumerate(results, 1):
            f.write(f"| {i} | {sms} | {answer} |\n")
        

def main():
    """
    Главная функция скрипта.
    
    Парсит аргументы командной строки, запускает тестирование всех SMS сообщений
    из списка в выбранном режиме и сохраняет результаты в отчёт.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fastapi", "ollama"], default="ollama")
    args = parser.parse_args()
    
    use_fastapi = (args.mode == "fastapi")
    results = []
    
    for sms in SMS_LIST:
        prompt = f"Ты классификатор спама в сообщениях. Определи сообщение является спамом или нет. Ответь кратко: 1, если это спам. Или 0, если не спам. Отвечай только 1 или 0! Сообщение: {sms}"
        answer = ask_llm(prompt, use_fastapi)
        results.append((sms, answer))

    if (use_fastapi):
        save_to_md(results, args.mode, "results/report_fast_api.md")
    else:
        save_to_md(results, args.mode, "results/report.md")

if __name__ == "__main__":
    main()