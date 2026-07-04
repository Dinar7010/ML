import json
import os
import re
from pypdf import PdfReader
from openai import OpenAI

model = r"C:\Users\User\PyCharmMiscProject\model\T-lite-it-2.1-Q4_K_M.gguf"
documents = r"C:\Users\User\PyCharmMiscProject\documents"
OUTPUT_JSON_PATH = r"C:\Users\User\PyCharmMiscProject\results.json"

OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
OPENAI_API_KEY = "ключ"
MODEL_NAME = "local-model"

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def find_all_inn(text):
    inn_pattern = r'\b\d{10}\b|\b\d{12}\b'
    all_inn = re.findall(inn_pattern, text)
    unique_inn = []
    for inn in all_inn:
        if inn not in unique_inn:
            unique_inn.append(inn)
    return unique_inn


def determine_role_with_openai(inn, document_text, file_name):
    if len(document_text) > 3000:
        document_text = document_text[:3000] + "..."
    prompt = f"""
Ты - профессиональный юридический ассистент. Проанализируй текст судебного документа.

Найди человека или компанию с ИНН {inn} и определи его/её роль в судебном процессе.

В судебных документах роли обычно указаны явно:
- "Истец: ООО "Ромашка", ИНН 1234567890" → это Истец
- "Ответчик: Петров П.П., ИНН 0987654321" → это Ответчик
- "Третье лицо: ООО "Василек", ИНН 1122334455" → это Третье лицо
- "Взыскатель: ... ИНН ..." → это Истец
- "Должник: ... ИНН ..." → это Ответчик
- "Заявитель: ... ИНН ..." → это Истец или Заявитель

Вот текст документа:
{document_text}

Найди ИНН {inn} в тексте и определи его роль.
Ответь строго в формате:
Роль: [Истец/Ответчик/Третье лицо/Взыскатель/Должник/Заявитель/Не найден]
Пояснение: [одно предложение, почему ты так решил]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "Ты - полезный юридический ассистент, который строго следует инструкциям."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )

        response_text = response.choices[0].message.content.strip()
        print(f"    Ответ модели для ИНН {inn}: {response_text}")

        role_pattern = r"Роль:\s*([^\n]+)"
        match = re.search(role_pattern, response_text)

        if match:
            role_text = match.group(1).strip()
            known_roles = ["Истец", "Ответчик", "Третье лицо", "Взыскатель", "Должник", "Заявитель"]
            for known_role in known_roles:
                if known_role in role_text:
                    role = known_role
                    break
            else:
                role = "Не определена"
        else:
            role = "Не определена"
        return {
            'inn': inn,
            'role': role,
            'full_response': response_text
        }

    except Exception as e:
        print(f"Ошибка при запросе к OpenAI API: {e}")
        return {
            'inn': inn,
            'role': 'Ошибка API',
            'full_response': str(e)
        }

def process_document(file_path, file_name):
    text = read_pdf(file_path)
    if not text or len(text) < 100:
        print(f"{file_name}: Текст не извлечен")
        return None
    all_inn = find_all_inn(text)
    if not all_inn:
        print(f"{file_name}: ИНН не найдены")
        return None
    print(f"{file_name}: Найдены ИНН: {', '.join(all_inn)}")
    participants = []
    for inn in all_inn:
        result = determine_role_with_openai(inn, text, file_name)
        participants.append({
            'inn': result['inn'],
            'role': result['role']
        })
        print(f"ИНН: {result['inn']} -> Роль: {result['role']}")

    return {
        'fileName': file_name,
        'participants': participants
    }


def save_results_to_json(results):
    json_output = json.dumps(results, ensure_ascii=False, indent=2)
    print("РЕЗУЛЬТАТЫ В ФОРМАТЕ JSON:")
    print(json_output)
    print("=" * 60)

def main():
    print(f"API URL: {OPENAI_BASE_URL}")
    print(f"Модель: {MODEL_NAME}")
    all_files = os.listdir(documents)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("PDF файлы не найдены в папке documents.")
        return
    print(f"\nНайдено {len(pdf_files)} PDF файлов для обработки.\n")
    results = []
    for file_name in pdf_files:
        print(f"Обработка: {file_name}")
        full_path = os.path.join(documents, file_name)
        result = process_document(full_path, file_name)
        if result:
            results.append(result)
    save_results_to_json(results, OUTPUT_JSON_PATH)
if __name__ == "__main__":
    main()
