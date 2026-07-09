import json
import re
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

RIGHT_ANSWERS = r"C:\Users\User\PyCharmMiscProject\annotations.json"
OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
OPENAI_API_KEY = "key"
MODEL_NAME = "local-model"

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)


def load_annotations(annotations_path):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r'(\{"participants":\s*\[.*?\],\s*"resolution":\s*".*?"\})'
    matches = re.findall(pattern, content, re.DOTALL)

    annotations = []
    for match in matches:
        try:
            data = json.loads(match)
            annotations.append(data)
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга: {e}")
            continue

    print(f"Загружено {len(annotations)} документов из annotations.json")
    return annotations


def determine_role_with_openai(inn, document_text, document_index):
    if len(document_text) > 3000:
        document_text = document_text[:3000] + "..."

    prompt = f"""
Ты - профессиональный юридический ассистент. Твоя задача - найти в тексте судебного документа конкретный ИНН и точно определить роль этого лица в судебном процессе.

ИНН, который нужно найти: {inn}

Правила определения ролей:
1. В судебных документах роли указываются явно в начале документа или в описании участников процесса
2. Форматы указания ролей:
   - "Истец: ООО "Название", ИНН XXXXX" → это Истец
   - "Ответчик: ООО "Название", ИНН XXXXX" → это Ответчик  
   - "Третье лицо: ООО "Название", ИНН XXXXX" → это Третье лицо
   - "по иску [кого-то]" → тот, кто подает иск - Истец
   - "к [кому-то]" → тот, к кому предъявлен иск - Ответчик

3. ВАЖНО: Ищи именно ИНН {inn} в тексте. Не путай с другими ИНН!

Текст документа:
{document_text}

Найди в тексте ИНН {inn} и определи его роль.
Ответь строго в формате:
Роль: [Истец/Ответчик/Третье лицо]
Пояснение: [одно предложение, почему ты так решил, с указанием места в тексте, где найдена эта информация]
ВНИМАНИЕ: В тексте могут быть упоминания нескольких организаций с разными ИНН.
Тебя интересует ТОЛЬКО организация с ИНН {inn}.
Найди в тексте именно этот ИНН и определи роль ТОЛЬКО для него.
Не путай с другими ИНН, даже если они упоминаются рядом!
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
            temperature=0.0
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
                    predicted_role = known_role
                    break
            else:
                predicted_role = "Не определена"
        else:
            predicted_role = "Не определена"

        return predicted_role

    except Exception as e:
        print(f"Ошибка при запросе к OpenAI API: {e}")
        return "Ошибка API"

def build_confusion_matrix(annotations):
    y_true = []
    y_pred = []

    print("НАЧАЛО ОЦЕНКИ МОДЕЛИ")

    for doc_index, doc_data in enumerate(annotations):
        document_text = doc_data.get('resolution', '')
        participants = doc_data.get('participants', [])
        print(f"\nДокумент {doc_index + 1}/{len(annotations)}")
        print(f"Участников в документе: {len(participants)}")

        for participant in participants:
            inn = participant.get('inn')
            true_role = participant.get('role')

            if not inn or not true_role:
                continue
            predicted_role = determine_role_with_openai(inn, document_text, doc_index)
            y_true.append(true_role)
            y_pred.append(predicted_role)
            print(f" ИНН: {inn} -> Истина: {true_role}, Предсказано: {predicted_role}")
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    if not y_true:
        print("Нет данных для оценки!")
        return [], [], None, None
    labels = sorted(set(y_true + y_pred))
    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("   " + " ".join(f"{label:>12}" for label in labels))
    for i, label in enumerate(labels):
        print(f"{label:>12} " + " ".join(f"{cm[i][j]:>12}" for j in range(len(labels))))
    print("\nОТЧЕТ О КЛАССИФИКАЦИИ:")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    accuracy = accuracy_score(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    print(f"\nОБЩАЯ ТОЧНОСТЬ МОДЕЛИ: {accuracy:.4f} ({correct}/{total} правильных)")
    return y_true, y_pred, cm, report

def main():
    print(f"API URL: {OPENAI_BASE_URL}")
    print(f"Модель: {MODEL_NAME}")
    print("=" * 80)
    annotations = load_annotations(RIGHT_ANSWERS)
    if not annotations:
        print("Не удалось загрузить аннотации")
        return

    print(f"\nВсего документов для обработки: {len(annotations)}")
    print("=" * 80)
    y_true, y_pred, cm, report = build_confusion_matrix(annotations)

if __name__ == "__main__":
    main()
