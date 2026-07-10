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


def determine_role_with_openai(inn, doc_data):
    resolution_text = doc_data.get('resolution', '')
    if len(resolution_text) > 2000:
        resolution_text = resolution_text[:2000] + "..."

    json_context = {
        "participants": doc_data.get('participants', []),
        "resolution": resolution_text
    }
    json_string = json.dumps(json_context, ensure_ascii=False, indent=2)

    prompt = f"""
Ты — системный ассистент по анализу данных. Перед тобой документ в формате JSON.
В массиве "participants" содержатся объекты с ИНН ("inn") и соответствующими ролями ("role").

Твоя задача: найди в предоставленном JSON объект, у которого "inn" равен "{inn}", и извлеки из него значение "role".

JSON документ:
{json_string}

Ответь строго в формате:
Роль: [истец / ответчик / третье лицо]
Пояснение: [одно предложение, подтверждающее извлечение из JSON]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "Ты - точный робот-парсер JSON, который извлекает данные строго по инструкции."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.0
        )

        response_text = response.choices[0].message.content.strip()
        print(f"    Ответ модели для ИНН {inn}: {response_text}")

        role_pattern = r"Роль:\s*([^\n]+)"
        match = re.search(role_pattern, response_text)

        if match:
            role_text = match.group(1).strip().lower()
            if "истец" in role_text:
                return "истец"
            elif "ответчик" in role_text:
                return "ответчик"
            elif "третье" in role_text or "3-е" in role_text:
                return "третье лицо"
            else:
                return "не определена"
        else:
            return "не определена"

    except Exception as e:
        print(f"Ошибка при запросе к OpenAI API: {e}")
        return "ошибка api"

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
            predicted_role = determine_role_with_openai(inn, doc_data)
            y_true.append(true_role)
            y_pred.append(predicted_role)
            print(f" ИНН: {inn} -> Истина: {true_role}, Предсказано: {predicted_role}")
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    if not y_true:
        print("Нет данных для оценки!")
        return [], [], None, None
    y_true = [str(i).strip().lower() for i in y_true]
    y_pred = [str(i).strip().lower() for i in y_pred]
    labels = ["истец", "ответчик", "третье лицо", "не определена"]
    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"{'':>15} " + " ".join(f"{label:>15}" for label in labels))
    for i, label in enumerate(labels):
        print(f"{label:>15} " + " ".join(f"{cm[i][j]:>15}" for j in range(len(labels))))
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
