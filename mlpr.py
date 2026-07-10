import json
from openai import OpenAI
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
RIGHT_ANSWERS = r"C:\Users\User\PyCharmMiscProject\annotations.json"
OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
OPENAI_API_KEY = "key"
MODEL_NAME = "local-model"

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

ROLES = [
    "истец",
    "ответчик",
    "третье лицо",
    "иное"
]

def load_annotations(annotations_path):
    annotations = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                annotations.append(data)
            except json.JSONDecodeError as e:
                print(
                    f"Ошибка JSON в строке {line_number}: {e}"
                )
    print(
        f"Загружено {len(annotations)} документов из annotations.json"
    )
    return annotations

def determine_roles_with_openai(doc_data):
    resolution = doc_data.get("resolution", "")
    if len(resolution) > 8000:
        resolution = resolution[:8000]
    participants = []
    for p in doc_data.get("participants", []):
        participants.append({
            "inn": p.get("inn")
        })
    prompt = f"""
Ты эксперт по анализу арбитражных судебных решений.

Твоя задача определить процессуальную роль каждого участника дела.

Возможные роли:

- истец
- ответчик
- третье лицо
- иное


Участники:

{json.dumps(
    participants,
    ensure_ascii=False,
    indent=2
)}
Текст решения:

{resolution}
Правила анализа:
1. Определи истца по конструкции:
   "по иску X к Y"
2. Определи ответчика:
   лицо после "к"
3. Определи третьих лиц:
   конструкции:
   "при участии третьих лиц"
   "к участию привлечены"
4. Не выбирай роль по частоте слов.
5. Верни роль для каждого ИНН.
Ответ строго JSON:
{{
 "roles":[
    {{
      "inn":"123",
      "role":"истец"
    }}
 ]
}}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role":"system",
                "content":
                "Ты классификатор судебных участников. Отвечай только JSON."
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0,
        max_tokens=300,
        response_format={
            "type":"json_object"
        }
    )
    answer = response.choices[0].message.content
    print("\nОтвет модели:")
    print(answer)
    result = json.loads(answer)
    roles = {}
    for item in result.get("roles", []):
        inn = item.get("inn")
        role = item.get("role", "иное")
        if role not in ROLES:
            role = "иное"
        roles[inn] = role
    return roles

def build_confusion_matrix(annotations):
    y_true = []
    y_pred = []
    print("НАЧАЛО ОЦЕНКИ МОДЕЛИ")
    for doc_index, doc_data in enumerate(annotations):
        participants = doc_data.get("participants", [])
        print(
            f"\nДокумент {doc_index + 1}/{len(annotations)}"
        )
        print(
            f"Участников: {len(participants)}"
        )
        predicted_roles = determine_roles_with_openai(
            doc_data
        )
        for participant in participants:
            inn = participant.get("inn")
            true_role = participant.get("role")
            if not inn or not true_role:
                continue
            predicted_role = predicted_roles.get(
                inn,
                "иное"
            )
            print(
                f"ИНН: {inn}\n"
                f"Истина: {true_role}\n"
                f"Предсказание: {predicted_role}\n"
            )
            y_true.append(
                true_role.strip().lower()
            )
            y_pred.append(
                predicted_role.strip().lower()
            )
    if not y_true:
        print("Нет данных для оценки.")
        return None
    labels = ROLES
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )
    print(
        f"{'':15}",
        end=""
    )
    for label in labels:
        print(
            f"{label:15}",
            end=""
        )
    print()

    for i, label in enumerate(labels):

        print(
            f"{label:15}",
            end=""
        )
        for j in range(len(labels)):

            print(
                f"{cm[i][j]:15}",
                end=""
            )
        print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0
        )
    )
    accuracy = accuracy_score(
        y_true,
        y_pred
    )
    correct = sum(
        1
        for t, p in zip(y_true, y_pred)
        if t == p
    )
    total = len(y_true)
    print(
        f"Accuracy = {accuracy:.4f}"
    )
    print(
        f"Правильно: {correct}/{total}"
    )
    return cm

def main():
    print("Модель:", MODEL_NAME)
    print("API:", OPENAI_BASE_URL)
    annotations = load_annotations(RIGHT_ANSWERS)
    if not annotations:
        print("Аннотации не найдены.")
        return
    build_confusion_matrix(annotations)

if __name__ == "__main__":
    main()
