import json
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

ROLES = ["истец", "ответчик", "третье лицо", "иное"]

Y_TRUE = []
Y_PRED = []

def _extract_json(raw_text):
    start = raw_text.find("{")
    if start == -1:
        return {}
    decoder = json.JSONDecoder()
    try:
        result, _ = decoder.raw_decode(raw_text[start:])
        return result
    except json.JSONDecodeError:
        print("Не удалось распарсить ответ модели:")
        print(raw_text)
        return {}

def exec_predict_role(resolution, participants):
    if len(resolution) > 4000:
        resolution = resolution[:4000]
    participants_payload = [{"inn": p.get("inn")} for p in participants]
    prompt = f"""
Ты эксперт по анализу арбитражных судебных решений.

Твоя задача определить процессуальную роль каждого участника дела.

Возможные роли:

- истец
- ответчик
- третье лицо
- иное


Участники:

{json.dumps(participants_payload, ensure_ascii=False, indent=2)}
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
Верни ТОЛЬКО JSON.

Запрещено:
- писать объяснения;
- писать комментарии;
- использовать Markdown;
- использовать ```json;
- писать любой текст после закрывающей фигурной скобки.

Допустимый формат ответа:

{{
  "roles": [
    {{
      "inn": "...",
      "role": "истец"
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "Ты классификатор судебных участников. Отвечай только JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1000,
        response_format={
            "type": "json_object"
        }
    )
    answer = response.choices[0].message.content
    print("\nОтвет модели:")
    print(answer)
    result = _extract_json(answer)
    roles_by_inn = {}
    for item in result.get("roles", []):
        inn = item.get("inn")
        role = str(item.get("role", "иное")).strip().lower()
        if role not in ROLES:
            role = "иное"
        roles_by_inn[inn] = role
    return roles_by_inn

def exec_predict_roles(resolution, participants):
    predicted_roles = exec_predict_role(resolution, participants)
    for p in participants:
        inn = p.get("inn")
        truth_role = str(p.get("role", "")).strip().lower()
        if not inn or not truth_role:
            continue
        predict_role = predicted_roles.get(inn, "иное")
        print(truth_role, predict_role)
        Y_TRUE.append(truth_role)
        Y_PRED.append(predict_role)

def processing_data(file_path, predict_roles_func):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line_number, l in enumerate(lines, start=1):
        l = l.strip()
        if not l:
            continue
        try:
            samples.append(json.loads(l))
        except json.JSONDecodeError as e:
            print(f"Ошибка JSON в строке {line_number}: {e}")
    print(f"Загружено {len(samples)} документов")
    for doc_index, sample in enumerate(samples, start=1):
        participants = sample.get("participants", [])
        print(f"\nДокумент {doc_index}/{len(samples)}")
        print(f"Участников: {len(participants)}")
        predict_roles_func(sample.get("resolution", ""), participants)
    build_confusion_matrix_report()

def build_confusion_matrix_report():
    if not Y_TRUE:
        print("Нет данных для оценки.")
        return None
    labels = ROLES
    cm = confusion_matrix(Y_TRUE, Y_PRED, labels=labels)
    print(f"{'':15}", end="")
    for label in labels:
        print(f"{label:15}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:15}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:15}", end="")
        print()
    print(classification_report(Y_TRUE, Y_PRED, labels=labels, zero_division=0))
    accuracy = accuracy_score(Y_TRUE, Y_PRED)
    correct = sum(1 for t, p in zip(Y_TRUE, Y_PRED) if t == p)
    total = len(Y_TRUE)
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Правильно: {correct}/{total}")
    return cm

if __name__ == "__main__":
    print("Модель:", MODEL_NAME)
    print("API:", OPENAI_BASE_URL)
    processing_data(RIGHT_ANSWERS, exec_predict_roles)
