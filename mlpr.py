import json
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

RIGHT_ANSWERS = r"C:\Users\User\PyCharmMiscProject\mlpr-project\annotations.json"
OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
OPENAI_API_KEY = "key"
MODEL_NAME = "local-model"

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY
)

ROLES = ["истец", "ответчик", "третье лицо", "иное"]

HEAD_CHARS = 4000
TAIL_CHARS = 4000

Y_TRUE = []
Y_PRED = []

def build_excerpt(text, head_chars=HEAD_CHARS, tail_chars=TAIL_CHARS):
    if len(text) <= head_chars + tail_chars:
        return text
    head = text[:head_chars]
    tail = text[-tail_chars:]
    return f"{head}\n\n...\n\n{tail}"

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

def exec_predict_role(resolution, participants, _retries_left=1):
    resolution = build_excerpt(resolution)

    participants_payload = [{"inn": p.get("inn")} for p in participants]

    prompt = f"""
Ты эксперт по анализу арбитражных судебных решений.

Твоя задача определить процессуальную роль каждого участника дела.

Возможные роли:

- истец
- ответчик
- третье лицо
- иное

ВАЖНО про роль "иное":
Роль "иное" — редкое исключение, а не запасной вариант на случай сомнений.
Используй "иное" ТОЛЬКО если участник явно не истец, не ответчик и не третье
лицо (например, представитель суда, эксперт, оценщик, финансовый управляющий).
Если участник упомянут как сторона дела, но ты не до конца уверен — всё равно
выбирай наиболее вероятную из трёх основных ролей (истец / ответчик / третье
лицо), а не "иное". Почти все участники судебного дела относятся к одной из
этих трёх ролей.

Правила анализа:
1. Определи истца по конструкции:
   "по иску X к Y"
2. Определи ответчика:
   лицо после "к". Если после "к" перечислено несколько лиц через запятую
   или союз "и" — все они являются ответчиками.
3. Определи третьих лиц по конструкциям:
   "при участии третьего лица"
   "при участии третьих лиц"
   "к участию в деле привлечен(о/ы)"
   "с привлечением третьего лица"
   Третьи лица обычно перечисляются отдельным списком после указания истца
   и ответчика и не являются стороной иска напрямую.
4. Не выбирай роль по частоте слов или по положению в тексте — только по
   явной формулировке роли участника.
5. В тексте рядом с ИНН часто встречаются ДРУГИЕ похожие по виду номера —
   ОГРН, ОГРНИП, КПП. Это НЕ ИНН, не путай их между собой:
   - ИНН организации — 10 цифр, ИНН ИП — 12 цифр.
   - ОГРН — 13 цифр, ОГРНИП — 15 цифр, КПП — 9 цифр.
   В поле "inn" итогового ответа ты ОБЯЗАН вернуть ТОЧНО ТО ЖЕ значение,
   которое было передано тебе в поле "inn" входного списка участников —
   скопируй его без изменений.

КРИТИЧЕСКИ ВАЖНО:
В списке участников ниже ровно {len(participants_payload)} человек/организаций
(по числу ИНН). Ты ОБЯЗАН вернуть в массиве "roles" РОВНО {len(participants_payload)}
объектов — по одному для КАЖДОГО ИНН из списка, без единого пропуска, даже если
ты не до конца уверен в роли — в этом случае выбери наиболее вероятную роль.
Не останавливайся после первого участника. Пройди по списку до конца.
Значения "inn" в твоём ответе должны СОВПАДАТЬ СИМВОЛ В СИМВОЛ со значениями
"inn" из списка участников ниже.

Участники ({len(participants_payload)} шт.):

{json.dumps(participants_payload, ensure_ascii=False, indent=2)}
Текст решения:

{resolution}

Верни ТОЛЬКО JSON.

Запрещено:
- писать объяснения;
- писать комментарии;
- использовать Markdown;
- использовать ```json;
- писать любой текст после закрывающей фигурной скобки;
- пропускать участников — каждый ИНН из списка выше должен получить роль.

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
    missing = [p for p in participants if p.get("inn") not in roles_by_inn]
    if missing and _retries_left > 0:
        print(f"Модель пропустила {len(missing)} участников, повторный запрос")
        retry_roles = exec_predict_role(
            resolution, missing, _retries_left=_retries_left - 1
        )
        roles_by_inn.update(retry_roles)
    elif missing:
        print(f"Не удалось получить роль для {len(missing)} участников даже после повтора")
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