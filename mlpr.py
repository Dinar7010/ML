from llama_cpp import Llama
import re
import os
from pypdf import PdfReader

model_path = r"C:\Users\User\PyCharmMiscProject\model\T-lite-it-2.1-Q4_K_M.gguf"

neural_network = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_gpu_layers=-1,
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

def determine_role(inn, document_text):
    if len(document_text) > 3000:
        document_text = document_text[:3000] + "..."

    question = f"""
Ты - юридический помощник. Найди в тексте человека или компанию с ИНН  {inn}.
Определи, кто это: Истец, Ответчик или Третье лицо.

Текст документа:
{document_text}

Ответь строго в формате:
Роль: [Истец/Ответчик/Третье лицо/Не найден]
"""

    response = neural_network(
        question,
        max_tokens=100,
        temperature=0.1,
        echo=False
    )

    response_text = response['choices'][0]['text'].strip()

    match = re.search(r"Role:\s*(Истец|Ответчик|Третье лицо|Не найден)", response_text)

    if match:
        role = match.group(1)
    else:
        role = "Not Determined"

    return {
        'INN': inn,
        'Role': role,
        'Full_Response': response_text
    }

def main():
    documents_folder = r"C:\Users\User\PyCharmMiscProject\documents"
    pdf_files = os.listdir(documents_folder)
    for file_name in pdf_files:
        print(f"\n {file_name}")
        full_path = os.path.join(documents_folder, file_name)
        text = read_pdf(full_path)
        all_inn = find_all_inn(text)
        if not all_inn:
            print(" Не найдены ИНН")
            continue
        print(f"Найденные ИНН: {', '.join(all_inn)}")
        for inn in all_inn:
            result = determine_role(inn, text)
            print(f"  INN: {inn} -> Role: {result['Role']}")
main()