# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:35:13 2025

@author: Mina
"""

# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import cohere

# df = pd.read_csv("plant_diseases_treatment.csv")
# # Combine text
# def combine_columns(row):
#     return f"النبات: {row['اسم النبات']}, المرض: {row['اسم المرض']}, العلاج: {row['العلاج']}, طريقة الرش: {row['طريقة الرش']}, توقيت الرش: {row['توقيت الرش']}, إجراءات إضافية: {row['إجراءات إضافية']}"

# texts = df.apply(combine_columns, axis=1).tolist()
# # Embedding model
# embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# embeddings = embedding_model.encode(texts, show_progress_bar=True)

# # Create FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)
# # Search function
# #def retrieve_answer(question):
#     #query_vector = embedding_model.encode([question])
#    # D, I = index.search(query_vector, k=1)
#   #  return texts[I[0][0]]
# def extract_crop_from_question(question):
#     for crop in df["اسم النبات"].unique():
#         if crop in question:
#             return crop
#     return None

# def retrieve_answer(question):
#     crop = extract_crop_from_question(question)
#     if crop is None:
#         return "لا يمكن تحديد اسم النبات من السؤال."

#     filtered_df = df[df["اسم النبات"] == crop]
#     if filtered_df.empty:
#         return "لا توجد أمراض لهذا النبات في البيانات."

#     texts = filtered_df.apply(combine_columns, axis=1).tolist()
#     filtered_embeddings = embedding_model.encode(texts)

#     temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
#     temp_index.add(filtered_embeddings)

#     query_vector = embedding_model.encode([question])
#     D, I = temp_index.search(query_vector, k=3)  # ✅ زودنا عدد النتائج

#     retrieved_texts = [texts[i] for i in I[0]]
#     return " ".join(retrieved_texts)  # ✅ جمعهم كلهم في نص واحد
# co = cohere.Client("0Qi0p8wQzYzRkXND2ROae9kILPwQOXECDosUqxEU")  # ← استبدل بـ API key بتاعك

# def generate_response(question):
#     retrieved_context = retrieve_answer(question)
#     prompt = f"""
# السؤال: {question}
# المعلومات: {retrieved_context}
# أجب بدقة بجملة واحدة فقط دون إضافة أي معلومات خارجية.
# """
#     response = co.generate(
#         model="command-r-plus",
#         prompt=prompt,
#         max_tokens=100,
#         temperature=0.3,
#     )
#     return response.generations[0].text.strip()

# print("الرد:", generate_response("ما هو علاج مرض جرب التفاح ؟"))

# print("الرد:", generate_response("ما هو توقيت الرش  مرض جرب التفاح ؟"))

# print("الرد:", generate_response("ما هي امراض الطماطم ؟"))

# print("الرد:", generate_response("ما هي امراض التفاح ؟"))

# print("الرد:", generate_response("ما هي امراض البطاطس  ؟"))

# print("الرد:", generate_response("ما هي الاجراءات الاضافيه لمرض جرب التفاح ؟"))


##########################################################################################################


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import cohere


# Combine text function
def combine_columns(row):
    return f"النبات: {row['اسم النبات']}, المرض: {row['اسم المرض']}, العلاج: {row['العلاج']}, طريقة الرش: {row['طريقة الرش']}, توقيت الرش: {row['توقيت الرش']}, إجراءات إضافية: {row['إجراءات إضافية']}"

# Extract crop function
def extract_crop_from_question(question):
    # Assuming df is loaded
    # Need to load df if not already loaded
    try:
        df_exists = 'df' in globals()
    except NameError:
        df_exists = False

    if not df_exists:
        df = pd.read_csv("plant_diseases_treatment.csv")

    for crop in df["اسم النبات"].unique():
        if crop in question:
            return crop
    return None

# Retrieve answer function
def retrieve_answer(question):
    crop = extract_crop_from_question(question)
    if crop is None:
        return "لا يمكن تحديد اسم النبات من السؤال."

    # Assuming df is loaded
    # Need to load df if not already loaded
    try:
        df_exists = 'df' in globals()
    except NameError:
        df_exists = False

    if not df_exists:
        df = pd.read_csv("plant_diseases_treatment.csv")

    filtered_df = df[df["اسم النبات"] == crop]
    if filtered_df.empty:
        return "لا توجد أمراض لهذا النبات في البيانات."

    texts = filtered_df.apply(combine_columns, axis=1).tolist()
    # Assuming embedding_model is loaded
    # Need to load embedding_model if not already loaded
    try:
        embedding_model_exists = 'embedding_model' in globals()
    except NameError:
        embedding_model_exists = False

    if not embedding_model_exists:
         embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    filtered_embeddings = embedding_model.encode(texts)

    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)

    query_vector = embedding_model.encode([question])
    D, I = temp_index.search(query_vector, k=3)

    retrieved_texts = [texts[i] for i in I[0]]
    return " ".join(retrieved_texts)

# Generate response function
def generate_response(question):
    retrieved_context = retrieve_answer(question)
    prompt = f"""
السؤال: {question}
المعلومات: {retrieved_context}
أجب بدقة بجملة واحدة فقط دون إضافة أي معلومات خارجية.
"""
    # Assuming co is loaded
    # Need to load co if not already loaded
    try:
        co_exists = 'co' in globals()
    except NameError:
        co_exists = False

    if not co_exists:
        co = cohere.Client("0Qi0p8wQzYzRkXND2ROae9kILPwQOXECDosUqxEU") # Replace with your API key

    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=100,
        temperature=0.3,
    )
    return response.generations[0].text.strip()


app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    answer = generate_response(q.question)
    return {"question": q.question, "answer": answer}