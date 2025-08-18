from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
import os

# 1. Lokale Embeddings und DB laden
print("Lade lokalen Retrieval-Stack...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("medical_kb")

# 2. Hugging Face API einrichten
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Bitte setze den HF_TOKEN als Umgebungsvariable: export HF_TOKEN=dein_token")

# Funktionierendes Modell verwenden
llm_client = InferenceClient(
    model="google/flan-t5-xxl",  # ← Geändert
    token=HF_TOKEN
)

def generate_answer(prompt):
    try:
        response = llm_client.text_generation(
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.5
        )
        return response
    except Exception as e:
        print("Fehler bei API-Antwort:", str(e))
        return "Leider konnte keine Antwort generiert werden."

# 3. Chat-Loop
print("\n⚕️ Hybrid-RAG Chatbot (lokal Retrieval + Cloud Generierung) (Tippe 'ende' zum Beenden)\n")

while True:
    user_input = input("Du: ")
    if user_input.lower() == "ende":
        break

    # 4. Lokales Retrieval
    query_embedding = embedder.encode(user_input)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context = "\n".join(results["documents"][0])

    # 5. Prompt für Modell erstellen
    prompt = (
        "Du bist ein medizinischer Experte. Beantworte die Frage präzise basierend auf dem Kontext.\n\n"
        f"Kontext: {context}\n\n"
        f"Frage: {user_input}\n"
        f"Antwort:"
    )

    # 6. Antwort von Cloud-API holen
    print("Generiere Antwort...")
    full_response = generate_answer(prompt)

    # 7. Antwort bereinigen
    if "Antwort:" in full_response:
        answer = full_response.split("Antwort:")[-1].strip()
    else:
        answer = full_response

    print(f"Bot: {answer}\n")
