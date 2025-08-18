import gradio as gr
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

# Llama 3.1 mit conversational API
llm_client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)


# 3. Lernmodus-Funktion
def save_to_knowledge(question, answer):
    """Speichert neue Erkenntnisse in der Wissensdatenbank"""
    new_entry = f"Frage: {question}\nAntwort: {answer}"

    existing_ids = collection.get()["ids"]
    new_id = f"learned_{len(existing_ids)}"

    collection.add(
        ids=[new_id],
        embeddings=embedder.encode([new_entry]),
        documents=[new_entry]
    )

    return f"✅ Wissenspunkt gespeichert! (ID: {new_id})"


# 4. Globale Variable für letzte Interaktion
last_interaction = {"question": "", "answer": ""}


def generate_answer(messages):
    try:
        response = llm_client.chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Fehler bei API-Antwort: {str(e)}"


def chatbot_fn(user_input, history):
    global last_interaction

    # Lernmodus aktivieren
    if user_input.lower().strip() in ["lerne das", "merke das", "speichere das"]:
        if last_interaction["question"]:
            result = save_to_knowledge(
                last_interaction["question"],
                last_interaction["answer"]
            )
            return result
        else:
            return "❌ Keine vorherige Interaktion zum Speichern gefunden."

    # Lokales Retrieval
    query_embedding = embedder.encode(user_input)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    context = "\n".join(results["documents"][0])

    # Verbesserter Prompt für Llama 3.1
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein medizinischer Experte. Antworte präzise und sachlich basierend auf dem bereitgestellten Kontext. "
                "Verwende nur Informationen aus dem Kontext. "
                "Wenn die Antwort im Kontext nicht enthalten ist, sage höflich: 'Dazu habe ich keine Informationen im Kontext.' "
                "Füge keine eigenen Interpretationen hinzu."
            )
        },
        {
            "role": "user",
            "content": f"Kontext: {context}\n\nFrage: {user_input}"
        }
    ]

    # Antwort von Cloud-API holen
    response = generate_answer(messages)

    # Letzte Interaktion merken
    last_interaction = {
        "question": user_input,
        "answer": response
    }

    return response


# Gradio Interface
interface = gr.ChatInterface(
    fn=chatbot_fn,
    title="⚕️ Medizinischer RAG-Chatbot (mit Lernmodus)",
    description="Tippe 'lerne das' nach einer Antwort, um sie zu speichern | Llama 3.1 + Lokale Wissensdatenbank",
    examples=[
        ["Was ist eine Grippe?"],
        ["lerne das"],
        ["Wie erkennt man Bluthochdruck?"],
        ["Ich habe eine Chilikernunverträglichkeit!"],
        ["lerne das"]
    ]
)

if __name__ == "__main__":
    interface.launch()
