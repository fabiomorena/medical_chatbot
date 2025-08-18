import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb
import json

# 1. Verbindung zur Datenbank
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("medical_kb")


# 2. Funktionen für Admin-Interface

def list_all_entries():
    """Zeigt alle Einträge in der Datenbank"""
    try:
        results = collection.get()
        entries = []
        for i, (doc, id_) in enumerate(zip(results["documents"], results["ids"])):
            entries.append(f"**ID:** {id_}\n**Inhalt:** {doc}\n---")
        return "\n".join(entries) if entries else "Keine Einträge gefunden."
    except Exception as e:
        return f"Fehler beim Laden: {str(e)}"


def add_entry(content):
    """Fügt neuen Eintrag hinzu"""
    try:
        # Neue ID generieren
        existing_ids = collection.get()["ids"]
        new_id = f"manual_{len(existing_ids)}"

        # Speichern
        collection.add(
            ids=[new_id],
            embeddings=embedder.encode([content]),
            documents=[content]
        )
        return f"✅ Eintrag hinzugefügt! (ID: {new_id})"
    except Exception as e:
        return f"Fehler beim Hinzufügen: {str(e)}"


def delete_entry(entry_id):
    """Löscht Eintrag mit gegebener ID"""
    try:
        collection.delete(ids=[entry_id])
        return f"✅ Eintrag '{entry_id}' gelöscht!"
    except Exception as e:
        return f"Fehler beim Löschen: {str(e)}"


def search_entries(query):
    """Sucht Einträge basierend auf Query"""
    try:
        query_embedding = embedder.encode(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        entries = []
        for doc, id_, score in zip(results["documents"][0], results["ids"][0], results["distances"][0]):
            entries.append(f"**ID:** {id_}\n**Score:** {score:.4f}\n**Inhalt:** {doc}\n---")
        return "\n".join(entries) if entries else "Keine Ergebnisse gefunden."
    except Exception as e:
        return f"Fehler bei der Suche: {str(e)}"


# 3. Gradio Interface erstellen
with gr.Blocks(title="🧠 Admin-Interface für Wissensdatenbank") as demo:
    gr.Markdown("# 🧠 Admin-Interface für Wissensdatenbank")
    gr.Markdown("Verwalte deine RAG-Wissensdatenbank")

    with gr.Tab("Alle Einträge anzeigen"):
        output_list = gr.Textbox(label="Alle Einträge", lines=20)
        btn_list = gr.Button("Liste laden")
        btn_list.click(fn=list_all_entries, outputs=output_list)

    with gr.Tab("Neuen Eintrag hinzufügen"):
        input_new = gr.Textbox(label="Neuer Eintrag", lines=5)
        output_add = gr.Textbox(label="Ergebnis")
        btn_add = gr.Button("Hinzufügen")
        btn_add.click(fn=add_entry, inputs=input_new, outputs=output_add)

    with gr.Tab("Eintrag löschen"):
        input_delete = gr.Textbox(label="ID des zu löschenden Eintrags")
        output_delete = gr.Textbox(label="Ergebnis")
        btn_delete = gr.Button("Löschen")
        btn_delete.click(fn=delete_entry, inputs=input_delete, outputs=output_delete)

    with gr.Tab("Suchen"):
        input_search = gr.Textbox(label="Suchbegriff")
        output_search = gr.Textbox(label="Suchergebnisse", lines=15)
        btn_search = gr.Button("Suchen")
        btn_search.click(fn=search_entries, inputs=input_search, outputs=output_search)

# 4. Starten
if __name__ == "__main__":
    demo.launch(server_port=7861)

