# 🏥 Medizinischer Chatbot mit RAG

Ein fortschrittlicher, medizinischer Chatbot basierend auf **TinyLlama** und **Llama 3.1**, kombiniert mit **Retrieval-Augmented Generation (RAG)** für präzise, kontextbasierte Antworten.

## 🌟 Funktionen

- 💬 **Chatbot im Browser** (Gradio)
- 🧠 **RAG-basiert** (lokales Wissen + Cloud-Generierung)
- 📚 **Lernmodus**: Speichert neue Erkenntnisse aus Gesprächen
- 🛠️ **Admin-Interface**: Verwalte die Wissensdatenbank
- 🔒 **Datenschutz**: Lokales Retrieval, keine externen Speicherungen

## 🧰 Voraussetzungen

- Python 3.10+
- Mindestens 8 GB RAM
- (Optional) GPU für bessere Performance

## 🚀 Schnellstart

### 1. Repository klonen

```bash
git clone <dein-repo-link>
cd medical_chatbot

python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

pip install -r requirements.txt

Du benötigst einen Hugging Face Token für Llama 3.1:

Gehe zu: https://huggingface.co/settings/tokens
Erstelle einen Read-Token
Setze ihn in der Konsole:

export HF_TOKEN=hf_xxxxxxxxxxxxxxxx


5. Anwendung starten
🌐 Webinterface starten

python web_rag_llama3.py

🛠️ Admin-Interface starten

python admin_interface.py

🧪 Nutzung
Chatbot-Befehle
Normale Frage stellen: Was ist eine Grippe?
Wissen speichern: Tippe lerne das nach einer Antwort
Admin-Interface
Alle Einträge anzeigen
Neue Einträge hinzufügen
Einträge löschen
In Wissensdatenbank suchen

📁 Projektstruktur

medical_chatbot/
├── README.md               # Diese Datei
├── requirements.txt        # Python-Abhängigkeiten
├── medical_data.json       # Trainingsdaten
├── train_medical_bot.py    # Lokales Training (optional)
├── chatbot_cli.py          # Konsolen-Chatbot
├── chatbot_web.py          # Einfaches Webinterface
├── hybrid_rag.py           # RAG mit lokalem Retrieval
├── web_rag_llama3.py       # Haupt-Webinterface mit Llama 3.1
├── admin_interface.py      # Admin-Tool zur Wissensverwaltung
└── chroma_db/              # Lokale Vektordatenbank (wird automatisch erstellt)

🧠 Modellinformationen
Verwendete Modelle
Embedding: all-MiniLM-L6-v2 (lokal)
Generatives Modell: meta-llama/Llama-3.1-8B-Instruct (Cloud)
Basis-Modell: SpookyFab/tinyllama-pretrained-custom (lokal trainierbar)
Modellzugriff freischalten
Gehe zu: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
Klicke auf „Access repository“
Akzeptiere die Nutzungsbedingungen

📚 Datenquellen
Öffentliche medizinische Datensätze
Wikipedia: Medizin-Kapitel
Eigene Erfahrungen aus Chats (mit Lernmodus)

🛡️ Datenschutz
Alle Retrieval-Prozesse laufen lokal
Fragen werden nicht dauerhaft gespeichert
Antworten werden nicht extern archiviert
Lernmodus speichert nur explizit genehmigte Inhalte

🧾 Nächste Schritte
Wenn du das Projekt erweitern willst:

Mehr medizinische Daten hinzufügen
Weitere Experten-Bots bauen (Recht, Technik, etc.)
Deployment in der Cloud (AWS, Google Cloud)
API-Schnittstelle bereitstellen

🤝 Mitwirken
Repository forken
Branch erstellen (feature/NeueFunktion)
Änderungen committen
Pull Request erstellen

📄 Lizenz
MIT License – siehe LICENSE Datei

📞 Kontakt
Bei Fragen oder Problemen: Erstelle ein Issue im Repository.



