# 🏥 Medizinischer Chatbot mit RAG

Ein medizinischer Chatbot basierend auf **TinyLlama** und **Llama 3.1**, kombiniert mit **Retrieval-Augmented Generation (RAG)** für präzise, kontextbasierte Antworten.

---

## 🌟 Features
- 💬 **Chatbot im Browser** (Gradio)
- 🧠 **RAG-basiert**: Lokales Wissen + Cloud-Generierung
- 📚 **Lernmodus**: Speichert neue Erkenntnisse aus Gesprächen
- 🛠️ **Admin-Interface**: Verwalte die Wissensdatenbank
- 🔒 **Datenschutz**: Lokales Retrieval, keine externen Speicherungen

---

## 🧰 Voraussetzungen
- Python **3.10+**
- Mindestens **8 GB RAM**
- (Optional) GPU für bessere Performance
- Hugging Face Account + Token: [Einstellungen → Tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Installation & Schnellstart

### 1. Repository klonen
```bash
git clone <dein-repo-link>
cd medical_chatbot
```

### 2. Virtuelle Umgebung erstellen
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 4. Hugging Face Token setzen
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx   # Linux/Mac
setx HF_TOKEN hf_xxxxxxxxxxxxxxxx     # Windows
```

### 5. Anwendung starten
- 🌐 **Webinterface**  
  ```bash
  python web_rag_llama3.py
  ```

- 🛠️ **Admin-Interface**  
  ```bash
  python admin_interface.py
  ```

---

## 🧪 Nutzung

### Chatbot-Befehle
- Normale Frage: `Was ist eine Grippe?`
- Wissen speichern: `lerne das` nach einer Antwort

### Admin-Interface
- Alle Einträge anzeigen  
- Neue Einträge hinzufügen  
- Einträge löschen  
- In Wissensdatenbank suchen  

---

## 📁 Projektstruktur
```
medical_chatbot/
├── README.md               # Diese Datei
├── requirements.txt        # Abhängigkeiten
├── medical_data.json       # Beispieldaten
├── train_medical_bot.py    # Optionales Training
├── chatbot_cli.py          # Konsolen-Chatbot
├── chatbot_web.py          # Einfaches Webinterface
├── hybrid_rag.py           # RAG mit lokalem Retrieval
├── web_rag_llama3.py       # Haupt-Webinterface (Llama 3.1)
├── admin_interface.py      # Admin-Tool
└── chroma_db/              # Lokale Vektordatenbank
```

---

## 🧠 Modellinformationen
- **Embedding**: `all-MiniLM-L6-v2` (lokal)  
- **Generativ**: `meta-llama/Llama-3.1-8B-Instruct` (Cloud)  
- **Optionales Pretraining**: `SpookyFab/tinyllama-pretrained-custom`  

👉 Für Zugriff: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) öffnen und Nutzungsbedingungen akzeptieren.

---

## 📚 Datenquellen
- Öffentliche medizinische Datensätze  
- Wikipedia (Medizin-Kapitel)  
- Eigene Erfahrungen über Lernmodus  

---

## 🛡️ Datenschutz
- Retrieval läuft **lokal**  
- Fragen werden nicht dauerhaft gespeichert  
- Antworten werden nicht extern archiviert  
- Lernmodus speichert nur explizit genehmigte Inhalte  

---

## 🧾 Nächste Schritte
- Erweiterung mit mehr medizinischen Daten  
- Experten-Bots für weitere Fachgebiete (Recht, Technik, etc.)  
- Deployment in der Cloud (AWS, Google Cloud)  
- API-Schnittstelle bereitstellen  

---

## 🤝 Mitwirken
1. Repo forken  
2. Branch erstellen: `feature/NeueFunktion`  
3. Änderungen committen  
4. Pull Request öffnen  

---

## 📄 Lizenz
MIT License – siehe [LICENSE](LICENSE)  

---

## 📞 Kontakt
Fragen oder Probleme? → [Issue erstellen](../../issues)
