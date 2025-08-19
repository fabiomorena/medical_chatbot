# 🏥 Medical Chatbot with RAG

A medical chatbot based on **TinyLlama** and **Llama 3.1**, combined with **Retrieval-Augmented Generation (RAG)** for precise, context-aware answers.

---

## 🌟 Features
- 💬 **Web-based chatbot** (Gradio)
- 🧠 **RAG-based**: Local knowledge + cloud generation
- 📚 **Learning mode**: Stores new insights from conversations
- 🛠️ **Admin interface**: Manage the knowledge base
- 🔒 **Privacy**: Local retrieval, no external storage

---

## 🧰 Requirements
- Python **3.10+**
- At least **8 GB RAM**
- (Optional) GPU for better performance
- Hugging Face account + token: [Settings → Tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Installation & Quickstart

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd medical_chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Hugging Face token
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx   # Linux/Mac
setx HF_TOKEN hf_xxxxxxxxxxxxxxxx     # Windows
```

### 5. Start application
- 🌐 **Web interface**  
  ```bash
  python web_rag_llama3.py
  ```

- 🛠️ **Admin interface**  
  ```bash
  python admin_interface.py
  ```

---

## 🧪 Usage

### Chatbot commands
- Normal question: `What is the flu?`
- Store knowledge: `learn this` after a response

### Admin interface
- List all entries  
- Add new entries  
- Delete entries  
- Search the knowledge base  

---

## 📁 Project structure
```
medical_chatbot/
├── README.md               # This file
├── requirements.txt        # Dependencies
├── medical_data.json       # Example data
├── train_medical_bot.py    # Optional training
├── chatbot_cli.py          # Console chatbot
├── chatbot_web.py          # Simple web interface
├── hybrid_rag.py           # RAG with local retrieval
├── web_rag_llama3.py       # Main web interface (Llama 3.1)
├── admin_interface.py      # Admin tool
└── chroma_db/              # Local vector database
```

---

## 🧠 Model information
- **Embedding**: `all-MiniLM-L6-v2` (local)  
- **Generative**: `meta-llama/Llama-3.1-8B-Instruct` (cloud)  
- **Optional pretraining**: `SpookyFab/tinyllama-pretrained-custom`  

👉 To gain access: open [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and accept the usage terms.

---

## 📚 Data sources
- Public medical datasets  
- Wikipedia (medical chapters)  
- Own experiences via learning mode  

---

## 🛡️ Privacy
- Retrieval runs **locally**  
- Questions are not permanently stored  
- Answers are not externally archived  
- Learning mode only saves explicitly approved content  

---

## 🧾 Next steps
- Extend with more medical data  
- Build expert bots for other domains (law, technology, etc.)  
- Deploy to the cloud (AWS, Google Cloud)  
- Provide API interface  

---

## 🤝 Contributing
1. Fork the repo  
2. Create a branch: `feature/NewFeature`  
3. Commit your changes  
4. Open a pull request  

---

## 📄 License
MIT License – see [LICENSE](LICENSE)  

---

## 📞 Contact
Questions or issues? → [Create an issue](../../issues)
