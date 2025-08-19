# 📌 Medical Chatbot Roadmap

Dies ist die Entwicklungsroadmap für das Projekt. Die Issues sind priorisiert und in thematische Bereiche gegliedert.

---

## 🚀 Architektur & Modularität
**Issue:** Code-Refactoring: Trennung von Core, Interfaces, Persistence  
- Core-Module (RAG, Embeddings, Retrieval) entkoppeln  
- Interfaces (CLI, Gradio, Admin) in eigene Dateien verschieben  
- Persistence (Chroma, Dateisystem) abstrahieren  
**Priorität:** Hoch  
**Labels:** enhancement, refactor  

---

## 🔒 Sicherheit & Datenschutz
**Issue:** Eingabevalidierung & Datenschutzprüfung  
- User-Eingaben sanitizen (gegen Prompt Injection)  
- Logging überprüfen, sensible Daten entfernen  
- Disclaimer “Kein medizinischer Rat” in jedem Interface sichtbar machen  
**Priorität:** Hoch  
**Labels:** security, bug  

---

## 📊 RAG-Optimierung
**Issue:** Verbesserung Embeddings und Retrieval  
- Embedding-Modell austauschbar machen (z. B. bge-small, sentence-transformers)  
- Adaptive Chunking (Overlap 20–30 Tokens)  
- Retrieval-Benchmarks mit Precision/Recall auf Testset  
**Priorität:** Mittel  
**Labels:** enhancement, ml  

---

## ⚡ Skalierbarkeit
**Issue:** Vorbereitung auf produktives Deployment  
- FastAPI + Uvicorn als Backend  
- Dockerfile hinzufügen  
- Option für persistente DB (Postgres + pgvector)  
**Priorität:** Mittel  
**Labels:** deployment, enhancement  

---

## ✅ Tests & CI
**Issue:** Testsuite und GitHub Actions Pipeline  
- pytest-Setup für Unit- und Integrationstests  
- Tests für RAG-Queries mit Fixtures  
- GitHub Actions: Linting (flake8/black), Tests, Build  
**Priorität:** Hoch  
**Labels:** testing, ci/cd  

---

## 🎨 UX & UI
**Issue:** Verbesserte Gradio-UI mit Quellenangaben  
- Gesprächsverlauf speichern und anzeigen  
- Quellen neben Antworten ausgeben  
- Admin-Interface absichern (Auth)  
**Priorität:** Mittel  
**Labels:** ui/ux, enhancement  

---

## 📖 Dokumentation
**Issue:** Verbesserte README und Contribution Guide  
- Setup-Anleitung mit Beispieldaten  
- requirements.txt aufteilen (dev/prod)  
- Contribution Guide + Code Style Guide  
**Priorität:** Niedrig  
**Labels:** documentation  
