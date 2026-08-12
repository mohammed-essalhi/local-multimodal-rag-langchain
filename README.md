# Local Multimodal RAG Assistant (LangChain & Streamlit)

##  Overview
This repository contains a fully local, privacy-preserving Multimodal Retrieval-Augmented Generation (RAG) application. Built with **LangChain** and **Streamlit**, the system processes complex PDF documents by isolating raw text and extracting visual figures. It leverages **LLaVA** (via Ollama) to generate semantic summaries of images and charts, fusing them into a unified **ChromaDB** vector space for highly accurate, context-aware querying.

![App Screenshot](screenshots/Capture.PNG) 

##  Architecture & Pipeline
1. **Text Indexing (`01_text_indexing.ipynb`):** Parses the source PDF, applies recursive character splitting, and embeds the raw text into ChromaDB using the `nomic-embed-text` model.
2. **Multimodal Vision Ingestion (`02_multimodal_vision_indexing.ipynb`):** Utilizes `pdf2image` and Poppler to extract document figures. Passes images to the **LLaVA** Vision-Language Model to generate descriptive summaries, which are then embedded and fused into the vector store.
3. **Conversational Interface (`app.py`):** A Streamlit-based web application featuring custom conversational memory (`session_state`) and real-time source citation, allowing users to interact naturally with the ingested multimodal data.

## ⚙️ Tech Stack
* **Orchestration:** LangChain
* **Frontend:** Streamlit
* **Vector Database:** ChromaDB
* **Local LLM / VLM:** Ollama (LLaVA for vision, Nomic for embeddings)
* **Document Processing:** PyPDF, pdf2image, Poppler

## 🚀 How to Run Locally
1. Ensure **Ollama** is installed and running the required models:
```bash
   ollama pull llava
   ollama pull nomic-embed-text
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

##  Author
**Mohammed Essalhi**
* [LinkedIn](https://linkedin.com/in/mohammed-essalhi-23794b24b)


3. Run the indexing notebooks sequentially (01_text_indexing.ipynb then 02_multimodal_vision_indexing.ipynb) to generate the local ollama_mrag_db.

4. Launch the Streamlit application:
```bash
   streamlit run app.py
```
