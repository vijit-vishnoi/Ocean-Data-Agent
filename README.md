# 🌊 Argo Ocean Data Agent

An AI-powered, RAG-based Text-to-SQL agent that allows users to ask plain-English questions about real-world oceanographic data (temperature, salinity, depth) and generates analytical summaries with charts.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI, Python |
| **Database & Search** | DuckDB, FAISS |
| **AI / LLM** | LLaMA-3.3 (via Groq), SentenceTransformers |

---

## 🚀 How to Run Locally

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/yourusername/Ocean-Bot.git
cd Ocean-Bot
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

---

### 2. Generate the Data

You must build the local database before starting the server. Run the fetch script to download live data from the Argo program:

```bash
python fetch_real_data.py
```

---

### 3. Start the Servers

Open **two terminals** and run the following:

**Terminal 1 — Backend:**

```bash
uvicorn main:app --reload
```

**Terminal 2 — Frontend:**

```bash
streamlit run ui.py
```

Then open your browser and navigate to the Streamlit URL shown in the terminal (usually `http://localhost:8501`).
