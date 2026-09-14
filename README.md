# 🌊 Argo Ocean Data Agent

**[Live Demo](https://llnqqsgtx2ecnarzxkxitw.streamlit.app/)**

An AI-powered, RAG-based Text-to-SQL agent that allows users to ask plain-English questions about real-world oceanographic data (temperature, salinity, depth) and generates analytical summaries with charts.

<img width="1916" height="910" alt="Screenshot 2026-09-14 150025" src="https://github.com/user-attachments/assets/4448ad74-c43e-4b79-9031-ce0c0e74c4fa" />


---

## 🏗️ Architecture & Cloud Deployment

The project is structured around a decoupled, serverless-ready architecture designed for free-tier cloud deployment:
- **FastAPI Backend (Render)**: Acts as a REST API managing the DuckDB analytical queries, NumPy vector similarity search, and LLM routing/summarization logic. The removal of heavy ML libraries ensures the application runs easily within Render's 512MB RAM limit.
- **Streamlit Frontend (Streamlit Community Cloud)**: A lightweight UI that handles user inputs and dynamically renders the JSON responses, charts, and debug metrics returned by the live deployed backend.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI, Python |
| **Database & Search** | DuckDB, NumPy (for local vector math) |
| **AI / LLM** | LLaMA-3.3 (via Groq), Hugging Face Inference API (for serverless embeddings) |

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
pip install -r requirements-ui.txt
```

### 2. Configure Local Environment Variables

Create the required configuration files for both backend and frontend:

**Backend (`.env`)**
Create a `.env` file in the root directory:
```ini
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_hf_token_here
```

**Frontend (`.streamlit/secrets.toml`)**
Create a `.streamlit/secrets.toml` file in the root directory:
```toml
API_URL = "http://127.0.0.1:8000"
```

### 3. Generate the Data

You must build the local database before starting the server. Run the fetch script to download live data from the Argo program:

```bash
python fetch_real_data.py
```

### 4. Start the Servers

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
