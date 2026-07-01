# AI Research Agent (RAG + LLM)

## Author
Aastha Deep

## Project Overview

AI Research Agent is a Retrieval-Augmented Generation (RAG) based application that performs web research and generates structured research reports using Large Language Models (LLMs).

The system searches the web for relevant information, extracts content from multiple sources, converts the content into embeddings, stores them in a FAISS vector database, retrieves the most relevant information, and generates a professional research report.

---

## Features

- Web Search using DuckDuckGo
- Web Content Extraction
- Text Chunking
- Vector Embeddings
- FAISS Vector Database
- Similarity Search
- LLM-based Research Report Generation
- Structured Research Output
- Streamlit User Interface

---

## Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM (Llama 3.3 70B)
- BeautifulSoup
- Requests
- DuckDuckGo Search

---

## Project Workflow

User Query
↓
Web Search
↓
Content Extraction
↓
Text Chunking
↓
Embeddings Generation
↓
FAISS Vector Store
↓
Similarity Retrieval
↓
LLM Analysis
↓
Research Report Generation

---
```

AI-Research-Agent/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Required Python packages
├── .env                    # Environment variables (excluded from Git)
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd AI-Research-Agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Create Environment File

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### 6. Run Application

```bash
streamlit run app.py
```

---

## How It Works

1. User enters a research topic.
2. DuckDuckGo retrieves relevant web sources.
3. Website content is extracted using BeautifulSoup.
4. Content is split into smaller chunks.
5. Embeddings are generated using HuggingFace.
6. FAISS stores and retrieves relevant chunks.
7. Groq Llama 3.3 analyzes the retrieved information.
8. A structured research report is generated.

---

## Future Improvements

- Research Report Download
- Source Ranking
- Query Rewriting
- Research History
- Advanced Retrieval Methods
- Multi-Agent Research Pipeline

---

## License

This project is developed for educational and learning purposes.
