import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="AI Research Agent", layout="wide")

st.title("🔎 AI Research Agent (RAG + LLM)")

# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("Settings")
    max_results = st.slider("Search Results", 3, 10, 5)
    st.info("This agent searches web + uses RAG + LLM for answers.")

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# -----------------------------
# Helper: Web Search
# -----------------------------
def search_web(query, max_results=5):
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["href"])

    return results

# -----------------------------
# Helper: Extract page text
# -----------------------------
def extract_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(
            url,
            headers=headers,
            timeout=5
        )

        soup = BeautifulSoup(
            res.text,
            "html.parser"
        )

        for tag in soup(
            ["script", "style", "noscript"]
        ):
            tag.extract()

        text = soup.get_text(separator=" ")

        return text

    except:
        return ""

# -----------------------------
# Step 1: Query input
# -----------------------------
query = st.text_input(
    "Enter your research topic"
)

# -----------------------------
# Main pipeline
# -----------------------------
if query:

    with st.spinner(
        "🔍 Searching the web..."
    ):
        urls = search_web(
            query,
            max_results
        )

    # Show sources early
    st.subheader("📌 Sources Found")

    for u in urls:
        st.write(u)

    # -----------------------------
    # Step 2: Load content
    # -----------------------------
    docs_text = ""

    with st.spinner(
        "📄 Extracting content..."
    ):
        for url in urls:
            text = extract_text(url)

    st.write("URL:", url)
    st.write("Characters extracted:", len(text))

    docs_text += text + "\n"
            

    # -----------------------------
    # Step 3: Chunking
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(
        docs_text
    )

    if not chunks:
        st.error(
            "Could not extract content from search results."
        )
        st.stop()

    # -----------------------------
    # Step 4: Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    # -----------------------------
    # Step 5: Retrieve relevant chunks
    # -----------------------------
    docs = vector_store.similarity_search(
        query,
        k=5
    )

    context = "\n\n".join(
        [
            d.page_content
            for d in docs
        ]
    )

    # -----------------------------
    # Professional Prompt
    # -----------------------------
    prompt = f"""
You are an expert AI research assistant like Perplexity AI.

Your job:
- Understand the user's query even if it is vague or incomplete
- Reformulate it mentally into a clear research question
- Use ONLY the provided context
- If context is weak, say "limited reliable information available"
- Do NOT hallucinate

Write a PROFESSIONAL research report with:

1. Title (clean and academic)
2. Overview (simple explanation)
3. Key Concepts (bullet points)
4. Detailed Explanation (well structured)
5. Real-world Applications
6. Advantages
7. Limitations / Challenges
8. Conclusion
9. Sources (from context URLs)

RULES:
- Be factual and accurate
- Use simple + professional English
- Do NOT mention “context provided”
- If question is unclear, interpret intelligently
- Make answer suitable for college assignment / research report

CONTEXT:
{context}

USER QUESTION:
{query}

FINAL ANSWER:
"""

    # -----------------------------
    # Step 6: Generate response
    # -----------------------------
    with st.spinner(
        "🧠 Thinking like a research analyst..."
    ):
        response = llm.invoke(
            prompt
        )

    st.subheader(
        "📊 Research Report"
    )

    st.write(
        response.content
    )