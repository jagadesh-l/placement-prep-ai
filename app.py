import time
import glob
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

MODEL = "gemini-3.8-flash"
NL = chr(10)

st.set_page_config(page_title="PlacementPrep AI", layout="centered")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.hero {
    background: linear-gradient(135deg, #3B82F6, #6366F1);
    padding: 28px; border-radius: 14px; color: white; margin-bottom: 20px;
}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 6px 0 0 0; opacity: 0.9;}
</style>
<div class="hero">
    <h1>PlacementPrep AI</h1>
    <p>RAG-based interview preparation assistant. Answers come only from your notes.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    col = client.get_or_create_collection("notes", embedding_function=ef)

    # Database empty-a irundha, notes folder-la irundhu thaanaave build pannum
    if col.count() == 0:
        text = ""
        for f in glob.glob("notes/*.pdf"):
            for page in PdfReader(f).pages:
                text += (page.extract_text() or "") + NL
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = [c for c in splitter.split_text(text) if len(c.strip()) > 30]
        if chunks:
            col.upsert(
                documents=chunks,
                ids=["chunk_" + str(i) for i in range(len(chunks))],
            )
    return col


def ask_gemini(prompt, retries=3):
    # Gemini busy-a (503) irundha thaanaave 3 thadava retry pannum
    llm = genai.Client()
    for attempt in range(retries):
        try:
            return llm.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            if "503" in str(e) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise


collection = load_collection()

if collection.count() == 0:
    st.warning("No notes found. Add a PDF to the notes folder and restart the app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def set_pending(q):
    st.session_state.pending = q


# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Ask questions from your study notes and get answers with sources.")
    st.subheader("How it works")
    st.markdown("""
1. Notes are split into chunks
2. Chunks are stored as embeddings
3. Relevant chunks are retrieved
4. Gemini answers using only those chunks
""")
    st.subheader("Tech stack")
    st.markdown("Python, ChromaDB, Sentence-Transformers, Gemini, Streamlit")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Example questions (chat empty-a irukkum bodhu mattum)
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    examples = [
        "What is normalization?",
        "What is a primary key?",
        "Explain ER modeling",
    ]
    cols = st.columns(len(examples))
    for col, q in zip(cols, examples):
        col.button(q, on_click=set_pending, args=(q,), use_container_width=True)

# Pazhaya chat-a kaatturadhu
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.caption(s)
                    st.divider()

# Puthu question
typed = st.chat_input("Ask a question from your notes...")
question = typed or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                # CHANGED (Edit 1): 5 -> 8 chunks
                results = collection.query(query_texts=[question], n_results=8)
                chunks = results["documents"][0]
                context = (NL + "---" + NL).join(chunks)

                # CHANGED (Edit 2): better tutor-style prompt
                prompt = (
                    "You are an interview prep tutor. Answer using the notes below. "
                    "Explain in simple words and give a short example if the notes "
                    "support it. If the notes do not cover the question, say so."
                    + NL + NL + "Notes:" + NL + context
                    + NL + NL + "Question: " + question
                )
                answer = ask_gemini(prompt)
            except Exception as e:
                # CHANGED: real error terminal-la theriyum, user-ku correct message
                print("GEMINI ERROR:", e)
                if "503" in str(e) or "429" in str(e):
                    answer = "The AI service is busy right now. Please try again in a minute."
                else:
                    answer = "Something went wrong. Please try again."
                chunks = []

        st.markdown(answer)
        if chunks:
            with st.expander("Sources used"):
                for s in chunks:
                    st.caption(s)
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks}
    )