import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from google import genai

MODEL = "gemini-3.8-flash"

st.set_page_config(page_title="PlacementPrep AI", page_icon="🎯", layout="centered")

# Menu/footer maraikka, header azhaga kaatta
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.hero {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    padding: 28px; border-radius: 14px; color: white; margin-bottom: 20px;
}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 6px 0 0 0; opacity: 0.9;}
</style>
<div class="hero">
    <h1>🎯 PlacementPrep AI</h1>
    <p>RAG-based interview preparation assistant. Answers come only from your notes.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection("notes", embedding_function=ef)


collection = load_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []


def set_pending(q):
    st.session_state.pending = q


# Sidebar
with st.sidebar:
    st.header("About")
    st.write("Ask questions from your study notes and get answers with sources.")
    st.subheader("How it works")
    st.markdown(
        "1. Notes are split into chunks\n"
        "2. Chunks are stored as embeddings\n"
        "3. Relevant chunks are retrieved\n"
        "4. Gemini answers using only those chunks"
    )
    st.subheader("Tech stack")
    st.markdown("Python · ChromaDB · Sentence-Transformers · Gemini · Streamlit")
    if st.button("🗑️ Clear chat", use_container_width=True):
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
            with st.expander("📄 Sources used"):
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
                results = collection.query(query_texts=[question], n_results=5)
                chunks = results["documents"][0]
                context = "\n---\n".join(chunks)

                llm = genai.Client()
                response = llm.models.generate_content(
                    model=MODEL,
                    contents=(
                        "Answer using ONLY these notes. If the answer is not in the "
                        f"notes, say so.\n\nNotes:\n{context}\n\nQuestion: {question}"
                    ),
                )
                answer = response.text
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
                chunks = []

        st.markdown(answer)
        if chunks:
            with st.expander("📄 Sources used"):
                for s in chunks:
                    st.caption(s)
                    st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks}
    )