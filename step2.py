from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# PDF padikkuradhu
reader = PdfReader("notes.pdf")
text = ""
for page in reader.pages:
    text += (page.extract_text() or "") + "\n"

# Chunks pirikkuradhu (perusa chunks, nalla overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = [c for c in splitter.split_text(text) if len(c.strip()) > 30]

# ChromaDB setup
client = chromadb.PersistentClient(path="db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Pazhaya chunks-a azhikkuradhu
try:
    client.delete_collection("notes")
except Exception:
    pass

collection = client.get_or_create_collection("notes", embedding_function=ef)

# Puthu chunks-a store pannuradhu
collection.upsert(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

print("Total chunks stored:", len(chunks))

# Test search
results = collection.query(query_texts=["What is normalization?"], n_results=3)
for doc in results["documents"][0]:
    print(doc)
    print("-----")