import chromadb
from chromadb.utils import embedding_functions
from google import genai

MODEL = "gemini-3.8-flash"   # error vandha idha maathunga

# Step 2 database-a open pannuradhu
client = chromadb.PersistentClient(path="db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection("notes", embedding_function=ef)

question = input("Question: ")

# Related chunks edukkuradhu (5 chunks)
results = collection.query(query_texts=[question], n_results=5)
context = "\n---\n".join(results["documents"][0])

# Gemini-kitta kekkuradhu
llm = genai.Client()
response = llm.models.generate_content(
    model=MODEL,
    contents=f"Answer using ONLY these notes. If the answer is not in the notes, say so.\n\nNotes:\n{context}\n\nQuestion: {question}",
)
print("\nAnswer:", response.text)