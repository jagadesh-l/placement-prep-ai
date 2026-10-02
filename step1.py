from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF-a padikkuradhu
reader = PdfReader("notes.pdf")
text = ""
for page in reader.pages:
    text += (page.extract_text() or "") + "\n"

# Chinna chinna chunks-a pirikkuradhu
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
print(chunks[0])