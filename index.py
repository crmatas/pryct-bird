
import os
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GENERAL SETTINGS
DOCUMENTS = "./documentos_pajaros"
CHROMA_FOLDER = "./chroma_pajaros_db"
COLLECTION = "coleccion_pajaros"

print("🦅 STARTING DOCUMENT INDEXING")

# 1. Check folder
os.makedirs(DOCUMENTS, exist_ok=True)

# 2. Load documents
documents = []

for file_path in Path(DOCUMENTS).rglob("*"):

    if not file_path.is_file():
        continue

    extension = file_path.suffix.lower()

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif extension in [".txt", ".md"]:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        else:
            continue

        docs = loader.load()
        documents.extend(docs)

        print(f"✅ Loaded: {file_path.name}")

    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")

if not documents:
    print("⚠️ No documents available to index.")
    print("Add PDF, TXT, or MD files to documentos_pajaros.")
    raise SystemExit

# 3. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"📚 Chunks created: {len(chunks)}")

# 4. Create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 5. Create database
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION,
    persist_directory=CHROMA_FOLDER
)

print("\n🦅 KNOWLEDGE BASE CREATED")
print(f"Original documents: {len(documents)}")
print(f"Indexed chunks: {len(chunks)}")
print(f"Location: {CHROMA_FOLDER}")
