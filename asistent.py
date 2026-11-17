
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# GENERAL SETTINGS
MODEL = "llama3.2"
EMBEDDING = "nomic-embed-text"

CHROMA_FOLDER = "./chroma_pajaros_db"
COLLECTION = "coleccion_pajaros"

print("🦅 LOADING ORNITHOLOGY ASSISTANT...")

# 1. Load embeddings
embeddings = OllamaEmbeddings(
    model=EMBEDDING
)

# 2. Connect to ChromaDB
db = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings
)

# 3. Configure AI model
llm = ChatOllama(
    model=MODEL,
    temperature=0.2
)

# 4. Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are SKYTERBIRD, an assistant specialized in ornithology.

Your specialties are:
- Bird identification.
- Taxonomy and classification.
- Feeding.
- Habitats.
- Reproduction.
- Migration.
- Behavior.
- Conservation of species.

INSTRUCTIONS:

1. Always answer in English.
2. Use primarily the information from the context.
3. Do not invent scientific data.
4. If you do not find enough information, say so clearly.
5. Explain scientific concepts simply.
6. When possible, include the scientific name.
7. Distinguish documented data from hypotheses.

KNOWLEDGE BASE CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
""")


# 5. Query function
def query_knowledge(question):

    documents = db.similarity_search(
        question,
        k=4
    )

    if not documents:
        return "I could not find information in my knowledge base."

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


# 6. Conversation console
print("\n====================================")
print("       🦅 SKYTERBIRD - AI ASSISTANT")
print("       Bird specialist")
print("====================================")
print("Type 'exit' to finish.\n")

while True:

    question = input("YOU: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("SKYTERBIRD: See you soon! 🦅")
        break

    if not question:
        continue

    try:
        response = query_knowledge(question)

        print("\nSKYTERBIRDh:", response)
        print("\n------------------------------------\n")

    except Exception as e:
        print("Error:", e)
