import os
from typing import TypedDict

import requests
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph


MODEL = "llama3.2"
EMBEDDING = "nomic-embed-text"
CHROMA_FOLDER = "./chroma_pajaros_db"
COLLECTION = "coleccion_pajaros"
SEARCH_API_URL = os.getenv(
    "SEARCH_API_URL", "http://localhost:8000").rstrip("/")
USER_ID = os.getenv("SEARCH_USER_ID", "local-user")

print("🦅 LOADING ORNITHOLOGY ASSISTANT...")

embeddings = OllamaEmbeddings(model=EMBEDDING)
db = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings,
)
llm = ChatOllama(model=MODEL, temperature=0.2)

prompt = ChatPromptTemplate.from_template(
    """
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
"""
)


@tool
def save_user_search(query: str, user_id: str) -> str:
    """Save a user's question through the search history API."""
    response = requests.post(
        f"{SEARCH_API_URL}/searches",
        json={"user_id": user_id, "query": query},
        timeout=10,
    )
    response.raise_for_status()
    saved_search = response.json()
    return f"Saved search {saved_search['id']} for user {user_id}."


@tool
def list_user_searches(user_id: str) -> str:
    """List the latest searches saved for a user in the local history API."""
    response = requests.get(
        f"{SEARCH_API_URL}/searches",
        params={"user_id": user_id, "limit": 100},
        timeout=10,
    )
    response.raise_for_status()
    searches = response.json()
    if not searches:
        return f"No searches found for user {user_id}."

    entries = [
        f"- [{search['id']}] {search['created_at']}: {search['query']}"
        for search in searches
    ]
    return f"Search history for {user_id}:\n" + "\n".join(entries)


class AgentState(TypedDict, total=False):
    question: str
    user_id: str
    welcome: str
    answer: str


def welcome_node(_: AgentState) -> dict[str, str]:
    return {
        "welcome": (
            "Welcome to SKYTERBIRD, your bird specialist. "
            "Ask me about identification, habitats, behavior, or conservation."
        )
    }


def route_after_welcome(state: AgentState) -> str:
    question = state.get("question", "").strip()
    if question.casefold() in {"historial", "/historial", "history", "/history"}:
        return "list_searches"
    return "query_knowledge" if question else END


def query_knowledge_node(state: AgentState) -> dict[str, str]:
    question = state["question"].strip()
    user_id = state.get("user_id", USER_ID)
    save_user_search.invoke({"query": question, "user_id": user_id})

    documents = db.similarity_search(question, k=4)
    if not documents:
        return {"answer": "I could not find information in my knowledge base."}

    context = "\n\n".join(document.page_content for document in documents)
    response = (prompt | llm).invoke(
        {"context": context, "question": question}
    )
    return {"answer": response.content}


def list_searches_node(state: AgentState) -> dict[str, str]:
    user_id = state.get("user_id", USER_ID)
    return {"answer": list_user_searches.invoke({"user_id": user_id})}


workflow = StateGraph(AgentState)
workflow.add_node("welcome", welcome_node)
workflow.add_node("query_knowledge", query_knowledge_node)
workflow.add_node("list_searches", list_searches_node)
workflow.add_edge(START, "welcome")
workflow.add_conditional_edges(
    "welcome",
    route_after_welcome,
    {
        "query_knowledge": "query_knowledge",
        "list_searches": "list_searches",
        END: END,
    },
)
workflow.add_edge("query_knowledge", END)
workflow.add_edge("list_searches", END)
graph = workflow.compile()


def main() -> None:
    initial_state = graph.invoke({"question": "", "user_id": USER_ID})
    print("\n====================================")
    print("       🦅 SKYTERBIRD - AI ASSISTANT")
    print("       Bird specialist")
    print("====================================")
    print(initial_state["welcome"])
    print("Type 'exit' to finish.\n")

    while True:
        question = input("YOU: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("SKYTERBIRD: See you soon! 🦅")
            break
        if not question:
            continue

        result = graph.invoke({"question": question, "user_id": USER_ID})
        print("\nSKYTERBIRD:", result["answer"])
        print("\n------------------------------------\n")


if __name__ == "__main__":
    main()
