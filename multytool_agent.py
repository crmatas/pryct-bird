
import os
import requests

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)

from typing import Annotated, Sequence, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    # 'add_messages' is a reducer that appends
    # new messages to the conversation history.
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================================
# 1. GENERAL SETTINGS
# ============================================================

MODEL = "llama3.2"
EMBEDDINGS_MODEL = "nomic-embed-text"

PERSIST_DIR = "./chroma_pajaros_db"
COLLECTION = "coleccion_pajaros"

OBSERVATION_LOG = "bitacora_avistamientos.txt"

# ============================================================
# 2. LOAD THE RAG KNOWLEDGE BASE
# ============================================================

print("Loading bird encyclopedia...")

if not os.path.isdir(PERSIST_DIR):
    raise FileNotFoundError(
        "The Chroma database does not exist. "
        "Run index.py first."
    )

embeddings = OllamaEmbeddings(
    model=EMBEDDINGS_MODEL
)

db_birds = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)

print("Knowledge base loaded successfully.")

# ============================================================
# 3. TOOL 1: SEARCH THE RAG ENCYCLOPEDIA
# ============================================================


@tool
def search_bird_encyclopedia(query: str) -> str:
    """
    Search for information in the bird encyclopedia.

    Use this tool to query species, characteristics,
    feeding, reproduction, habitats, migration, and behavior.
    """

    try:
        docs = db_birds.similarity_search(
            query,
            k=3
        )

        if not docs:
            return "No information was found in the encyclopedia."

        results = []

        for doc in docs:
            source = doc.metadata.get(
                "source",
                "Unknown source"
            )

            results.append(
                f"Source: {source}\n"
                f"Information: {doc.page_content}"
            )

        return "\n\n".join(results)

    except Exception as e:
        return f"Error querying the encyclopedia: {e}"


# ============================================================
# 4. TOOL 2: CHECK THE WEATHER
# ============================================================

@tool
def check_sighting_weather(location: str) -> str:
    """
    Query the current weather for a given location.

    Use this tool when the user asks about the weather
    in an area where they want to observe birds.
    """

    try:

        # Find the location coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Location not found: {location}"

        place = geo_data["results"][0]

        latitude = place["latitude"]
        longitude = place["longitude"]

        name = place.get("name", location)
        country = place.get("country", "")

        # Query meteorology
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "precipitation,"
                    "wind_speed_10m,"
                    "weather_code"
                ),
                "timezone": "auto"
            },
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()
        current = weather_data["current"]

        return (
            f"Location: {name}, {country}\n"
            f"Date and time: {current.get('time')}\n"
            f"Temperature: {current.get('temperature_2m')} °C\n"
            f"Humidity: {current.get('relative_humidity_2m')} %\n"
            f"Precipitation: {current.get('precipitation')} mm\n"
            f"Wind: {current.get('wind_speed_10m')} km/h\n"
            f"Weather code: {current.get('weather_code')}\n"
            "Source: Open-Meteo"
        )

    except requests.RequestException as e:
        return f"Error querying the weather service: {e}"

    except Exception as e:
        return f"Unexpected error: {e}"


# ============================================================
# 5. TOOL 3: REGISTER SIGHTINGS
# ============================================================

@tool
def register_sighting(
    species: str,
    location: str,
    details: str = ""
) -> str:
    """
    Save a sighting in the local log.

    Use this tool only when the user explicitly asks
    to save a sighting.

    Do not invent species or locations.
    """

    try:

        entry = (
            f"Species: {species} | "
            f"Location: {location} | "
            f"Details: {details}\n"
        )

        with open(
            OBSERVATION_LOG,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(entry)

        return (
            "Sighting recorded successfully.\n"
            f"File: {OBSERVATION_LOG}\n"
            f"Entry: {entry}"
        )

    except Exception as e:
        return f"Error saving the sighting: {e}"


# ============================================================
# 6. TOOL REGISTRY
# ============================================================

available_tools = [
    search_bird_encyclopedia,
    check_sighting_weather,
    register_sighting
]

tools_map = {
    tool.name: tool
    for tool in available_tools
}

# ============================================================
# 7. CONFIGURE THE LOCAL MODEL
# ============================================================

llm = ChatOllama(
    model=MODEL,
    temperature=0.2
)

llm_with_tools = llm.bind_tools(available_tools)

# ============================================================
# 8. AGENT INSTRUCTIONS
# ============================================================

instructions = """
You are SKYTERBIRD, an ornithology specialist agent.

Your job is to help users learn about,
identify, and study birds.

You have three tools available:

1. search_bird_encyclopedia:
   Query the RAG knowledge base.

2. check_sighting_weather:
   Query the current weather for a location.

3. register_sighting:
   Save observations to a local log.

RULES:

- Always answer in English.
- Use tools when necessary.
- You may use multiple tools in a single query.
- Analyze the results before answering.
- Do not invent scientific information.
- Do not invent meteorological results.
- Do not register sightings without explicit user request.
- If data is missing to register a sighting,
  ask the user first.
- If you do not have enough information,
  acknowledge that those details are unavailable.
- Explain answers clearly and didactically.
"""

# ============================================================
# 9. AGENT ENGINE: TOOL EXECUTION LOOP
# ============================================================


def execute_agent(user_question: str):

    messages = [
        SystemMessage(content=instructions),
        HumanMessage(content=user_question)
    ]

    max_iterations = 5

    for iteration in range(max_iterations):

        response = llm_with_tools.invoke(messages)

        # Save the model response
        messages.append(response)

        # If it does not request tools, finish
        if not response.tool_calls:

            return response.content

        print("\n🔧 The agent requested tools:")

        # Execute all requested tools
        for call in response.tool_calls:

            name = call["name"]
            arguments = call["args"]
            call_id = call["id"]

            # print(f"\nTool: {name}")
            # print(f"Arguments: {arguments}")

            tool_instance = tools_map.get(name)

            if tool_instance is None:

                result = "Tool not available."

            else:

                try:
                    result = tool_instance.invoke(arguments)

                except Exception as e:
                    result = f"Error executing tool: {e}"

            print(f"Result: {result}")

            # Return the result to the model
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call_id
                )
            )

    return (
        "The agent reached the iteration limit. "
        "Try asking a more specific question."
    )


# ============================================================
# 10. CONVERSATION CONSOLE
# ============================================================

def start_agent():

    print("\n========================================")
    print("       SKYTERBIRD - ORNITHOLOGY AGENT")
    print("========================================")

    print("Available tools:")
    print("- RAG encyclopedia")
    print("- Weather query")
    print("- Sightings log")

    print("\nType 'exit' to finish.\n")

    while True:

        question = input("YOU: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nSKYTERBIRD: See you soon! 🦅")
            break

        if not question:
            continue

        try:

            response = execute_agent(question)

            print("\n🦅 SKYTERBIRD:")
            print(response)
            print("\n----------------------------------------")

        except Exception as e:

            print(f"\nAgent error: {e}")


# ============================================================
# 11. G
# ============================================================

# ============================================================
# 12. RUN THE PROGRAM
# ============================================================

if __name__ == "__main__":
    start_agent()
