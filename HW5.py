import json
import streamlit as st
from openai import OpenAI
import sys
import os
from pathlib import Path
from bs4 import BeautifulSoup

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

if 'openai_client' not in st.session_state:
    try:
        openai_api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    except Exception:
        st.session_state.openai_client = None

DATA_FOLDER = "./su_orgs/"
MODEL = "gpt-5-mini" 

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
collection = chroma_client.get_or_create_collection('HW4Collection')

def extract_text_from_html(html_path):
    """Reads an HTML file and returns its visible text content."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator=" ", strip=True)


def chunk_text(text):
    midpoint = len(text) // 2
    split_point = text.rfind(" ", 0, midpoint)
    if split_point == -1:
        split_point = midpoint
    chunk_1 = text[:split_point].strip()
    chunk_2 = text[split_point:].strip()
    return [chunk_1, chunk_2]


def add_to_collection(collection, text, chunk_id, source_filename):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[chunk_id],
        embeddings=[embedding],
        metadatas=[{"filename": source_filename}]
    )


def load_html_files_to_collection(folder_path, collection):
    loaded_files = []
    folder = Path(folder_path)

    for html_path in folder.glob("*.html"):
        filename = html_path.name
        text = extract_text_from_html(str(html_path))

        if not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                chunk_id = f"{filename}_chunk{i}"
                add_to_collection(collection, chunk, chunk_id, filename)

        loaded_files.append(filename)

    return loaded_files

if 'HW5_VectorDB' not in st.session_state:
    if collection.count() == 0:
        with st.spinner("Building vector database from student org pages..."):
            loaded = load_html_files_to_collection(DATA_FOLDER, collection)
            st.sidebar.success(f"Loaded {len(loaded)} org pages ({collection.count()} chunks) into ChromaDB.")
    st.session_state.HW5_VectorDB = collection

collection = st.session_state.HW5_VectorDB

def relevant_club_info(query, n_results=4):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding
 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
 
    context_pieces = []
    sources = []
    for i in range(len(results['documents'][0])):
        doc_text = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        source_file = results['metadatas'][0][i].get('filename', doc_id)
        context_pieces.append(f"--- From {source_file} ---\n{doc_text}")
        sources.append(source_file)
 
    return "\n\n".join(context_pieces), sources
 
 
relevant_club_info_tool = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": (
            "Search the knowledge base of Syracuse University student "
            "organization pages for information relevant to a query. Use "
            "this whenever answering the user's question requires specific "
            "details about a student organization (e.g. its purpose, "
            "meeting times, contact info, how to join). Do not use it for "
            "generic questions that don't need lookup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise search query describing what information "
                        "to look up in the student organizations knowledge base."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}




st.title("Enhanced iSchool Student Organizations Chatbot")

st.write(
    "Ask me about Syracuse University student organizations. Unlike the "
    "HW4 version, this bot doesn't automatically search the knowledge base "
    "on every message. Instead, the LLM is given a `relevant_club_info` "
    "tool and decides on its own when a lookup is needed and what to "
    "search for."
)

st.sidebar.header("Chatbot Settings")
st.sidebar.write("Model: gpt-5-mini")

if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = []

for message in st.session_state.hw5_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question about student organizations...")


if prompt:
    if st.session_state.openai_client is None:
        st.error("OpenAI API key was not found. Please add openai_api_key to Streamlit secrets.")
        st.stop()
 
    client = st.session_state.openai_client
 
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    conversation_buffer = st.session_state.hw5_messages[-10:]
 
    base_system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about "
            "Syracuse University student organizations. You have access to "
            "a relevant_club_info tool that searches a knowledge base of "
            "student org pages. Call it whenever the user's question needs "
            "specific information about an organization. If the question "
            "doesn't need lookup (e.g. small talk, general advice), answer "
            "directly without calling the tool."
        ),
    }
 
    first_messages = [base_system_prompt] + conversation_buffer
    first_response = client.chat.completions.create(
        model=MODEL,
        messages=first_messages,
        tools=[relevant_club_info_tool],
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls
    answer = first_message.content or ""
 
    if tool_calls:
        args = json.loads(tool_calls[0].function.arguments)
        query = args.get("query", prompt)
        rag_context, sources = relevant_club_info(query)
 
        final_system_prompt = {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about "
                "Syracuse University student organizations using the "
                "excerpts provided below. Base your answer on this "
                "retrieved context when it is relevant. If the answer "
                "isn't in the provided context, say so clearly rather than "
                "guessing. Always tell the user which source page(s) your "
                "answer is based on, and make clear when you are using "
                "information retrieved from the knowledge base versus "
                "general knowledge.\n\n"
                f"RETRIEVED CONTEXT (searched for: '{query}'):\n\n{rag_context}"
            ),
        }
        final_messages = [final_system_prompt] + conversation_buffer
 
        try:
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=final_messages,
                    stream=True,
                )
                answer = st.write_stream(stream)
            st.session_state.hw5_messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
 
    else:
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.hw5_messages.append({"role": "assistant", "content": answer})
 