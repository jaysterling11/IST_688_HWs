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

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
collection = chroma_client.get_or_create_collection('HW4Collection')

def extract_text_from_html(html_path):
    """Reads an HTML file and returns its visible text content."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator=" ", strip=True)


def chunk_text(text):
    """Splits text into two roughly equal chunks, breaking at whitespace."""
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

if 'HW4_VectorDB' not in st.session_state:
    if collection.count() == 0:
        with st.spinner("Building vector database from student org pages..."):
            loaded = load_html_files_to_collection(DATA_FOLDER, collection)
            st.sidebar.success(f"Loaded {len(loaded)} org pages ({collection.count()} chunks) into ChromaDB.")
    st.session_state.HW4_VectorDB = collection

collection = st.session_state.HW4_VectorDB

st.title("iSchool Student Organizations Chatbot")

st.write(
    "Ask me about Syracuse University student organizations. I use a "
    "ChromaDB vector database built from student org web pages to find "
    "relevant information and answer your questions. "
    "I remember the last 5 exchanges in our conversation."
)

st.sidebar.header("Chatbot Settings")
st.sidebar.write("Model: gpt-5-mini")

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = []

for message in st.session_state.hw4_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question about student organizations...")


def get_relevant_context(user_prompt, n_results=4):
    """Embeds the user prompt and retrieves the top matching chunks."""
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=user_prompt,
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

if prompt:
    if st.session_state.openai_client is None:
        st.error("OpenAI API key was not found. Please add openai_api_key to Streamlit secrets.")
        st.stop()

    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve relevant context from ChromaDB
    rag_context, sources = get_relevant_context(prompt)

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about "
            "Syracuse University student organizations using the excerpts "
            "provided below. Base your answer on this retrieved context "
            "when it is relevant. If the answer isn't in the provided "
            "context, say so clearly rather than guessing. Always tell the "
            "user which source page(s) your answer is based on, and make "
            "clear when you are using information retrieved from the RAG "
            "knowledge base versus general knowledge.\n\n"
            f"RETRIEVED CONTEXT:\n\n{rag_context}"
        )
    }

    conversation_buffer = st.session_state.hw4_messages[-10:]
    conversation = [system_prompt] + conversation_buffer

    try:
        with st.chat_message("assistant"):
            stream = st.session_state.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=conversation,
                stream=True
            )
            response = st.write_stream(stream)

        st.session_state.hw4_messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"OpenAI API error: {e}")