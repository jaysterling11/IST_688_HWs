import streamlit as st
from openai import OpenAI
import sys
import os
from pathlib import Path
from PyPDF2 import PdfReader

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

if 'openai_client' not in st.session_state:
    try:
        openai_api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    except Exception:
        st.session_state.openai_client = None

DATA_FOLDER = "./Lab-04-Data/"  
 
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('Lab4Collection')
 
 
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
 
 
def add_to_collection(collection, text, filename):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
 
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding],
        metadatas=[{"filename": filename}]
    )
 
 
def load_pdfs_to_collection(folder_path, collection):
    loaded_files = []
    folder = Path(folder_path)
 
    for pdf_path in folder.glob("*.pdf"):
        filename = pdf_path.name
        text = extract_text_from_pdf(str(pdf_path))
 
        if text.strip():
            add_to_collection(collection, text, filename)
            loaded_files.append(filename)
 
    return loaded_files
 
 
if 'Lab4_VectorDB' not in st.session_state:
    if collection.count() == 0:
        with st.spinner("Building vector database from course PDFs..."):
            loaded = load_pdfs_to_collection(DATA_FOLDER, collection)
            st.sidebar.success(f"Loaded {len(loaded)} PDFs into ChromaDB.")
    st.session_state.Lab4_VectorDB = collection
 
collection = st.session_state.Lab4_VectorDB

st.title("Lab 4 Chatbot using RAG (ChromaDB)")

st.write(
    "This chatbot answers questions about Syracuse iSchool course syllabi "
    "using a ChromaDB vector database built from the course PDFs. "
    "It retrieves the most relevant syllabus excerpts for each question and "
    "passes them to the LLM as context. "
    "A short-term memory buffer of the last 6 messages is also maintained."
)

st.sidebar.header("Chatbot Settings")
st.sidebar.write("Model: gpt-5-mini")

# test_topic = st.sidebar.text_input('Test topic', placeholder='e.g., Generative AI')
# if test_topic:
#     client = st.session_state.openai_client
#     response = client.embeddings.create(input=test_topic, model='text-embedding-3-small')
#     query_embedding = response.data[0].embedding
#     results = collection.query(query_embeddings=[query_embedding], n_results=3)
#     st.sidebar.subheader(f"Results for: {test_topic}")
#     for i in range(len(results['documents'][0])):
#         doc_id = results['ids'][0][i]
#         st.sidebar.write(f"**{i+1}. {doc_id}**")

# Part B
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
prompt = st.chat_input("Ask a question about the courses...")
 
 
def get_relevant_context(user_prompt, n_results=3):
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
        context_pieces.append(f"--- From {doc_id} ---\n{doc_text}")
        sources.append(doc_id)
 
    return "\n\n".join(context_pieces), sources
 
 
if prompt:
    if st.session_state.openai_client is None:
        st.error("OpenAI API key was not found. Please add openai_api_key to Streamlit secrets "
                  "(needed for embeddings, even if you chat with Gemini).")
        st.stop()
 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    # Retrieve relevant context from ChromaDB
    rag_context, sources = get_relevant_context(prompt)
 
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about Syracuse "
            "iSchool courses using the syllabus excerpts provided below. "
            "Base your answer on this retrieved context when it is relevant. "
            "If the answer isn't in the provided context, say so clearly. "
            "Always tell the user which source document(s) your answer is "
            "based on, and make clear when you are using information "
            "retrieved from the RAG knowledge base versus general knowledge.\n\n"
            f"RETRIEVED CONTEXT:\n\n{rag_context}"
        )
    }
 
    conversation_buffer = st.session_state.messages[-6:]
    conversation = [system_prompt] + conversation_buffer
 
    try:
        with st.chat_message("assistant"):
            stream = st.session_state.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=conversation,
                stream=True
            )
            response = st.write_stream(stream)
 
        st.session_state.messages.append({"role": "assistant", "content": response})
 
    except Exception as e:
        st.error(f"OpenAI API error: {e}")