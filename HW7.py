import json
import os
import sys
import pandas as pd
import streamlit as st
from openai import OpenAI

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

if 'openai_client' not in st.session_state:
    try:
        openai_api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    except Exception:
        st.session_state.openai_client = None

CSV_FILE = "news.csv"
CHROMA_PATH = "./ChromaDB_HW7"
COLLECTION_NAME = "LawFirmNewsCollection"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)


def initialize_vector_db():
    if collection.count() == 0 and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        client = st.session_state.openai_client
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            docs = batch["Document"].tolist()
            ids = [f"news_{idx}" for idx in batch.index]
            metadatas = [
                {
                    "company_name": str(row["company_name"]),
                    "date": str(row["Date"]),
                    "url": str(row["URL"]),
                    "days_since_2000": int(row["days_since_2000"]),
                }
                for _, row in batch.iterrows()
            ]

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=docs
            )
            embeddings = [record.embedding for record in response.data]

            collection.add(
                documents=docs,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )


if "HW7_VectorDB" not in st.session_state:
    if collection.count() == 0 and st.session_state.openai_client:
        with st.spinner("Indexing news dataset into ChromaDB..."):
            initialize_vector_db()
    st.session_state.HW7_VectorDB = collection

collection = st.session_state.HW7_VectorDB


def search_news(query, company_filter=None, n_results=6):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding

    where_filter = None
    if company_filter and company_filter.strip():
        where_filter = {"company_name": {"$eq": company_filter.strip()}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

    context_pieces = []
    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        context_pieces.append(
            f"Client/Company: {meta.get('company_name')}\n"
            f"Date: {meta.get('date')}\n"
            f"Article: {doc_text}\n"
            f"URL: {meta.get('url')}"
        )
    return "\n\n---\n\n".join(context_pieces)


search_news_tool = {
    "type": "function",
    "function": {
        "name": "search_news",
        "description": (
            "Searches the curated news database for client updates, legal risks, "
            "corporate developments, or high-impact events. Call this to retrieve "
            "news for specific companies or to evaluate news stories for ranking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or semantic concepts to retrieve relevant stories.",
                },
                "company_filter": {
                    "type": "string",
                    "description": "Optional exact company or client name to filter results.",
                },
            },
            "required": ["query"],
        },
    },
}

st.title("Law Firm Client News Bot")
st.write(
    "Monitors corporate actions, litigation, regulatory investigations, and major "
    "developments across firm clients using curated news records."
)

st.sidebar.header("Model Configuration")
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    options=["gpt-4o-mini", "gpt-4o"],
    index=0,
)

if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = []

for message in st.session_state.hw7_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Inquire about client news or request ranked developments...")

if prompt:
    if st.session_state.openai_client is None:
        st.error("OpenAI API key missing. Please provide it in Streamlit secrets.")
        st.stop()

    client = st.session_state.openai_client
    st.session_state.hw7_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    conversation_buffer = st.session_state.hw7_messages[-8:]

    base_system_prompt = {
        "role": "system",
        "content": (
            "You are an intelligence analyst for a global law firm monitoring corporate clients. "
            "You have access to the search_news tool containing verified news records. "
            "When users request the most interesting news, search for high-impact developments "
            "and rank articles in descending order of significance. "
            "Provide explicit business and legal context explaining why each ranked story matters. "
            "Never fabricate outside stories or discuss unverified information."
        ),
    }

    first_response = client.chat.completions.create(
        model=selected_model,
        messages=[base_system_prompt] + conversation_buffer,
        tools=[search_news_tool],
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls
    answer = first_message.content or ""

    if tool_calls:
        args = json.loads(tool_calls[0].function.arguments)
        search_query = args.get("query", prompt)
        client_target = args.get("company_filter", None)
        rag_context = search_news(search_query, company_filter=client_target)

        final_system_prompt = {
            "role": "system",
            "content": (
                "You are an intelligence analyst for a global law firm. "
                "Synthesize an answer using exclusively the retrieved records provided below. "
                "For rankings, present the stories in clear descending order and detail the legal risk "
                "or strategic significance of each event. "
                "Cite the company, publication date, and source URL for every referenced item.\n\n"
                f"RETRIEVED NEWS RECORDS:\n{rag_context}"
            ),
        }

        try:
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=[final_system_prompt] + conversation_buffer,
                    stream=True,
                )
                answer = st.write_stream(stream)
            st.session_state.hw7_messages.append({"role": "assistant", "content": answer})
        except Exception as err:
            st.error(f"OpenAI API execution error: {err}")
    else:
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.hw7_messages.append({"role": "assistant", "content": answer})