import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from google import genai

st.title("Baseball Question Answering Chatbot")

st.write(
    "This chatbot answers questions using information from up to two "
    "webpages that you provide. The webpages are included in the chatbot's "
    "system prompt so they remain available throughout the conversation. "
    "The chatbot also uses a conversation memory buffer containing the "
    "last 6 messages, which represents 3 user-assistant exchanges. "
    "You can choose between OpenAI and Google Gemini in the sidebar."
)

st.sidebar.header("Chatbot Settings")

url_1 = st.sidebar.text_input(
    "Enter URL 1:",
    placeholder="https://www.example.com"
)

url_2 = st.sidebar.text_input(
    "Enter URL 2 (optional):",
    placeholder="https://www.example.com"
)

llm_choice = st.sidebar.selectbox(
    "Select the LLM:",
    (
        "OpenAI",
        "Google Gemini",
    )
)

if llm_choice == "OpenAI":

    st.sidebar.write("Model: gpt-5.6")

else:

    st.sidebar.write("Model: gemini-2.5-pro")

if 'openai_client' not in st.session_state:
    openai_api_key = st.secrets.get("openai_api_key")
    if openai_api_key:
        st.session_state.openai_client = OpenAI(api_key=openai_api_key)
else:
    st.session_state.openai_client = None

if 'gemini_client' not in st.session_state:
    gemini_api_key = st.secrets.get("gemini_api_key")
    if gemini_api_key:
        st.session_state.gemini_client = genai.Client(api_key=gemini_api_key)
else:
    st.session_state.gemini_client = None

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

documents = []

if url_1:
    content_1 = read_url_content(url_1)
    if content_1:
        documents.append(
            f"Document 1\n URL: {url_1}\n\n {content_1}"
        )

if url_2:
    content_2 = read_url_content(url_2)
    if content_2:
        documents.append(
            f"Document 2\n URL: {url_2}\n\n {content_2}"
        )

if documents:
    webpage_content = "\n\n".join(documents)
else:
    webpage_content = ("No webpages have been provided yet. "
        "Ask the user to enter at least one URL."
    )

system_prompt = {
    "role": "system",
    "content": (
        "You are a friendly question-answering chatbot. "
        "Answer questions using the webpage information provided below "
        "when it is relevant. Do not make up information that is not "
        "supported by the webpages. If the answer cannot be found in "
        "the provided webpages, clearly say that the information was "
        "not found in the provided sources. "
        "\n\n"
        "The following webpage information is permanent context for "
        "this conversation and should be included in every API request. "
        "\n\n"
        "WEBPAGE CONTENT:\n\n"
        f"{webpage_content}"
    )
}

if documents:
    st.sidebar.success(
        f"{len(documents)} webpage(s) loaded."
    )

else:
    st.sidebar.info(
        "Enter at least one URL to provide information "
        "for the chatbot."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(
            message["content"]
        )

prompt = st.chat_input(
    "Ask a question about the webpages..."
)


if prompt:
    if not documents:
        st.error(
            "Please enter at least one URL in the sidebar "
            "before asking a question."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)

    conversation_buffer = (
        st.session_state.messages[-6:]
    )


    conversation = (
        [system_prompt]
        + conversation_buffer
    )

if llm_choice == "OpenAI":

        if st.session_state.openai_client is None:

            st.error(
                "OpenAI API key was not found. "
                "Please add openai_api_key to Streamlit secrets."
            )

        else:

            try:

                with st.chat_message("assistant"):

                    stream = (
                        st.session_state.openai_client
                        .chat.completions.create(
                            model="gpt-5.6",
                            messages=conversation,
                            stream=True
                        )
                    )

                    response = st.write_stream(
                        stream
                    )


                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )


            except Exception as e:

                st.error(
                    f"OpenAI API error: {e}"
                )

elif llm_choice == "Google Gemini":

        if st.session_state.gemini_client is None:

            st.error(
                "Google Gemini API key was not found. "
                "Please add gemini_api_key to Streamlit secrets."
            )

        else:

            try:
                gemini_prompt = ""

                for message in conversation:

                    if message["role"] == "system":

                        gemini_prompt += (
                            "SYSTEM INSTRUCTIONS:\n"
                            + message["content"]
                            + "\n\n"
                        )

                    elif message["role"] == "user":

                        gemini_prompt += (
                            "USER:\n"
                            + message["content"]
                            + "\n\n"
                        )

                    elif message["role"] == "assistant":

                        gemini_prompt += (
                            "ASSISTANT:\n"
                            + message["content"]
                            + "\n\n"
                        )


                with st.chat_message("assistant"):

                    response = (
                        st.session_state.gemini_client
                        .models.generate_content(
                            model="gemini-2.5-pro",
                            contents=gemini_prompt
                        )
                    )

                    if response.text:

                        st.write(
                            response.text
                        )

                    else:

                        st.error(
                            "Gemini did not return a response."
                        )

                if response.text:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.text
                        }
                    )


            except Exception as e:

                st.error(
                    f"Google Gemini API error: {e}"
                )