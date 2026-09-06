import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import APIError, AuthenticationError, OpenAI

# Show title and description
st.title("URL Summarizer")

st.write(
    "Enter a webpage URL below, then choose a summary type and model "
    "from the sidebar."
)

openai_api_key = st.secrets["openai_api_key"]

client = OpenAI(api_key=openai_api_key)

# Create OpenAI client
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)

    try:
        client.models.list()

    except AuthenticationError:
        st.error("Invalid OpenAI API key. Please check the key you entered.")
        st.stop()

    except APIError as e:
        st.error(f"OpenAI API error: {e}")
        st.stop()

    except Exception as e:
        st.error(f"Unable to connect to OpenAI: {e}")
        st.stop()

# URL input
url = st.text_input(
    "Enter a webpage URL:",
    placeholder="https://example.com"
)

summary_type = st.sidebar.selectbox(
    "Choose a summary type",
    (
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ),
)

use_advanced_model = st.sidebar.checkbox("Use advanced model")
 
# Map the checkbox choice to an actual model name
model = "gpt-5-mini" if use_advanced_model else "gpt-5-nano"

# Output language
language = st.sidebar.selectbox(
    "Select the output language",
    (
        "English",
        "French",
        "Spanish",
    ),
)
 
instruction_map = {
    "Summarize in 100 words": "Summarize the following document in about 100 words.",
    "Summarize in 2 connecting paragraphs": (
        "Summarize the following document in 2 connecting paragraphs."
    ),
    "Summarize in 5 bullet points": (
        "Summarize the following document in 5 concise bullet points."
    ),
}
instruction = instruction_map[summary_type]

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

if url:

    document = read_url_content(url)

    if document:

        messages = [
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    f"Write the summary in {language}.\n\n"
                    f"Webpage content:\n\n"
                    f"{document}"
                ),
            }
        ]

        try:

            # Generate an answer using the OpenAI API
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            st.write_stream(stream)

        except APIError as e:

            st.error(f"OpenAI API error: {e}"
