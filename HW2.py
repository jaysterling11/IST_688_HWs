import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import APIError, AuthenticationError, OpenAI
from google import genai
from google.genai import types

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

llm_choice = st.sidebar.selectbox(
    "Select the LLM to use",
    (
        "OpenAI",
        "Google Gemini",
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

            st.error(f"OpenAI API error: {e}")

def create_prompt(document):
    return (
        f"{instruction}\n\n"
        f"Write the summary in {language}.\n\n"
        f"Webpage content:\n\n"
        f"{document}"
    )

def summarize_with_openai(document):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": create_prompt(document)}],
            stream=True,
        )
        st.write_stream(stream)
    except APIError as e:
        st.error(f"OpenAI API error: {e}")

def summarize_with_gemini(document):

    gemini_api_key = st.secrets.get("gemini_api_key")

    if not gemini_api_key:
        st.error(
            "Google Gemini API key not found. "
            "Please add gemini_api_key to your Streamlit secrets."
        )
        return

    try:

        # Create Gemini client
        client = genai.Client(
            api_key=gemini_api_key
        )

        # Choose Gemini model
        if use_advanced_model:
            model = "gemini-2.5-pro"
        else:
            model = "gemini-2.5-flash"

        prompt = create_prompt(document)

        # Generate the summary
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        if response.text:
            st.write(response.text)
        else:
            st.error(
                "Gemini did not return a response."
            )

    except Exception as e:

        st.error(
            f"Google Gemini API error: {e}"
        )


if url:

    document = read_url_content(url)

    if document:

        # Display which LLM is being used
        st.subheader(
            f"Summary using {llm_choice}"
        )

        # Switch between LLM providers
        if llm_choice == "OpenAI":

            summarize_with_openai(document)

        elif llm_choice == "Google Gemini":

            summarize_with_gemini(document)