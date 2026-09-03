import streamlit as st
from openai import APIError, AuthenticationError, OpenAI
import fitz

# Show title and description
st.title("Document Summarizer")

st.write(
    "Upload a document below, then choose a summary type and model from the "
    "sidebar – GPT will generate a summary for you!"
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

# Function for reading PDFs
def read_pdf(file_obj):
    pdf_bytes = file_obj.getvalue()

    if not pdf_bytes:
        st.error("The uploaded PDF is empty or could not be read.")
        st.stop()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = "\n".join(
        page.get_text()
        for page in doc
    )

    doc.close()

    return text


# Upload document
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf"],
)


# Process uploaded document
document = ""
file_extension = None

if uploaded_file is not None:

    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "txt":

        document = uploaded_file.getvalue().decode("utf-8")

    elif file_extension == "pdf":

        document = read_pdf(uploaded_file)

    else:

        st.error("Unsupported file type.")
        st.stop()


# Answer question
if uploaded_file is not None:
    messages = [
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"Document:\n\n"
                f"{document}\n\n"
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
