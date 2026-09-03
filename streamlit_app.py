import streamlit as st
from openai import APIError, AuthenticationError, OpenAI
import fitz

# Show title and description
st.title("MY Document Question Answering")

st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password"
)

client = None

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

    st.success("API key accepted! You can use the document Q&A app.")

else:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")


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
    disabled=not openai_api_key,
)


# Ask question
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=uploaded_file is None,
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
if uploaded_file is not None and question and client is not None:

    messages = [
        {
            "role": "user",
            "content": (
                f"Here's a document:\n\n"
                f"{document}\n\n"
                f"---\n\n"
                f"Question: {question}"
            ),
        }
    ]

    try:

        # Generate an answer using the OpenAI API
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )

        st.write_stream(stream)

    except APIError as e:

        st.error(f"OpenAI API error: {e}")
