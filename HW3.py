import streamlit as st 
from openai import OpenAI
import numpy as np

st.title("MY Lab 3 question answering chatbot")

openAI_model = st.sidebar.selectbox(
    "Which Model?",
    ("mini", "regular")
)

if openAI_model == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

if 'client' not in st.session_state:
    openai_api_key = st.secrets["openai_api_key"]
    st.session_state.client = OpenAI(api_key=openai_api_key)

system_prompt = {
    "role": "system",
    "content": (
        "You are a friendly chatbot. Answer questions so that a 10-year-old "
        "can understand them. Use simple words, short explanations, and "
        "examples when helpful. Avoid complicated technical language. "
        "Do not ask the user if they want more information. The program "
        "will ask that question separately."
    )
}

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your chatbot. How can I assist you today?"
        }
    ]

if "waiting_for_more_info" not in st.session_state:
    st.session_state["waiting_for_more_info"] = False

if "current_topic" not in st.session_state:
    st.session_state["current_topic"] = ""

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    client = st.session_state.client

    if st.session_state.waiting_for_more_info:
        if prompt.lower().strip() in ["yes", "y", "yeah", "yep"]:
            conversation = (
                [system_prompt]
                + st.session_state.messages[-4:]
            )

            stream = client.chat.completions.create(
                model=model_to_use,
                messages=conversation,
                stream=True
            )

            # Display additional information
            with st.chat_message("assistant"):
                response = st.write_stream(stream)

            # Save additional response
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            # Ask again
            more_info_question = "Do you want more info?"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": more_info_question
                }
            )

            with st.chat_message("assistant"):
                st.write(more_info_question)

        # User does not want more information
        else:
            st.session_state.waiting_for_more_info = False

            no_response = "Okay! What can I help you with?"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": no_response
                }
            )

            with st.chat_message("assistant"):
                st.write(no_response)

    # User is asking a new question
    else:

        # Save the current topic
        st.session_state.current_topic = prompt

        # Conversation buffer
        # Keep the system prompt plus the last 4 messages.
        # The last 4 messages represent 2 user/LLM exchanges.
        conversation = (
            [system_prompt]
            + st.session_state.messages[-4:]
        )

        # Send conversation to OpenAI
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=conversation,
            stream=True
        )

        # Display LLM response
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        # Save LLM response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Ask if user wants more information
        more_info_question = "Do you want more info?"

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": more_info_question
            }
        )

        with st.chat_message("assistant"):
            st.write(more_info_question)

        # Tell the chatbot to expect Yes or No
        st.session_state.waiting_for_more_info = True