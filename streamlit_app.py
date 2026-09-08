import streamlit as st

homework1_page = st.Page("HW1.py", title="Homework 1 - Create The App", icon="1️⃣")
homework2_page = st.Page("HW2.py", title="Homework 2 - URL Summarizer with Multiple LLMs", icon="2️⃣", default=True)
homework3_page = st.Page("HW3.py", title="Homework 3 - Chatbot with Memory and Context", icon="3️⃣")

pg = st.navigation([homework1_page, homework2_page])
pg.run()
