import re
import json
import streamlit as st
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

MODEL = "qwen3:latest"
model = ChatOllama(model=MODEL)

with open("data/IpasargadQuesationAnswer.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

docs = []
for category in faq_data["categories"]:
    for faq in category["faqs"]:
        content = f"سوال: {faq['question']}\n{faq['answer']}"
        docs.append(Document(page_content=content, metadata={"category": category["title"]}))

embeddings = OllamaEmbeddings(model="bge-m3")
vector_store = DocArrayInMemorySearch.from_documents(documents=docs, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

main_template = """
You are a friendly and helpful assistant for Ipasargad. 
Your task is to answer a user's question based on a specific set of rules.

Rule 1: If the answer to the question is explicitly available in the provided context, answer the question accurately based on that context.
Rule 2: If the question is a casual, conversational greeting or a simple non-technical request (e.g., "سلام", "حالت چطوره؟"), ignore the context and provide a short, friendly, and informal answer in Persian.
Rule 3: If neither of the above rules apply (i.e., the question is not in the context and is not a conversational one), reply with "I don't know.".

Context:
{context}

Question: {question}

Please follow the rules strictly.
"""
main_prompt = PromptTemplate.from_template(main_template)
main_chain = main_prompt | model | StrOutputParser()

def clean_answer(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def normalize_query(q: str) -> str:
    return re.sub(r"[^\u0600-\u06FF\s]", "", q).strip()

st.set_page_config(page_title="📘 FAQ Q&A Bot", page_icon="🤖", layout="centered")
st.markdown(
    """
    <style>
    .stApp, .css-1aumxhk, .stTextInput>div>div>input {
        direction: rtl; text-align: right;
    }
    .stTitle h1 { direction: rtl; text-align: right; }
    .stSubheader h3 { direction: rtl; text-align: right; }
    .user-msg {
        background-color: #DCF8C6; padding: 10px 15px;
        border-radius: 15px 15px 0px 15px; float: right;
        max-width: 80%; clear: both;
    }
    .bot-msg {
        background-color: #ECECEC; padding: 10px 15px;
        border-radius: 15px 15px 15px 0px; float: left;
        max-width: 80%; clear: both;
    }
    .timestamp {
        font-size: 0.7em; color: gray; display: block; margin-top: 2px;
    }
    .clearfix { clear: both; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("📘 چت‌بات پرسش و پاسخ آی‌پاسارگاد")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("سوال خود را وارد کنید.")

if user_input:
    normalized = normalize_query(user_input)

    with st.spinner("در حال پردازش..."):
        docs = retriever.get_relevant_documents(normalized)
        context_text = "\n\n".join([d.page_content for d in docs])

        raw_answer = main_chain.invoke({
            "context": context_text,
            "question": user_input
        })
        answer = clean_answer(raw_answer)

        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({
            "user": user_input,
            "bot": answer,
            "time": timestamp
        })

chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        st.markdown(
            f'<div class="user-msg">{chat["user"]}'
            f'<span class="timestamp">{chat["time"]}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="bot-msg">{chat["bot"]}'
            f'<span class="timestamp">{chat["time"]}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <script>
    const chatContainer = window.parent.document.querySelector('.stContainer');
    if (chatContainer) { chatContainer.scrollTop = chatContainer.scrollHeight; }
    </script>
    """,
    unsafe_allow_html=True
)
