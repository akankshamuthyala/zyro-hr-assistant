import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def build_pipeline():
    loader = PyPDFDirectoryLoader("docs/")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512, api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR assistant for Zyro Dynamics. Answer using ONLY the HR policy documents below. Be specific with numbers and details. If not found, say you dont have that information.\n\nContext: {context}"),
        ("human", "{question}")
    ])
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    def chain(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        response = llm.invoke(prompt.invoke({"context": context, "question": question}))
        return StrOutputParser().invoke(response)
    return chain

REFUSAL = "I'm only able to answer HR-related questions about Zyro Dynamics policies."

HR_KEYWORDS = ["leave", "salary", "policy", "wfh", "work from home", "probation", 
               "benefit", "conduct", "onboarding", "travel", "expense", "performance",
               "review", "compensation", "harassment", "separation", "it", "data"]

def is_hr_question(question):
    q = question.lower()
    return any(k in q for k in HR_KEYWORDS)

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about Zyro Dynamics HR policies")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("Loading HR documents..."):
    chain = build_pipeline()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_hr_question(user_input):
                response = chain(user_input)
            else:
                response = REFUSAL
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
