import os
import streamlit as st
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Streamlit secrets/config ---
# Set OPENAI_API_KEY and PINECONE_API_KEY in Streamlit Cloud secrets
# Accessed via st.secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("API keys not configured. Please set OPENAI_API_KEY and PINECONE_API_KEY in your app secrets.")
    st.stop()

# --- Initialize Pinecone client ---
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# --- Setup LangChain retriever ---
embeddings = OpenAIEmbeddings()
vectorstore = LC_Pinecone(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Setup LLM and QA chain ---
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="Calbright Info Chatbot", page_icon="🎓")
st.title("🎓 DEMO Calbright Info Chatbot")
st.write("Ask questions about the college's admissions, policies, and services.")

query = st.text_input("Enter your question:")
if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):
        result = qa(query)
        answer = result.get("result") or result.get("output_text")
        docs = result.get("source_documents", [])

    st.subheader("Answer:")
    st.write(answer)

    if docs:
        st.subheader("Sources:")
        for doc in docs:
            md = doc.metadata
            st.markdown(f"- **URL**: {md.get('url', 'N/A')}  \n  **Chunk**: {md.get('chunk_index', '')}")
