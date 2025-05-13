import os
import streamlit as st

from openai import OpenAI
from pinecone import Pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "calbright-docs"

# --- Initialize clients ---
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Load Pinecone index ---
index = pc = c = None
try:
    index = pc = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Error connecting to Pinecone: {e}")
    st.stop()

# --- Set up LangChain retriever ---
embeddings = OpenAIEmbeddings()
vectorstore = LC_Pinecone(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Set up the LLM and QA chain ---
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="DEMO - Calbright Info Chatbot", page_icon="🎓")
st.title("🎓 DEMO - Calbright Info Chatbot")
st.write("Ask questions about the college's admissions, policies, and services.")

query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):
        result = qa(query)
        answer, docs = result["result"], result["source_documents"]

    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Sources:")
    for doc in docs:
        md = doc.metadata
        st.markdown(f"- **URL**: {md.get('url', 'N/A')}  \n  **Chunk**: {md.get('chunk_index', '')}")
