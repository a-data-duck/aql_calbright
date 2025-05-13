import os
import streamlit as st
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# === Streamlit page config ===
st.set_page_config(
    page_title="Calbright College Q&A",
    page_icon="🎓",
    layout="centered",
)

# === Custom CSS for Calbright theme & hide sidebar by default ===
calbright_blue = "#005aad"
calbright_gold = "#ffd100"
hide_sidebar_css = """
<style>
    /* Hide default menu and footer */
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;} 
    /* Primary color for buttons/text */
    .stButton>button {background-color: %s; color: white; border: none;} 
    .st-badge {background-color: %s; color: white;} 
    /* Heading color */
    h1, h2, h3 {color: %s;} 
</style>
""" % (calbright_blue, calbright_blue, calbright_blue)
st.markdown(hide_sidebar_css, unsafe_allow_html=True)

# === Secrets/config ===
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ENV = st.secrets.get("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("API keys not configured. Please set OPENAI_API_KEY and PINECONE_API_KEY in your app secrets.")
    st.stop()

# === Sidebar (collapsed by default) ===
with st.sidebar.expander("🔧 Configuration", expanded=False):
    st.write("**Pinecone Settings**")
    pinecone_url = st.text_input(
        "Pinecone URL:",
        value=st.secrets.get("PINECONE_URL", "https://calbright-docs-h3y3rrq.svc.aped-4627-b74a.pinecone.io")
    )
    if st.button("Test Pinecone Connection"):
        try:
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            idx = pinecone.Index(INDEX_NAME)
            # simple metadata fetch
            idx.describe_index_stats()
            st.success("Successfully connected to Pinecone!")
        except Exception as e:
            st.error(f"Connection error: {e}")

# === Initialize Pinecone & LangChain QA ===
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings()
vectorstore = LC_Pinecone(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# === Main UI ===
st.title("🎓 Calbright College Q&A")
st.write("Ask questions about admissions, programs, and services at Calbright College.")
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            result = qa(query)
            answer = result.get("result") or result.get("output_text")
            docs = result.get("source_documents", [])

        st.subheader("Answer")
        st.markdown(f"<div style='background-color:{calbright_gold}; padding:10px; border-radius:5px;'>**{answer}**</div>", unsafe_allow_html=True)

        if docs:
            st.subheader("Sources")
            for doc in docs:
                md = doc.metadata
                st.markdown(f"- **{md.get('title','')}** (<a href='{md.get('url','')}' target='_blank'>link</a>) -- chunk {md.get('chunk_index','')}")
