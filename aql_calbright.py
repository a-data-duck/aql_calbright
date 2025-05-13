import os
import streamlit as st
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# --- Streamlit secrets/config ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("API keys not configured. Please set OPENAI_API_KEY and PINECONE_API_KEY in your app secrets.")
    st.stop()

# --- Initialize Pinecone client (updated for newer SDK) ---
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    st.session_state.pinecone_initialized = True
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# --- Setup LangChain components with better exception handling ---
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Create a Pinecone vector store
    vectorstore = LC_Pinecone(
        index=index,
        embedding_function=embeddings.embed_query,
        text_key="text"
    )
    
    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create a custom prompt template
    custom_prompt_template = """You are a helpful assistant for Calbright College, a fully online California community college.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always be helpful, friendly, and concise.

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Setup LLM and QA chain with custom prompt
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
except Exception as e:
    st.error(f"Error setting up LangChain components: {e}")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="Calbright Info Chatbot", page_icon="🎓")
st.title("🎓 DEMO Calbright Info Chatbot")
st.write("Ask questions about the college's admissions, policies, and services.")

# Add some example questions as buttons
st.write("Try one of these questions:")
example_questions = [
    "Who provides wellness services at Calbright?",
    "What programs does Calbright offer?",
    "What financial support is available to Calbright students?",
    "How long does it take to complete a program at Calbright?"
]

# Create two columns for the example buttons
cols = st.columns(2)
for i, question in enumerate(example_questions):
    col_idx = i % 2
    if cols[col_idx].button(question):
        st.session_state.query = question

# Text input for custom questions
query = st.text_input("Or enter your own question:", value=st.session_state.get("query", ""))

if st.button("Ask") or ('query' in st.session_state and st.session_state.query):
    # Get the query from session state if it exists, otherwise use the input
    query_to_use = query or st.session_state.get("query", "")
    
    if query_to_use:
        with st.spinner("Retrieving answer..."):
            try:
                result = qa({"query": query_to_use})
                answer = result.get("result", "")
                docs = result.get("source_documents", [])
                
                # Clear the session state query after using it
                if 'query' in st.session_state:
                    del st.session_state.query
                    
                st.subheader("Answer:")
                st.write(answer)
                
                if docs:
                    with st.expander("Sources"):
                        for i, doc in enumerate(docs):
                            md = doc.metadata
                            st.markdown(f"**Source {i+1}:** {md.get('title', 'No title')}")
                            st.markdown(f"**URL:** {md.get('url', 'N/A')}")
                            st.markdown(f"**Chunk Index:** {md.get('chunk_index', '')}")
                            st.markdown("---")
            except Exception as e:
                st.error(f"Error getting answer: {e}")
