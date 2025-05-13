import streamlit as st
import os
import json
from openai import OpenAI
import pinecone

# Set page configuration
st.set_page_config(
    page_title="Calbright College Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Display app header
st.title("🎓 Calbright College Chatbot")
st.markdown("Ask questions about Calbright College's programs, services, and resources.")

# Configuration and API keys
api_keys_configured = True

try:
    # Get API keys from Streamlit secrets
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")
    
    # Initialize clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Initialize Pinecone index
    index = pinecone_client.Index(PINECONE_INDEX_NAME)
    
    # Test connections
    st.sidebar.success("✅ Connected to OpenAI and Pinecone")
    
except Exception as e:
    st.sidebar.error(f"❌ Configuration error: {str(e)}")
    api_keys_configured = False
    st.stop()

# Functions for the chatbot
def get_embedding(text):
    """Generate embedding for text using OpenAI API"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def search_documents(query_embedding, top_k=5):
    """Search Pinecone for relevant documents"""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches
    except Exception as e:
        st.error(f"Error searching Pinecone: {str(e)}")
        return []

def generate_answer(query, context):
    """Generate answer using OpenAI"""
    try:
        prompt = f"""You are a helpful assistant for Calbright College, a fully online California community college.
Answer the question based ONLY on the context provided below. If the information isn't in the context, say you don't know.
Be specific about services, programs, and resources offered by Calbright College.

Context:
{context}

Question: {query}
Answer:"""

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Calbright College."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error generating your answer."

def format_context(matches):
    """Format search results into context"""
    context = ""
    sources = []
    
    for i, match in enumerate(matches):
        metadata = match.metadata
        text = metadata.get("text", "No text available")
        url = metadata.get("url", "N/A")
        title = metadata.get("title", "Unknown")
        
        context += f"\nDocument {i+1}:\n{text}\n"
        
        sources.append({
            "title": title,
            "url": url,
            "score": match.score
        })
    
    return context, sources

# Display example questions
st.sidebar.header("Example Questions")
example_questions = [
    "Who provides wellness services at Calbright?",
    "What programs does Calbright offer?",
    "Is Calbright College free?",
    "How long does it take to complete a program?"
]

for question in example_questions:
    if st.sidebar.button(question):
        st.session_state.question = question

# Input area
if "question" not in st.session_state:
    st.session_state.question = ""

question = st.text_input("Your question:", value=st.session_state.question)

# Ask button
if st.button("Ask") and question:
    st.session_state.question = ""  # Clear for next question
    
    with st.spinner("Thinking..."):
        # 1. Generate embedding for query
        embedding = get_embedding(question)
        if embedding is None:
            st.error("Failed to generate embedding for query")
            st.stop()
        
        # 2. Search for relevant documents
        matches = search_documents(embedding)
        if not matches:
            st.warning("No relevant information found in the knowledge base.")
            st.stop()
            
        # 3. Format context from search results
        context, sources = format_context(matches)
        
        # 4. Generate answer
        answer = generate_answer(question, context)
        
        # 5. Display answer
        st.markdown("### Answer")
        st.markdown(answer)
        
        # 6. Display sources
        if sources:
            st.markdown("### Sources")
            for i, source in enumerate(sources):
                with st.expander(f"Source {i+1}: {source['title']}"):
                    st.markdown(f"**URL:** {source['url']}")
                    st.markdown(f"**Relevance Score:** {source['score']:.2f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*This is a demo chatbot for Calbright College*")
