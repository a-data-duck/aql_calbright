import streamlit as st
import openai
import pinecone
import os

# --- Streamlit secrets/config ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("API keys not configured. Please set OPENAI_API_KEY and PINECONE_API_KEY in your app secrets.")
    st.stop()

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1") 
    index = pinecone.Index(PINECONE_INDEX_NAME)
    st.success("Connected to Pinecone successfully!")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def search_pinecone(query_embedding, top_k=5):
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

def format_context(matches):
    context = ""
    sources = []
    
    for i, match in enumerate(matches):
        metadata = match.metadata
        context += f"\nDocument {i+1}:\n"
        context += metadata.get("text", "No text available") + "\n"
        
        source = {
            "title": metadata.get("title", "Unknown"),
            "url": metadata.get("url", "N/A"),
        }
        sources.append(source)
    
    return context, sources

def generate_answer(query, context):
    prompt = f"""You are a helpful assistant for Calbright College, a fully online California community college.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always be helpful, friendly, and concise.

Context:
{context}

Question: {query}
Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Calbright College."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="Calbright Info Chatbot", page_icon="🎓")
st.title("🎓 DEMO Calbright Info Chatbot")
st.write("Ask questions about the college's admissions, policies, and services.")

# Text input for questions
query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Retrieving answer..."):
        try:
            # Get query embedding
            query_embedding = get_embedding(query)
            
            # Search Pinecone
            matches = search_pinecone(query_embedding)
            
            # Format context and get sources
            context, sources = format_context(matches)
            
            # Generate answer
            answer = generate_answer(query, context)
            
            st.subheader("Answer:")
            st.write(answer)
            
            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {source['title']}")
                        st.markdown(f"**URL:** {source['url']}")
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error getting answer: {e}")
