import streamlit as st
import json
import requests

# Set title
st.title("Calbright College Q&A")

# Configure API keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")
PINECONE_ENV = "us-east-1"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Missing API keys. Please set them in your Streamlit secrets.")
    st.stop()

# Function to get embedding directly via API
def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        st.error(f"OpenAI API error: {response.text}")
        return None
    
    result = response.json()
    return result["data"][0]["embedding"]

# Function to query Pinecone directly via API
def query_pinecone(vector, top_k=3):
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }
    
    data = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    # FIXED URL FORMAT - removed the extra https://
    pinecone_url = f"https://{PINECONE_INDEX_NAME}-{PINECONE_ENV}.svc.{PINECONE_ENV}.pinecone.io/query"
    
    st.write(f"Debug - Using Pinecone URL: {pinecone_url}")
    
    response = requests.post(
        pinecone_url,
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        st.error(f"Pinecone API error: {response.text}")
        return []
    
    result = response.json()
    return result.get("matches", [])

# Function to generate answer via OpenAI API
def generate_answer(question, context):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant for Calbright College, a fully online California community college. Answer questions based ONLY on the provided context."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            }
        ],
        "temperature": 0.2
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        st.error(f"OpenAI API error: {response.text}")
        return "Sorry, I couldn't generate an answer."
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Add option to manually enter Pinecone URL
pinecone_url_format = st.sidebar.radio(
    "Pinecone URL Format",
    ["Standard", "Serverless", "Custom"],
    index=0
)

if pinecone_url_format == "Custom":
    custom_url = st.sidebar.text_input(
        "Enter full Pinecone URL:",
        value=f"https://{PINECONE_INDEX_NAME}.svc.{PINECONE_ENV}.pinecone.io/query"
    )
    
    def query_pinecone_custom(vector, top_k=3):
        headers = {
            "Content-Type": "application/json",
            "Api-Key": PINECONE_API_KEY
        }
        
        data = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True
        }
        
        st.write(f"Debug - Using custom URL: {custom_url}")
        
        response = requests.post(
            custom_url,
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            st.error(f"Pinecone API error: {response.text}")
            return []
        
        result = response.json()
        return result.get("matches", [])
    
    # Override the query function
    query_pinecone = query_pinecone_custom

# Main interface
st.write("Ask questions about Calbright College's programs, services, and more.")

# Example questions
st.sidebar.header("Example Questions")
examples = [
    "Who provides wellness services at Calbright?",
    "What programs does Calbright offer?",
    "Is Calbright College free?",
    "How long does it take to complete a program?"
]

selected_example = st.sidebar.selectbox("Try an example:", [""] + examples)

# Question input
question = st.text_input("Your question:", value=selected_example)

if st.button("Submit"):
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching for information..."):
                # 1. Generate embedding
                embedding = get_embedding(question)
                if not embedding:
                    st.error("Could not generate embedding.")
                    st.stop()
                
                # 2. Query Pinecone
                matches = query_pinecone(embedding)
                if not matches:
                    st.warning("No relevant information found.")
                    st.stop()
                
                # 3. Format context
                context = ""
                sources = []
                
                for i, match in enumerate(matches):
                    metadata = match.get("metadata", {})
                    text = metadata.get("text", "No text available")
                    url = metadata.get("url", "")
                    title = metadata.get("title", "")
                    
                    context += f"\nDocument {i+1}:\n{text}\n"
                    sources.append((title, url))
                
                # 4. Generate answer
                answer = generate_answer(question, context)
                
                # 5. Display answer
                st.subheader("Answer")
                st.write(answer)
                
                # 6. Display sources
                st.subheader("Sources")
                for i, (title, url) in enumerate(sources):
                    st.write(f"{i+1}. {title}")
                    st.write(f"URL: {url}")
                    st.write("---")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again later.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("This is a demo of a Calbright College information chatbot.")
