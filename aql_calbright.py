import streamlit as st
import json
import requests

# Hide sidebar
st.set_page_config(page_title="Calbright College Q&A", page_icon="🎓", initial_sidebar_state="collapsed")

# Custom CSS to hide the sidebar completely
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    .big-font {
        font-size: 24px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .small-italic {
        font-size: 14px;
        font-style: italic;
        color: #666;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Set title
st.title("Calbright College Q&A")

# Configure API keys (now hidden from sidebar)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_URL = "https://calbright-docs-h3y3rrq.svc.aped-4627-b74a.pinecone.io"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.error("Missing API keys. Please contact the administrator.")
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
def query_pinecone(vector, base_url, top_k=3):
    # Make sure the URL ends with /query
    if not base_url.endswith("/query"):
        query_url = f"{base_url}/query"
    else:
        query_url = base_url
    
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }
    
    data = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    try:
        response = requests.post(
            query_url,
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code != 200:
            st.error(f"Pinecone API error: {response.text}")
            return []
        
        # Check if response is empty
        if not response.text:
            st.error("Pinecone returned an empty response")
            return []
            
        # Try to parse the JSON
        try:
            result = response.json()
            return result.get("matches", [])
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return []

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

# Main interface
st.write("Ask questions about Calbright College's programs, services, and more.")

# Example questions moved to main interface as buttons
st.write("Try an example:")
col1, col2 = st.columns(2)
with col1:
    if st.button("Who provides wellness services?"):
        question = "Who provides wellness services at Calbright?"
    elif st.button("Is Calbright College free?"):
        question = "Is Calbright College free?"
with col2:
    if st.button("What programs are offered?"):
        question = "What programs does Calbright offer?"
    elif st.button("How long to complete a program?"):
        question = "How long does it take to complete a program?"

# Question input
question = st.text_input("Or type your own question:")

if st.button("Submit") or "question" in locals():
    if not question:
        st.warning("Please enter a question or select an example.")
    else:
        try:
            with st.spinner("Searching for information..."):
                # 1. Generate embedding
                embedding = get_embedding(question)
                if not embedding:
                    st.error("Could not generate embedding.")
                    st.stop()
                
                # 2. Query Pinecone
                matches = query_pinecone(embedding, PINECONE_URL)
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
                
                # 5. Display answer in larger font (without a heading)
                st.markdown(f'<div class="big-font">{answer}</div>', unsafe_allow_html=True)
                
                # 6. Display sources with smaller, italicized heading
                st.markdown('<div class="small-italic">sources</div>', unsafe_allow_html=True)
                for i, (title, url) in enumerate(sources):
                    st.write(f"{i+1}. {title}")
                    st.write(f"URL: {url}")
                    st.write("---")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again later.")


'''
import streamlit as st
import json
import requests

# Set title
st.title("Calbright College Q&A")

# Configure API keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")

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
def query_pinecone(vector, base_url, top_k=3):
    # Make sure the URL ends with /query
    if not base_url.endswith("/query"):
        query_url = f"{base_url}/query"
    else:
        query_url = base_url
    
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }
    
    data = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    # Removed debug info
    
    try:
        response = requests.post(
            query_url,
            headers=headers,
            json=data,
            timeout=10  # Add timeout
        )
        
        # Removed status code output
        
        if response.status_code != 200:
            st.error(f"Pinecone API error: {response.text}")
            return []
        
        # Check if response is empty
        if not response.text:
            st.error("Pinecone returned an empty response")
            return []
            
        # Try to parse the JSON
        try:
            result = response.json()
            return result.get("matches", [])
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {e}")
            st.write(f"Raw response: {response.text[:500]}...")  # Show first 500 chars
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return []

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

# Pinecone configuration
st.sidebar.header("Pinecone Configuration")
pinecone_url = st.sidebar.text_input(
    "Pinecone URL:",
    value="https://calbright-docs-h3y3rrq.svc.aped-4627-b74a.pinecone.io"
)

# Test Pinecone connection button
if st.sidebar.button("Test Pinecone Connection"):
    test_headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }
    
    # Try adding /query if not present
    if not pinecone_url.endswith("/query"):
        test_url = f"{pinecone_url}/query"
    else:
        test_url = pinecone_url
    
    try:
        # Just a simple describe request to test connection
        test_vector = [0.0] * 1536  # Sample embedding with all zeros
        test_data = {
            "vector": test_vector,
            "top_k": 1
        }
        
        test_response = requests.post(
            test_url,
            headers=test_headers,
            json=test_data,
            timeout=10
        )
        
        st.sidebar.write(f"Status: {test_response.status_code}")
        
        if test_response.status_code == 200:
            st.sidebar.success("Successfully connected to Pinecone!")
        else:
            st.sidebar.error(f"Failed to connect: {test_response.text}")
        
    except Exception as e:
        st.sidebar.error(f"Connection error: {str(e)}")

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
                matches = query_pinecone(embedding, pinecone_url)
                if not matches:
                    st.warning("No relevant information found. Please check the Pinecone connection.")
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
'''
