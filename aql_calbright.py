import streamlit as st
import openai
import pinecone

# Set page configuration
st.set_page_config(page_title="Calbright College Chatbot", page_icon="🎓")
st.title("🎓 Calbright College Chatbot")

# Get API keys from Streamlit secrets
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")
except Exception as e:
    st.error(f"Error with API keys: {e}")
    st.stop()

# Initialize Pinecone
try:
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")
    index = pinecone.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# Get embedding function
def get_embedding(text):
    response = openai.Embedding.create(
        input=text, 
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Search function
def search_docs(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]

# Create answer
def generate_answer(query, context):
    prompt = f"""You are a helpful assistant for Calbright College, a fully online California community college.
Use only the following context to answer the question. If you don't know the answer based on the context, say so.

Context:
{context}

Question: {query}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    
    return response["choices"][0]["message"]["content"]

# Format context
def format_context(matches):
    context = ""
    sources = []
    
    for i, match in enumerate(matches):
        metadata = match["metadata"]
        score = match["score"]
        text = metadata.get("text", "No text available")
        url = metadata.get("url", "")
        title = metadata.get("title", "")
        
        context += f"\nDocument {i+1}:\n{text}\n"
        sources.append((title, url, score))
    
    return context, sources

# Main app function
def main():
    st.write("Ask questions about Calbright College's programs, services, and resources.")
    
    query = st.text_input("Your question:")
    
    if st.button("Ask") and query:
        with st.spinner("Searching for information..."):
            try:
                # Search for documents
                matches = search_docs(query)
                
                if not matches:
                    st.warning("No relevant information found.")
                    return
                
                # Format the context and extract sources
                context, sources = format_context(matches)
                
                # Generate the answer
                answer = generate_answer(query, context)
                
                # Display the answer
                st.subheader("Answer:")
                st.write(answer)
                
                # Display sources
                st.subheader("Sources:")
                for i, (title, url, score) in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {title}")
                    st.markdown(f"**URL:** {url}")
                    st.markdown(f"**Relevance:** {score:.2f}")
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
if __name__ == "__main__":
    main()
