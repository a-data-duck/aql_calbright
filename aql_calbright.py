import streamlit as st
import openai
import pinecone

# Simple app with minimal dependencies
st.title("Calbright College Q&A")

# API keys
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"] 
    pinecone_index = st.secrets.get("PINECONE_INDEX_NAME", "calbright-docs")
except Exception as e:
    st.error(f"API key error: {e}")
    st.stop()

# Initialize Pinecone (older API style)
try:
    pinecone.init(api_key=pinecone_api_key, environment="us-east-1")
    index = pinecone.Index(pinecone_index)
    st.success("Connected to services")
except Exception as e:
    st.error(f"Pinecone error: {e}")
    st.info("Check your API keys and index name")
    st.stop()

# Question input
question = st.text_input("What would you like to know about Calbright College?")

# Process question when submitted
if st.button("Submit") and question:
    try:
        # 1. Get embedding for the query
        with st.spinner("Processing..."):
            embedding_response = openai.Embedding.create(
                input=question,
                model="text-embedding-ada-002"
            )
            embedding = embedding_response["data"][0]["embedding"]
        
        # 2. Query Pinecone
        with st.spinner("Searching knowledge base..."):
            query_response = index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )
            
        # 3. Format results into context
        context = ""
        sources = []
        
        for i, match in enumerate(query_response["matches"]):
            metadata = match["metadata"]
            text = metadata.get("text", "")
            url = metadata.get("url", "")
            title = metadata.get("title", "")
            context += f"\n## Document {i+1}:\n{text}\n"
            sources.append((title, url))
        
        # 4. Generate answer with OpenAI
        with st.spinner("Generating answer..."):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Calbright College. Answer questions about the college based ONLY on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
                ],
                temperature=0.2
            )
            
            answer = completion["choices"][0]["message"]["content"]
        
        # 5. Display answer and sources
        st.subheader("Answer")
        st.write(answer)
        
        st.subheader("Sources")
        for i, (title, url) in enumerate(sources):
            st.write(f"{i+1}. {title}")
            st.write(f"   URL: {url}")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Please try again with a different question.")
