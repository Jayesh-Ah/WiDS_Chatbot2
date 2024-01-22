import streamlit as st
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to convert text into sentence chunks
def chunk_text(text):
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]
    return chunks

# Function to generate embeddings for text chunks
def generate_embeddings(chunks):
    chunk_embeddings = []
    for chunk in chunks:
        chunk_embeddings.append(np.mean([token.vector for token in nlp(chunk)], axis=0))
    return np.array(chunk_embeddings)

# Function to retrieve chunks based on similarity score
def retrieve_chunks(query_embedding, embeddings_database, chunks_database, top_k=3):
    similarities = cosine_similarity([query_embedding], embeddings_database)[0]
    indexes = np.argsort(similarities)[::-1][:top_k]
    similar_chunks = [chunks_database[i] for i in indexes]
    return similar_chunks

# Main Streamlit app
st.title("PDF Chatbot with Embeddings")

uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if st.button("Process"):
    if uploaded_files:
        # Process PDF files and build vector database
        chunks_database = []
        embeddings_database = []

        for file in uploaded_files:
            pdf_text = file.read().decode("utf-8")  # Read file content as string
            pdf_chunks = chunk_text(pdf_text)
            chunks_database.extend(pdf_chunks)

            # Generate embeddings for the chunks
            embeddings_database.extend(generate_embeddings(pdf_chunks))

        embeddings_database = np.array(embeddings_database)

        # User query
        user_query = st.text_input("Enter your query:")

        if user_query:
            # Process user query and retrieve relevant chunks
            query_chunks = chunk_text(user_query)
            query_embedding = generate_embeddings(query_chunks)[0]

            # Retrieve similar chunks
            similar_chunks = retrieve_chunks(query_embedding, embeddings_database, chunks_database)

            # Display results
            st.write("### Results:")
            for chunk in similar_chunks:
                st.write(chunk)
