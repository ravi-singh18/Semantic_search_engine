import streamlit as st
import chromadb     
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer


# Load your ChromaDB collection
client = chromadb.PersistentClient(path=r"C:\Users\errav\genai\search_engine\chromadb4")
collection = client.get_collection(name="search_engine_project_3")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

st.header("Movie Subtitle Search Engine")
search_query = st.text_input("Enter a dialogue to search....")

def clean_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

    # Tokenize the text
    tokens = word_tokenize(clean_text)

    # Remove stopwords and lemmatize tokens
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens back into a string
    clean_text = ' '.join(clean_tokens)

    return clean_text.strip()

if st.button("Search"):
    st.subheader("Relevant Subtitle Files")
    search_query = clean_text(search_query)
    query_embed = model.encode([search_query])[0].tolist()

    results = collection.query(
        query_embeddings=query_embed,
        n_results=5,
        include=['documents']
    )
    
    #Process and display documents
    for query_documents in results['documents']:
        for document in query_documents:
            st.markdown(f"[{document}]")
    documents = results['documents']

# Iterate over the documents and print each document
    for i, query_documents in enumerate(documents):
        for j, document in enumerate(query_documents):
            st.markdown(f"Document {i+1}, Item {j+1}: {document}")
