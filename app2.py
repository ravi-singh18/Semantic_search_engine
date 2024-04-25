from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import chromadb

app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the ChromaDB client and collection
client = chromadb.PersistentClient(path=r"C:\Users\errav\genai\search_engine\chromadb4")

collection = client.get_collection(name="search_engine_project_3")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        
        # Preprocess and encode the query
        cleaned_query = clean_text(query)
        query_embedding = model.encode([cleaned_query])

        # Perform the query
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=10,
            include=['documents', 'metadatas']  # Assuming this correctly fetches documents and metadata
        )

        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])

        # Combine documents with their metadata
        processed_documents = []
        for doc, meta in zip(documents[0], metadatas[0]):  # Assuming each is a list and they're parallel
            processed_documents.append({'document': doc, 'metadata': meta})
        
        return render_template('index.html', query=query, documents=processed_documents)
    return render_template('index.html')


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

if __name__ == '__main__':
    app.run(debug=True)