from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import(
    CACHE_DIR,
    load_movies,
    DEFAULT_SEARCH_LIMIT
)
import os





class SemanticSearch:

    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None 
        self.document_map = {}


    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("Error: Text string is empty or only has whitespace")   
        To_encode_list = []
        To_encode_list.append(text)
        embedding = self.model.encode(To_encode_list)
        return embedding[0]

    def build_embeddings(self, documents:list[dict]):
        #populating documents and document map for vector to movie data lookup later
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]]= doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(f"{CACHE_DIR}/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        #populating documents and document map for vector to movie data lookup later
        self.documents = documents
        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]]= doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        #building path to movie_embeddings.npy inside cache
        path = f"{CACHE_DIR}/movie_embeddings.npy"
        #checking if the cache already exists
        if os.path.exists(path):
            #if exists we just load it to embeddings
            self.embeddings = np.load(path)
            #a check for whether the cache is caught up with embeddings
            if len(self.embeddings) == len(documents):
                #if it is no need to rebuild
                return self.embeddings
        #if it isnt then have to rebuild
        return self.build_embeddings(documents)

    def search(self, query, limit = DEFAULT_SEARCH_LIMIT):
        #checking if embeddings have been generated or not
        if self.embeddings is None or self.embeddings.size==0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        #converting query to vector embedding
        query_vec = self.generate_embedding(query)

        #can do this since vector at each index maps to movie at the same index to wich vector belongs
        pairs = []
        for i, doc_vec in enumerate(self.embeddings):
            #movie is a dict at the same index in self.documents(the list of dictionaries)
            document = self.documents[i]
            score = cosine_similarity(query_vec, doc_vec)
            #building result tuples for each result
            pairs.append((score, document))
        #sorting all results in reverse order
        pairs.sort(key = lambda x: x[0], reverse = True)
        top_results = pairs[:limit]

        return[
            {
                "score": score,
                "title": document["title"],
                "description": document["description"],
            }
            for score, document in top_results
        ]

        
        

        



        

def verify_model():
    verification_class = SemanticSearch()
    MODEL = verification_class.model
    MAX_LENGTH = MODEL.max_seq_length
    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MAX_LENGTH}")

def embed_text(text):
    embedding_class = SemanticSearch()
    embedding = embedding_class.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape[0]}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


