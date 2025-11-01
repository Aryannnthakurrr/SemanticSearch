from sentence_transformers import SentenceTransformer
import numpy as np





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
