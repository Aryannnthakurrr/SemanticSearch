from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)
from .search_utils import load_stopwords
import string
from nltk.stem import PorterStemmer

STOPWORDS = load_stopwords()
stemmer = PorterStemmer()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    
    preprocessed_query = preprocess_text(query)
    # removing stopwords from query
    query_tokens = [token for token in preprocessed_query.split() if token not in STOPWORDS]
    # stemming the query tokens
    query_tokens = [stemmer.stem(token) for token in query_tokens]  
    
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        # removing stopwords from title
        title_tokens = [token for token in preprocessed_title.split() if token not in STOPWORDS]
        # stemming the title tokens
        title_tokens = [stemmer.stem(token) for token in title_tokens]        
        # token matching using filtered tokens without stopwords from both query and title
        if any(query_token in title_token for query_token in query_tokens for title_token in title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> str:
    
    # making case insensitive
    text = text.lower()
    
    # removing punctuation
    #No replacement only deleting all the punctuation characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text