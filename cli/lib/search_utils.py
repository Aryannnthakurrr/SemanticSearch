import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> set[str]:
    
    #Reads stopwords from STOPWORDS_PATH and stores them in a set

    stopwords_set = set()
    try:
        # Use the STOPWORDS_PATH constant
        with open(STOPWORDS_PATH, 'r') as f:
            for line in f:
                # .strip() removes whitespace and newline characters
                clean_word = line.strip()
                if clean_word:
                    stopwords_set.add(clean_word)
    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {STOPWORDS_PATH}.")
        print("No stopwords will be used.")
    
    return stopwords_set