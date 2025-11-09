# SemanticSearch

A comparative analysis of search algorithms: BM25, TF-IDF, and semantic embeddings.

**[View Full Analysis →](https://aryanthakur.vercel.app/projects/search-algorithms.html)**

## Implementation

- **BM25** - Probabilistic ranking with parameter tuning (K1=1.5, b=0.75)
- **TF-IDF** - Term frequency-inverse document frequency scoring
- **Semantic Search** - Dense embeddings via SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
- **Inverted Index** - Fast term lookup with document length normalization

## Performance Analysis

Benchmarked on 5,000 movie documents with 85,577 unique terms.

### Query Latency

| Algorithm | Avg Latency | Speed vs Fastest | Use Case |
|-----------|-------------|------------------|----------|
| **Semantic** | 68ms | 1.0x (baseline) | Understanding user intent, semantic similarity |
| **TF-IDF** | 377ms | 5.5x slower | Quick exact-match keyword search |
| **BM25** | 954ms | 14.0x slower | Relevance ranking with length normalization |

### Key Findings

**Speed & Efficiency:**
- Semantic search is **14x faster** than BM25 and **5.5x faster** than TF-IDF for queries
- TF-IDF is **2.5x faster** than BM25 due to simpler scoring
- Index build time: 23.7 seconds for 5,000 documents
- Embedding generation: ~79ms per query

**Memory & Storage:**
- Inverted index: 89.25 MB for 5,000 documents
- Semantic embeddings: 384 dimensions per document
- Unique vocabulary: 85,577 terms

**Trade-offs:**
- **Semantic**: Fast queries, semantic understanding, but requires upfront embedding generation
- **BM25**: Best relevance ranking, handles document length well, but slower queries
- **TF-IDF**: Simple and interpretable, good baseline, but no length normalization

### Algorithm Comparison

**When to use each:**

| Scenario | Best Algorithm | Reason |
|----------|----------------|--------|
| User queries with typos/synonyms | Semantic | Understands meaning, not just keywords |
| Exact keyword matching | TF-IDF | Fast and straightforward |
| Ranking long vs short documents | BM25 | Length normalization prevents bias |
| Low-latency requirements | Semantic | 14x faster than BM25 after initial setup |
| Interpretable results | TF-IDF/BM25 | Clear term-based scoring |

## Future Work

- **Hybrid Search**: Combine keyword precision with semantic understanding
- **Document Chunking**: Implement RAG-optimized segmentation strategies  
- **Reranking**: Add cross-encoder for two-stage retrieval
- **Query Expansion**: Improve recall with semantic query augmentation

## Installation

```bash
git clone https://github.com/Aryannnthakurrr/SemanticSearch.git
cd SemanticSearch
uv install
```

## Usage

```bash
# Build index
uv run cli/keyword_search_cli.py build

# Search with different algorithms
uv run cli/keyword_search_cli.py search "space adventure"
uv run cli/keyword_search_cli.py bm25search "space adventure"
uv run cli/semantic_search_cli.py search "space adventure"

# Run benchmarks
uv run tests/benchmark.py
```

## Technical Details

### BM25 Formula
```
BM25(D, Q) = Σ IDF(qi) × (TF(qi, D) × (K1 + 1)) / (TF(qi, D) + K1 × (1 - b + b × |D| / avgdl))
```
Where: K1=1.5 (saturation), b=0.75 (length norm), avgdl=average document length

### Semantic Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Max sequence length: 256 tokens
- Similarity: Cosine similarity

## License

MIT