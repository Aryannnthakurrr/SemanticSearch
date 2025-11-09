import time
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "cli"))

from lib.keyword_search import InvertedIndex, search_command
from lib.semantic_search import SemanticSearch


class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            "tfidf": {},
            "bm25": {},
            "semantic": {}
        }
        # Load documents once for semantic search
        self.documents = self.load_documents()
    
    def load_documents(self):
        """Load movie dataset"""
        data_path = Path(__file__).parent.parent / "data" / "movies.json"
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data.get("movies", [])
        
    def benchmark_tfidf(self, query, runs=5):
        """Benchmark TF-IDF search"""
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            results = search_command(query)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return {
            "avg_latency_ms": np.mean(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "std_dev_ms": np.std(times),
            "result_count": len(results)
        }
    
    def benchmark_bm25(self, query, runs=5):
        """Benchmark BM25 search"""
        idx = InvertedIndex()
        idx.load()
        
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            results = idx.bm25_search(query)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            "avg_latency_ms": np.mean(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "std_dev_ms": np.std(times),
            "result_count": len(results)
        }
    
    def benchmark_semantic(self, query, runs=3):
        """Benchmark Semantic search (slower, fewer runs)"""
        semantic = SemanticSearch()
        semantic.load_or_create_embeddings(self.documents)  # Load embeddings first
        times = []
        
        for _ in range(runs):
            start = time.perf_counter()
            results = semantic.search(query)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            "avg_latency_ms": np.mean(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "std_dev_ms": np.std(times),
            "result_count": len(results)
        }
    
    def benchmark_index_build(self):
        """Benchmark index building"""
        idx = InvertedIndex()
        
        start = time.perf_counter()
        idx.build()
        end = time.perf_counter()
        
        build_time = (end - start) * 1000
        index_size_mb = sum(sys.getsizeof(v) for v in idx.index.values()) / (1024 * 1024)
        
        return {
            "build_time_ms": build_time,
            "index_size_mb": index_size_mb,
            "doc_count": len(idx.docmap),
            "unique_terms": len(idx.index)
        }
    
    def benchmark_embedding_generation(self):
        """Benchmark embedding generation"""
        semantic = SemanticSearch()
        test_texts = [
            "Space adventure movie",
            "Animated family film",
            "Action thriller",
            "Comedy romance",
            "Science fiction drama"
        ]
        
        times = []
        for text in test_texts:
            start = time.perf_counter()
            embedding = semantic.generate_embedding(text)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            "avg_latency_ms": np.mean(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "embedding_dimension": embedding.shape[0]
        }
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("=" * 60)
        print("SEMANTIC SEARCH ALGORITHM BENCHMARK")
        print("=" * 60)
        
        # Index building
        print("\n[1/4] Benchmarking Index Building...")
        index_results = self.benchmark_index_build()
        print(f"  Build Time: {index_results['build_time_ms']:.2f}ms")
        print(f"  Index Size: {index_results['index_size_mb']:.2f}MB")
        print(f"  Documents: {index_results['doc_count']}")
        print(f"  Unique Terms: {index_results['unique_terms']}")
        self.results["index_build"] = index_results
        
        # Embedding generation
        print("\n[2/4] Benchmarking Embedding Generation...")
        embed_results = self.benchmark_embedding_generation()
        print(f"  Avg Latency: {embed_results['avg_latency_ms']:.2f}ms")
        print(f"  Dimensions: {embed_results['embedding_dimension']}")
        self.results["embedding_generation"] = embed_results
        
        # Search queries
        queries = [
            "space adventure",
            "animated family",
            "action thriller"
        ]
        
        print("\n[3/4] Benchmarking Search Queries...")
        print("\nQuery: 'space adventure'")
        
        tfidf_results = self.benchmark_tfidf(queries[0])
        print(f"  TF-IDF:    {tfidf_results['avg_latency_ms']:.2f}ms (±{tfidf_results['std_dev_ms']:.2f}ms)")
        self.results["tfidf"]["space_adventure"] = tfidf_results
        
        bm25_results = self.benchmark_bm25(queries[0])
        print(f"  BM25:      {bm25_results['avg_latency_ms']:.2f}ms (±{bm25_results['std_dev_ms']:.2f}ms)")
        self.results["bm25"]["space_adventure"] = bm25_results
        
        semantic_results = self.benchmark_semantic(queries[0], runs=3)
        print(f"  Semantic:  {semantic_results['avg_latency_ms']:.2f}ms (±{semantic_results['std_dev_ms']:.2f}ms)")
        self.results["semantic"]["space_adventure"] = semantic_results
        
        print("\nQuery: 'animated family'")
        
        tfidf_results = self.benchmark_tfidf(queries[1])
        print(f"  TF-IDF:    {tfidf_results['avg_latency_ms']:.2f}ms (±{tfidf_results['std_dev_ms']:.2f}ms)")
        self.results["tfidf"]["animated_family"] = tfidf_results
        
        bm25_results = self.benchmark_bm25(queries[1])
        print(f"  BM25:      {bm25_results['avg_latency_ms']:.2f}ms (±{bm25_results['std_dev_ms']:.2f}ms)")
        self.results["bm25"]["animated_family"] = bm25_results
        
        semantic_results = self.benchmark_semantic(queries[1], runs=3)
        print(f"  Semantic:  {semantic_results['avg_latency_ms']:.2f}ms (±{semantic_results['std_dev_ms']:.2f}ms)")
        self.results["semantic"]["animated_family"] = semantic_results
        
        print("\nQuery: 'action thriller'")
        
        tfidf_results = self.benchmark_tfidf(queries[2])
        print(f"  TF-IDF:    {tfidf_results['avg_latency_ms']:.2f}ms (±{tfidf_results['std_dev_ms']:.2f}ms)")
        self.results["tfidf"]["action_thriller"] = tfidf_results
        
        bm25_results = self.benchmark_bm25(queries[2])
        print(f"  BM25:      {bm25_results['avg_latency_ms']:.2f}ms (±{bm25_results['std_dev_ms']:.2f}ms)")
        self.results["bm25"]["action_thriller"] = bm25_results
        
        semantic_results = self.benchmark_semantic(queries[2], runs=3)
        print(f"  Semantic:  {semantic_results['avg_latency_ms']:.2f}ms (±{semantic_results['std_dev_ms']:.2f}ms)")
        self.results["semantic"]["action_thriller"] = semantic_results
        
        print("\n[4/4] Generating Summary Statistics...")
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print summary analysis"""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Extract average latencies for comparison
        tfidf_avg = np.mean([v['avg_latency_ms'] for v in self.results['tfidf'].values()])
        bm25_avg = np.mean([v['avg_latency_ms'] for v in self.results['bm25'].values()])
        semantic_avg = np.mean([v['avg_latency_ms'] for v in self.results['semantic'].values()])
        
        print(f"\nAverage Query Latency:")
        print(f"  TF-IDF:   {tfidf_avg:.2f}ms")
        print(f"  BM25:     {bm25_avg:.2f}ms")
        print(f"  Semantic: {semantic_avg:.2f}ms")
        
        # Speed comparison
        print(f"\nSpeed Comparison (relative to fastest):")
        min_latency = min(tfidf_avg, bm25_avg, semantic_avg)
        print(f"  TF-IDF:   {(tfidf_avg / min_latency):.2f}x")
        print(f"  BM25:     {(bm25_avg / min_latency):.2f}x")
        print(f"  Semantic: {(semantic_avg / min_latency):.2f}x")
        
        print(f"\nKey Observations:")
        if tfidf_avg < bm25_avg:
            print(f"  • TF-IDF is {(bm25_avg/tfidf_avg):.1f}x faster than BM25")
        else:
            print(f"  • BM25 is {(tfidf_avg/bm25_avg):.1f}x faster than TF-IDF")
        
        if semantic_avg > bm25_avg:
            print(f"  • Semantic search is {(semantic_avg/bm25_avg):.1f}x slower than BM25")
        
        print(f"  • Embedding generation: ~{self.results['embedding_generation']['avg_latency_ms']:.0f}ms per query")
        print(f"  • Index build time: {self.results['index_build']['build_time_ms']:.0f}ms")
        print(f"  • Semantic vectors: {self.results['embedding_generation']['embedding_dimension']} dimensions")


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results to JSON
    import json
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")
