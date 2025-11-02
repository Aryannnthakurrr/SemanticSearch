import argparse
from lib.semantic_search import(
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch
)
from lib.search_utils import(
    load_movies,
    DEFAULT_SEARCH_LIMIT
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Prints embedding model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generates vector embedding for text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="gives number of docs embedded and their dimension")

    embed_query_parser = subparsers.add_parser("embedquery", help="embeds the user query to a vector")
    embed_query_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser("search", help="performs semantic search on data")
    search_parser.add_argument("query", type=str, help="query for semantic search")
    search_parser.add_argument("--limit", type=int, default = DEFAULT_SEARCH_LIMIT, help="number of results to return(default: 5)" )

    args = parser.parse_args()

    match args.command:

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            engine = SemanticSearch()
            documents = load_movies()
            engine.load_or_create_embeddings(documents)
            results = engine.search(args.query, args.limit)
            for i,r in enumerate(results, 1):
                print(f"{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {r['description']}\n")


            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()