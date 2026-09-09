import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.core.logging import setup_logging
from app.evaluation.end_to_end import PipelineEvaluator
from app.ingestion.indexer import IngestionIndexer
from app.services.container import (
    get_bm25_index,
    get_chunk_repository,
    get_embedding_service,
    get_ingestion_pipeline,
    get_qdrant_service,
)

logger = logging.getLogger("evaluation_runner")


def index_sample_documents(docs_dir: Path) -> int:
    indexer = IngestionIndexer(
        ingestion_pipeline=get_ingestion_pipeline(),
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
        bm25_index=get_bm25_index(),
        chunk_repository=get_chunk_repository(),
    )

    doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
    indexed_count = 0

    for file_path in doc_files:
        try:
            doc, chunks = indexer.index_file(file_path)
            indexed_count += 1
            logger.info("Indexed '%s': %d chunks", doc.title, len(chunks))
        except Exception as e:
            logger.error("Failed indexing '%s': %s", file_path, e)

    return indexed_count


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG Pipeline Benchmark Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation_dataset.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/raw",
        help="Path to raw document directory to index before evaluation",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="evaluation_report.md",
        help="Path to output markdown report file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation_scorecard.json",
        help="Path to output JSON scorecard file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max retrieved contexts per query",
    )

    args = parser.parse_args()
    setup_logging()

    # Index sample files if requested
    index_path = Path(args.index_dir)
    if index_path.exists() and index_path.is_dir():
        print(f"\n[1/3] Indexing documents from '{index_path}'...")
        count = index_sample_documents(index_path)
        print(f"      Indexed {count} documents successfully.")
    else:
        print(f"\n[1/3] Skipping document indexing ('{index_path}' not found).")

    # Run pipeline evaluation
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset '{dataset_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/3] Running benchmark evaluation on '{dataset_path}'...")
    evaluator = PipelineEvaluator()
    scorecard = evaluator.evaluate_dataset(
        dataset_path=dataset_path,
        limit_contexts=args.limit,
    )

    # Output report
    report_md = evaluator.generate_markdown_report(scorecard)

    print("\n[3/3] Evaluation Summary:")
    print("=" * 60)
    print(f"Total Queries Evaluated:    {scorecard.total_queries}")
    print(f"Mean Reciprocal Rank (MRR): {scorecard.mean_mrr:.4f}")
    print(f"Hit Rate @ 1:               {scorecard.mean_hit_rate_at_1:.4f}")
    print(f"Hit Rate @ 5:               {scorecard.mean_hit_rate_at_5:.4f}")
    print(f"Precision @ 5:              {scorecard.mean_precision_at_5:.4f}")
    print(f"Recall @ 5:                 {scorecard.mean_recall_at_5:.4f}")
    print(f"nDCG @ 5:                   {scorecard.mean_ndcg_at_5:.4f}")
    print(f"Token Compression Ratio:    {scorecard.mean_compression_ratio:.4f}")
    print(f"Keyword Retention:          {scorecard.mean_keyword_retention:.4f}")
    print(f"Faithfulness Score:         {scorecard.mean_faithfulness_score:.4f}")
    print(f"Grounding Pass Rate:        {scorecard.grounding_pass_rate:.4f}")
    print(f"Answer Token F1:            {scorecard.mean_answer_f1:.4f}")
    print(f"Citation Validity Rate:     {scorecard.mean_citation_validity_rate:.4f}")
    print("=" * 60)

    if args.output_report:
        report_file = Path(args.output_report)
        report_file.write_text(report_md, encoding="utf-8")
        print(f"Markdown report saved to: {report_file.resolve()}")

    if args.output_json:
        json_file = Path(args.output_json)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(asdict(scorecard), f, indent=2)
        print(f"JSON scorecard saved to:  {json_file.resolve()}")


if __name__ == "__main__":
    main()
