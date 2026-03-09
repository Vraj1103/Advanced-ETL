#!/usr/bin/env python3
"""
Export all chunks (vector DB) and all tables (PostgreSQL) to files for evaluation.

Use this to verify:
- Data is stored correctly (format, metadata, page numbers)
- Chunk boundaries and table structure are suitable for accurate answer retrieval
- Gaps or improvements (e.g. missing page_number, empty tables, duplicate content)

Usage:
  python export_for_evaluation.py [--output-dir DIR] [--namespace NS]

Output files (in --output-dir, default: ./export_evaluation):
  - chunks_export.json   : All vector chunks (content, file_name, namespace, page_number, etc.)
  - tables_export.json   : All PostgreSQL tables (table_id, headers, data, metadata)
  - facts_export.json    : All PostgreSQL facts (for fact-based retrieval evaluation)
  - export_summary.txt   : Counts and short summary for quick review
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Project imports
import config
from vector_service import VectorStorageService
from structured_storage_service import StructuredStorageService


DEFAULT_OUTPUT_DIR = "export_evaluation"
BATCH_SIZE = 1000


def sanitize_for_json(obj):
    """Convert non-JSON-serializable values (e.g. datetime, bytes)."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return "<bytes>"
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


async def export_chunks(vector_service: VectorStorageService, output_path: Path) -> int:
    """Export all chunks from Azure Cognitive Search. Returns count."""
    await vector_service.initialize()
    chunks = []
    skip = 0
    while True:
        results = vector_service.search_client.search(
            search_text="*",
            top=BATCH_SIZE,
            skip=skip,
            include_total_count=True,
        )
        n = 0
        async for doc in results:
            n += 1
            page_number_raw = doc.get("page_number") or "[]"
            try:
                page_numbers = json.loads(page_number_raw) if isinstance(page_number_raw, str) else page_number_raw
                if not isinstance(page_numbers, list):
                    page_numbers = [page_numbers] if page_numbers is not None else []
                page_numbers = [int(p) for p in page_numbers if p is not None]
            except (json.JSONDecodeError, TypeError, ValueError):
                page_numbers = []
            bounding_box = doc.get("bounding_box", "{}")
            try:
                bounding_box = json.loads(bounding_box) if isinstance(bounding_box, str) else bounding_box
            except (json.JSONDecodeError, TypeError):
                bounding_box = {}
            chunk = {
                "id": doc.get("id"),
                "file_id": doc.get("file_id"),
                "file_name": doc.get("file_name"),
                "namespace": doc.get("namespace"),
                "content": doc.get("content"),
                "page_number": page_numbers,
                "bounding_box": bounding_box,
                "page_info": doc.get("page_info"),
            }
            chunks.append(sanitize_for_json(chunk))
        if n == 0:
            break
        skip += n
        if n < BATCH_SIZE:
            break
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"exported_at": datetime.utcnow().isoformat(), "total_chunks": len(chunks), "chunks": chunks},
            f,
            indent=2,
            ensure_ascii=False,
        )
    return len(chunks)


async def export_tables(storage: StructuredStorageService, output_path: Path) -> int:
    """Export all tables from PostgreSQL. Returns count."""
    await storage.initialize()
    async with storage.connection_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT table_id, file_id, file_name, namespace, headers, data, metadata, created_at
            FROM tables
            ORDER BY namespace, table_id
            """
        )
    tables = []
    for row in rows:
        r = dict(row)
        for key in ("headers", "data", "metadata"):
            if isinstance(r.get(key), str):
                try:
                    r[key] = json.loads(r[key])
                except json.JSONDecodeError:
                    pass
        tables.append(sanitize_for_json(r))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"exported_at": datetime.utcnow().isoformat(), "total_tables": len(tables), "tables": tables},
            f,
            indent=2,
            ensure_ascii=False,
        )
    return len(tables)


async def export_facts(storage: StructuredStorageService, output_path: Path) -> int:
    """Export all facts from PostgreSQL. Returns count."""
    await storage.initialize()
    async with storage.connection_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, file_id, file_name, namespace, entity_type, value, page, source_quote, confidence, created_at
            FROM facts
            ORDER BY namespace, created_at
            """
        )
    facts = [sanitize_for_json(dict(row)) for row in rows]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"exported_at": datetime.utcnow().isoformat(), "total_facts": len(facts), "facts": facts},
            f,
            indent=2,
            ensure_ascii=False,
        )
    return len(facts)


async def run(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    namespace: str = None,
    skip_chunks: bool = False,
    skip_tables: bool = False,
    skip_facts: bool = False,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vector_service = VectorStorageService()
    storage = StructuredStorageService()

    summary_lines = [
        f"Export for evaluation — {datetime.utcnow().isoformat()}Z",
        f"Output directory: {out.resolve()}",
        f"Namespace filter: {namespace or '(all)'}",
        "",
    ]
    chunk_count = 0
    table_count = 0
    fact_count = 0

    try:
        if not skip_chunks:
            print("Exporting chunks from vector DB...")
            chunk_count = await export_chunks(vector_service, out / "chunks_export.json")
            summary_lines.append(f"Chunks exported: {chunk_count} -> chunks_export.json")
            print(f"  -> {chunk_count} chunks written to chunks_export.json")
        else:
            summary_lines.append("Chunks: skipped")

        if not skip_tables:
            print("Exporting tables from PostgreSQL...")
            table_count = await export_tables(storage, out / "tables_export.json")
            summary_lines.append(f"Tables exported: {table_count} -> tables_export.json")
            print(f"  -> {table_count} tables written to tables_export.json")
        else:
            summary_lines.append("Tables: skipped")

        if not skip_facts:
            print("Exporting facts from PostgreSQL...")
            fact_count = await export_facts(storage, out / "facts_export.json")
            summary_lines.append(f"Facts exported: {fact_count} -> facts_export.json")
            print(f"  -> {fact_count} facts written to facts_export.json")
        else:
            summary_lines.append("Facts: skipped")

        summary_lines.extend([
            "",
            "--- What to check ---",
            "1. Chunks: page_number present and correct? content continuous? no duplicate paragraphs?",
            "2. Tables: metadata.page_number set? headers and data aligned? numeric columns parsed?",
            "3. Facts: value as string? source_quote and page present? enough coverage for key numbers?",
            "4. Readiness: enough chunks for semantic retrieval? tables usable for compare_data/query_table?",
        ])
        with open(out / "export_summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print("\nSummary written to export_summary.txt")

    finally:
        await vector_service.close()
        await storage.close()

    return {"chunks": chunk_count, "tables": table_count, "facts": fact_count}


def main():
    parser = argparse.ArgumentParser(
        description="Export chunks and PostgreSQL tables/facts for storage and retrieval evaluation."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional: only export data for this namespace (note: current export is all; filter when evaluating)",
    )
    parser.add_argument("--skip-chunks", action="store_true", help="Do not export vector chunks")
    parser.add_argument("--skip-tables", action="store_true", help="Do not export PostgreSQL tables")
    parser.add_argument("--skip-facts", action="store_true", help="Do not export PostgreSQL facts")
    args = parser.parse_args()

    counts = asyncio.run(
        run(
            output_dir=args.output_dir,
            namespace=args.namespace,
            skip_chunks=args.skip_chunks,
            skip_tables=args.skip_tables,
            skip_facts=args.skip_facts,
        )
    )
    print(f"\nDone. Chunks: {counts['chunks']}, Tables: {counts['tables']}, Facts: {counts['facts']}")


if __name__ == "__main__":
    main()
