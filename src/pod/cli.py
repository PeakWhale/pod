"""Command-line interface for PeakWhale™ Pod."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .engine import cluster_csv, cluster_dataframe, list_examples, load_example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pod",
        description="PeakWhale™ Pod: group rows from any CSV into clear K-means segments with plain-English profiles.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_cluster = sub.add_parser("cluster", help="Cluster a CSV file or built-in example")
    p_cluster.add_argument("csv_path", nargs="?", help="Path to CSV file")
    p_cluster.add_argument("--example", choices=list_examples(), help="Run a built-in synthetic example")
    p_cluster.add_argument("-k", type=int, default=None, help="Number of clusters (auto-selected if omitted)")
    p_cluster.add_argument("--id-column", default=None, help="Column name to use as row identifier")
    p_cluster.add_argument("--features", default=None, help="Comma-separated list of numeric columns to cluster")
    p_cluster.add_argument("--out-csv", default=None, help="Write labeled assignments to CSV")
    p_cluster.add_argument("--json", action="store_true", help="Print full result as JSON")

    p_serve = sub.add_parser("serve", help="Run the local HTTP API and web interface")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        uvicorn.run("pod.api:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.cmd != "cluster":
        parser.print_help()
        return 1

    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()] if args.features else None
    if args.example:
        df = load_example(args.example)
        res = cluster_dataframe(df, k=args.k, id_column=args.id_column, feature_columns=feat_cols)
    elif args.csv_path:
        res = cluster_csv(Path(args.csv_path), k=args.k, id_column=args.id_column, feature_columns=feat_cols)
    else:
        p_cluster.error("Provide a CSV path or --example.")
        return 2

    if args.out_csv:
        import pandas as pd
        rows = []
        for a in res.assignments:
            r = {"row_id": a.row_id, "cluster_id": a.cluster_id, "segment_label": a.segment_label, "distance_to_center": a.distance_to_center, "is_outlier": a.is_outlier}
            r.update(a.values)
            rows.append(r)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    if args.json:
        print(json.dumps(res.model_dump(), indent=2))
    else:
        print(res.headline)
        print()
        for seg in res.segments:
            print(f"  [{seg.cluster_id}] {seg.label} ({seg.count} rows, {seg.share_pct}%)")
            print(f"      Standout: {'; '.join(seg.standout_features)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
