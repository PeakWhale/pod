"""MCP server for PeakWhale™ Pod."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from .engine import cluster_csv, cluster_dataframe, list_examples, load_example

mcp = FastMCP("peakwhale-pod")


@mcp.tool()
def pod_list_examples() -> list[str]:
    """List built-in synthetic datasets available in PeakWhale™ Pod."""
    return list_examples()


@mcp.tool()
def pod_cluster_example(example: str = "regional-markets", k: int | None = None) -> dict:
    """Cluster a built-in example dataset and return plain-English segment profiles."""
    df = load_example(example)
    return cluster_dataframe(df, k=k).model_dump()


@mcp.tool()
def pod_cluster_csv(csv_text: str, k: int | None = None, id_column: str | None = None) -> dict:
    """Cluster raw CSV text into K-means segments with plain-English labels and outlier flags."""
    return cluster_csv(csv_text, k=k, id_column=id_column).model_dump()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
