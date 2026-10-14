"""Tests for PeakWhale™ Pod clustering engine, CLI, and HTTP API."""
from __future__ import annotations

from fastapi.testclient import TestClient

from pod.api import app
from pod.cli import main as cli_main
from pod.engine import cluster_csv, cluster_dataframe, list_examples, load_example


def test_examples_load_and_auto_cluster():
    examples = list_examples()
    assert "regional-markets" in examples
    assert "customer-accounts" in examples

    df = load_example("regional-markets")
    res = cluster_dataframe(df)
    assert res.auto_selected_k is True
    assert 2 <= res.chosen_k <= 6
    assert res.row_count == 24
    assert sum(s.count for s in res.segments) == 24
    assert len(res.assignments) == 24
    assert res.silhouette > 0.35


def test_explicit_k_and_csv_roundtrip(tmp_path):
    out_csv = tmp_path / "segmented.csv"
    rc = cli_main(["cluster", "--example", "customer-accounts", "-k", "3", "--out-csv", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    content = out_csv.read_text()
    assert "segment_label" in content
    assert "ACCT-101" in content


def test_api_endpoints():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "peakwhale-pod"

    r2 = client.post("/api/cluster", json={"example": "regional-markets", "k": 3})
    assert r2.status_code == 200
    body = r2.json()
    assert body["chosen_k"] == 3
    assert len(body["segments"]) == 3


def test_rejects_bad_inputs():
    client = TestClient(app)
    r = client.post("/api/cluster", json={"csv_text": "name,city\nAlice,Boston\nBob,NYC\n"})
    assert r.status_code == 400
    assert "numeric" in r.json()["detail"].lower()
