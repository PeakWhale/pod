"""Core K-means clustering engine with automatic k selection, plain-English labels, and outlier detection."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class CandidateK(BaseModel):
    k: int
    inertia: float
    silhouette: float


class SegmentProfile(BaseModel):
    cluster_id: int
    label: str
    count: int
    share_pct: float
    feature_means: dict[str, float]
    standout_features: list[str]


class RowAssignment(BaseModel):
    row_id: str
    cluster_id: int
    segment_label: str
    distance_to_center: float
    is_outlier: bool
    values: dict[str, float]


class ClusterResult(BaseModel):
    chosen_k: int
    auto_selected_k: bool
    silhouette: float
    id_column: str | None
    feature_columns: list[str]
    row_count: int
    candidates: list[CandidateK]
    segments: list[SegmentProfile]
    assignments: list[RowAssignment]
    headline: str


def _examples_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "examples").is_dir():
        return here / "examples"
    repo_ex = here.parent.parent / "examples"
    if repo_ex.is_dir():
        return repo_ex
    raise FileNotFoundError("Could not locate examples directory.")


def list_examples() -> list[str]:
    ex = _examples_dir()
    return sorted(p.stem for p in ex.glob("*.csv"))


def load_example(name: str) -> pd.DataFrame:
    ex = _examples_dir() / f"{name}.csv"
    if not ex.exists():
        available = ", ".join(list_examples())
        raise ValueError(f"Unknown example '{name}'. Available: {available}")
    return pd.read_csv(ex)


def _describe_segment(cluster_id: int, z_center: np.ndarray, feature_cols: list[str]) -> tuple[str, list[str]]:
    ranked = sorted(enumerate(z_center), key=lambda idx_val: abs(idx_val[1]), reverse=True)
    standout: list[str] = []
    short_parts: list[str] = []
    for idx, z_val in ranked[:3]:
        col = feature_cols[idx]
        clean_col = col.replace("_", " ")
        if z_val >= 0.55:
            direction = "High"
        elif z_val <= -0.55:
            direction = "Low"
        else:
            direction = "Near-average"
        standout.append(f"{direction} {clean_col} ({z_val:+.2f} SD)")
        if len(short_parts) < 2 and direction != "Near-average":
            short_parts.append(f"{direction.lower()} {clean_col}")
    if not short_parts:
        label = f"Segment {cluster_id + 1}: balanced profile"
    else:
        label = f"Segment {cluster_id + 1}: " + ", ".join(short_parts)
    return label, standout


def cluster_dataframe(
    df: pd.DataFrame,
    k: int | None = None,
    max_k: int = 6,
    id_column: str | None = None,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> ClusterResult:
    if df.empty:
        raise ValueError("Input table is empty.")

    # Identify ID column if not provided
    if id_column is None:
        non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            id_column = str(non_num[0])
    elif id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not found in columns: {list(df.columns)}")

    if feature_columns is None:
        feature_columns = [
            str(c) for c in df.columns
            if c != id_column and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")

    if not feature_columns:
        raise ValueError("No numeric feature columns found to cluster.")

    work = df.dropna(subset=feature_columns).copy()
    n_rows = len(work)
    if n_rows < 4:
        raise ValueError(f"Need at least 4 complete rows to cluster (got {n_rows}).")

    X_raw = work[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    candidates: list[CandidateK] = []
    auto_selected = k is None
    if k is not None:
        if k < 2 or k >= n_rows:
            raise ValueError(f"k must be between 2 and {n_rows - 1} (got {k}).")
        candidate_ks = [k]
    else:
        upper_k = min(max_k, n_rows - 1)
        candidate_ks = list(range(2, max(3, upper_k + 1)))

    best_k = candidate_ks[0]
    best_sil = -1.0
    models: dict[int, tuple[KMeans, float]] = {}

    for ck in candidate_ks:
        km = KMeans(n_clusters=ck, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0
        candidates.append(CandidateK(k=ck, inertia=round(float(km.inertia_), 4), silhouette=round(sil, 4)))
        models[ck] = (km, sil)
        if sil > best_sil:
            best_sil = sil
            best_k = ck

    chosen_k = best_k if auto_selected else int(k)
    km_model, chosen_sil = models[chosen_k]
    raw_labels = km_model.labels_
    centers_z = km_model.cluster_centers_

    # Order cluster IDs deterministically by descending size, then first feature mean
    counts = pd.Series(raw_labels).value_counts()
    ordered_old_ids = sorted(
        range(chosen_k),
        key=lambda cid: (-int(counts.get(cid, 0)), -float(centers_z[cid, 0])),
    )
    remap = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)}
    mapped_labels = np.array([remap[int(c)] for c in raw_labels], dtype=int)
    ordered_centers_z = np.array([centers_z[old_id] for old_id in ordered_old_ids])

    # Distance of each row to its assigned cluster center in standardized space
    distances = np.linalg.norm(X_scaled - ordered_centers_z[mapped_labels], axis=1)
    outlier_cutoff = max(2.2, float(np.percentile(distances, 92)))

    segments: list[SegmentProfile] = []
    segment_labels: dict[int, str] = {}
    for cid in range(chosen_k):
        mask = mapped_labels == cid
        cnt = int(np.sum(mask))
        share = round((cnt / n_rows) * 100.0, 1)
        means = {
            col: round(float(work.loc[mask, col].mean()), 3)
            for col in feature_columns
        }
        seg_label, standout = _describe_segment(cid, ordered_centers_z[cid], feature_columns)
        segment_labels[cid] = seg_label
        segments.append(
            SegmentProfile(
                cluster_id=cid,
                label=seg_label,
                count=cnt,
                share_pct=share,
                feature_means=means,
                standout_features=standout,
            )
        )

    assignments: list[RowAssignment] = []
    for idx_pos, (_, row) in enumerate(work.iterrows()):
        rid = str(row[id_column]) if id_column else f"row-{idx_pos + 1}"
        cid = int(mapped_labels[idx_pos])
        dist = round(float(distances[idx_pos]), 3)
        is_out = bool(dist > outlier_cutoff and n_rows >= 8)
        vals = {col: round(float(row[col]), 4) for col in feature_columns}
        assignments.append(
            RowAssignment(
                row_id=rid,
                cluster_id=cid,
                segment_label=segment_labels[cid],
                distance_to_center=dist,
                is_outlier=is_out,
                values=vals,
            )
        )

    n_outliers = sum(1 for a in assignments if a.is_outlier)
    headline = (
        f"Grouped {n_rows} rows across {len(feature_columns)} features into {chosen_k} segments "
        f"(silhouette {chosen_sil:.2f}, {n_outliers} outlier{'s' if n_outliers != 1 else ''} flagged)."
    )

    return ClusterResult(
        chosen_k=chosen_k,
        auto_selected_k=auto_selected,
        silhouette=round(chosen_sil, 4),
        id_column=id_column,
        feature_columns=feature_columns,
        row_count=n_rows,
        candidates=candidates,
        segments=segments,
        assignments=assignments,
        headline=headline,
    )


def cluster_csv(
    csv_source: str | Path | io.StringIO,
    k: int | None = None,
    max_k: int = 6,
    id_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> ClusterResult:
    if isinstance(csv_source, (str, Path)) and Path(csv_source).exists():
        df = pd.read_csv(csv_source)
    elif isinstance(csv_source, str):
        df = pd.read_csv(io.StringIO(csv_source))
    else:
        df = pd.read_csv(csv_source)
    return cluster_dataframe(df, k=k, max_k=max_k, id_column=id_column, feature_columns=feature_columns)
