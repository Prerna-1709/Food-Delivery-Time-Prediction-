"""K-Means and Agglomerative Clustering for delivery pattern segmentation.

Functions
---------
run_kmeans            -- K-Means with Elbow Method to find optimal k.
run_hierarchical      -- Agglomerative Clustering + Dendrogram.
plot_pca_clusters     -- 2D PCA scatter coloured by cluster label.
analyze_clusters      -- Avg delivery time and size per cluster.
run_all               -- Orchestrates all steps and saves figures.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.config import RANDOM_SEED, REPORTS_FIGURES

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
CLUSTER_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                   "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
BG = "#F7F9FC"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

K_RANGE = range(2, 11)    # candidate k values for Elbow / Silhouette


def _save(fig: plt.Figure, filename: str) -> str:
    os.makedirs(REPORTS_FIGURES, exist_ok=True)
    path = os.path.join(REPORTS_FIGURES, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ---------------------------------------------------------------------------
# 1. K-Means + Elbow Method
# ---------------------------------------------------------------------------
def run_kmeans(X: np.ndarray, k_range=K_RANGE) -> tuple[KMeans, int, str]:
    """Fit K-Means for each k in k_range; pick optimal k via Elbow + Silhouette.

    Parameters
    ----------
    X       : np.ndarray — encoded feature matrix (already scaled).
    k_range : iterable  — candidate k values to test.

    Returns
    -------
    best_model : KMeans  — fitted with the optimal k.
    best_k     : int
    elbow_path : str     — path to the saved Elbow plot PNG.
    """
    wcss        = []
    sil_scores  = []
    k_list      = list(k_range)

    for k in k_list:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil = silhouette_score(X, labels) if k > 1 else 0.0
        sil_scores.append(sil)

    # --- Elbow: largest second-derivative drop in WCSS ---
    wcss_arr = np.array(wcss)
    deltas   = np.diff(wcss_arr)           # first differences
    knee_idx = int(np.argmax(np.abs(np.diff(deltas)))) + 1   # second difference
    best_k   = k_list[knee_idx]

    # Also check silhouette peak as a tie-breaker hint
    sil_best_k = k_list[int(np.argmax(sil_scores))]
    print(f"  Elbow suggests k={best_k}  |  Silhouette peak at k={sil_best_k}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # WCSS (Elbow)
    ax = axes[0]
    ax.plot(k_list, wcss, "o-", color="#4C72B0", lw=2, markersize=7)
    ax.axvline(best_k, color="#C44E52", linestyle="--", lw=1.5,
               label=f"Optimal k = {best_k}")
    ax.set_title("Elbow Method (WCSS)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Within-Cluster Sum of Squares", fontsize=11)
    ax.legend(fontsize=10)

    # Silhouette
    ax = axes[1]
    ax.plot(k_list, sil_scores, "s-", color="#DD8452", lw=2, markersize=7)
    ax.axvline(sil_best_k, color="#55A868", linestyle="--", lw=1.5,
               label=f"Silhouette peak k = {sil_best_k}")
    ax.set_title("Silhouette Score vs k", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.legend(fontsize=10)

    fig.suptitle("K-Means Cluster Selection", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    elbow_path = _save(fig, "kmeans_elbow.png")

    # Refit with best k
    best_model = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    best_model.fit(X)
    print(f"  K-Means fitted with k={best_k}  |  Inertia={best_model.inertia_:.2f}")
    return best_model, best_k, elbow_path


# ---------------------------------------------------------------------------
# 2. Agglomerative Clustering + Dendrogram
# ---------------------------------------------------------------------------
def run_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    method: str = "ward",
) -> tuple[AgglomerativeClustering, str]:
    """Fit Agglomerative Clustering and save a Dendrogram.

    Parameters
    ----------
    X          : np.ndarray — encoded feature matrix.
    n_clusters : int        — number of clusters (use best_k from Elbow).
    method     : str        — linkage method ('ward', 'complete', 'average').

    Returns
    -------
    model : AgglomerativeClustering — fitted model.
    path  : str                     — path to saved Dendrogram PNG.
    """
    # Dendrogram (uses scipy linkage on full data)
    Z = linkage(X, method=method)

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode="lastp",   # show only last p merged clusters
        p=30,
        leaf_rotation=90,
        leaf_font_size=9,
        color_threshold=Z[-(n_clusters - 1), 2],   # colour cut at n_clusters
        ax=ax,
    )
    # Horizontal cut line
    cut_height = Z[-(n_clusters), 2] if n_clusters <= len(Z) else Z[-1, 2]
    ax.axhline(cut_height, color="#C44E52", linestyle="--", lw=1.5,
               label=f"Cut → {n_clusters} clusters")
    ax.set_title(f"Hierarchical Clustering Dendrogram  (linkage='{method}')",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index (or cluster size)", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    dendro_path = _save(fig, "hierarchical_dendrogram.png")

    # Fit model
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    model.fit(X)
    print(f"  Agglomerative fitted: {n_clusters} clusters, linkage='{method}'")
    return model, dendro_path


# ---------------------------------------------------------------------------
# 3. PCA 2D Scatter
# ---------------------------------------------------------------------------
def plot_pca_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA Cluster Scatter",
    filename: str = "pca_clusters.png",
) -> str:
    """Reduce X to 2D via PCA and plot a scatter coloured by cluster label.

    Parameters
    ----------
    X        : np.ndarray  — encoded feature matrix.
    labels   : np.ndarray  — integer cluster assignments.
    title    : str
    filename : str         — output filename in REPORTS_FIGURES.

    Returns
    -------
    str — path to saved PNG.
    """
    pca  = PCA(n_components=2, random_state=RANDOM_SEED)
    pcs  = pca.fit_transform(X)
    var1, var2 = pca.explained_variance_ratio_ * 100

    df_pca = pd.DataFrame({
        "PC1":    pcs[:, 0],
        "PC2":    pcs[:, 1],
        "Cluster": labels.astype(str),
    })

    n_clusters = len(np.unique(labels))
    palette    = {str(i): CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                  for i in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_id, grp in df_pca.groupby("Cluster"):
        ax.scatter(
            grp["PC1"], grp["PC2"],
            c=palette[cluster_id], label=f"Cluster {cluster_id}",
            alpha=0.85, edgecolors="white", linewidth=0.4, s=80,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.legend(title="Cluster", fontsize=9, title_fontsize=10)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 4. Cluster Analysis
# ---------------------------------------------------------------------------
def analyze_clusters(df_raw: pd.DataFrame, labels: np.ndarray,
                     algo_name: str = "K-Means") -> pd.DataFrame:
    """Attach cluster labels to the raw dataframe and summarise each cluster.

    Metrics per cluster:
      - count, avg / min / max delivery time, % Delayed, rush-hour rate.

    Parameters
    ----------
    df_raw    : pd.DataFrame — original (un-encoded) dataframe with Time_taken.
    labels    : np.ndarray   — cluster assignments (aligned with df_raw).
    algo_name : str          — displayed in output header.

    Returns
    -------
    pd.DataFrame — summary statistics, one row per cluster.
    """
    df = df_raw.copy()
    df["Cluster"] = labels

    summary = (
        df.groupby("Cluster")
          .agg(
              Count         = ("Time_taken(min)", "count"),
              Avg_Time_min  = ("Time_taken(min)", "mean"),
              Min_Time_min  = ("Time_taken(min)", "min"),
              Max_Time_min  = ("Time_taken(min)", "max"),
              Pct_Delayed   = ("Time_taken(min)", lambda s: (s > 40).mean() * 100),
          )
          .round(2)
    )

    sep = "-" * 65
    print(f"\n  [{algo_name}] Cluster Analysis:")
    print("  " + sep)
    print("  " + summary.to_string())
    print("  " + sep)
    return summary


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------
def run_all(df_raw: pd.DataFrame, X: np.ndarray) -> dict:
    """Run complete unsupervised pipeline and save all figures.

    Parameters
    ----------
    df_raw : pd.DataFrame — original dataframe (for cluster analysis labels).
    X      : np.ndarray  — encoded, scaled feature matrix from DataProcessor.

    Returns
    -------
    dict with keys: best_k, km_labels, hc_labels, summaries, paths.
    """
    sep = "=" * 60

    # ── K-Means ──────────────────────────────────────────────────
    print(f"\n[1] K-Means Clustering + Elbow Method ...")
    km_model, best_k, elbow_path = run_kmeans(X)
    km_labels = km_model.labels_

    km_pca_path = plot_pca_clusters(
        X, km_labels,
        title=f"K-Means Clusters (k={best_k}) — PCA Projection",
        filename="kmeans_pca_scatter.png",
    )
    km_summary = analyze_clusters(df_raw, km_labels, algo_name=f"K-Means k={best_k}")

    # ── Hierarchical ──────────────────────────────────────────────
    print(f"\n[2] Hierarchical Clustering (Agglomerative, n={best_k}) ...")
    hc_model, dendro_path = run_hierarchical(X, n_clusters=best_k)
    hc_labels = hc_model.labels_

    hc_pca_path = plot_pca_clusters(
        X, hc_labels,
        title=f"Agglomerative Clusters (n={best_k}) — PCA Projection",
        filename="hierarchical_pca_scatter.png",
    )
    hc_summary = analyze_clusters(df_raw, hc_labels, algo_name=f"Agglomerative n={best_k}")

    print(f"\n[3] Saved figures:")
    for path in [elbow_path, km_pca_path, dendro_path, hc_pca_path]:
        sz = os.path.getsize(path) / 1024
        print(f"    {os.path.basename(path):<45} ({sz:.1f} KB)")

    return dict(
        best_k=best_k,
        km_labels=km_labels,  hc_labels=hc_labels,
        km_summary=km_summary, hc_summary=hc_summary,
        paths=[elbow_path, km_pca_path, dendro_path, hc_pca_path],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing.processor import DataProcessor
    from src.utils.config import RAW_CSV

    sep = "=" * 60
    print(f"\n{sep}")
    print("  STEP 3 -- UNSUPERVISED CLUSTERING")
    print(sep)

    print("\n[0] Loading and preprocessing data ...")
    df_raw = pd.read_csv(RAW_CSV)
    proc   = DataProcessor()
    X_train, X_test, _, _ = proc.fit_transform(df_raw)

    # Use the full encoded matrix for clustering (unsupervised — no labels)
    X_full = np.vstack([X_train.values, X_test.values])
    # Align df_raw index order with the split (reset to natural order)
    df_aligned = df_raw.copy().reset_index(drop=True)
    print(f"    Feature matrix: {X_full.shape}")

    results = run_all(df_aligned, X_full)
    print(f"\n[DONE] Optimal k={results['best_k']}. All figures in reports/figures/.\n")
