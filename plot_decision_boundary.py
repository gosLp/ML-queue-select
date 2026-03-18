#!/usr/bin/env python3
"""
plot_decision_boundary.py

Generate interpretable plots for GPU queue-selection ML results.

Reads the aggregated ML dataset (from aggregate_results.py), retrains a model
on the same feature space, and produces:

1. Oracle queue selection map
2. Predicted queue selection map
3. Confusion matrix heatmap
4. Throughput vs threads line plot for a chosen test / ops_per_thread setting

This is meant for milestone/demo/report visuals.

Example:
  python plot_decision_boundary.py \
    --input outputs/queue_sweep_20260317_235346_agg_ml.csv \
    --queues sfq,broker \
    --model random_forest \
    --out-dir decision_plots

Optional:
  python plot_decision_boundary.py \
    --input outputs/queue_sweep_20260317_235346_agg_ml.csv \
    --queues wfq,sfq,broker \
    --model logistic_regression \
    --test-name balanced \
    --ops-per-thread 256 \
    --out-dir decision_plots_all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def infer_available_queues(df: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested

    queues = []
    for c in df.columns:
        if c.endswith("_throughput_mops") and not c.startswith("best_"):
            queues.append(c[: -len("_throughput_mops")])
    queues = sorted(set(queues))
    if not queues:
        raise ValueError("Could not infer queue names from *_throughput_mops columns.")
    return queues


def validate_columns(df: pd.DataFrame, queues: List[str]) -> None:
    required = ["test", "threads", "ops_per_thread", "producer_ratio", "block_size", "best_queue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_q = [f"{q}_throughput_mops" for q in queues if f"{q}_throughput_mops" not in df.columns]
    if missing_q:
        raise ValueError(f"Missing throughput columns: {missing_q}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["test", "threads", "ops_per_thread", "producer_ratio", "block_size"]].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["threads", "ops_per_thread", "producer_ratio", "block_size"]
    categorical_features = ["test"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_model(model_name: str, random_state: int) -> Pipeline:
    pre = build_preprocessor()

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
        )
    elif model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[
        ("preprocess", pre),
        ("model", clf),
    ])


def make_queue_color_map(queues: List[str]) -> dict:
    base_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    return {q: base_colors[i % len(base_colors)] for i, q in enumerate(queues)}


def plot_queue_map(
    df: pd.DataFrame,
    label_col: str,
    title: str,
    queues: List[str],
    out_path: Path,
) -> None:
    color_map = make_queue_color_map(queues)

    plt.figure(figsize=(8, 6))

    for q in queues:
        sub = df[df[label_col] == q]
        if sub.empty:
            continue

        plt.scatter(
            sub["threads"],
            sub["producer_ratio"],
            label=q,
            alpha=0.85,
            s=70,
            c=color_map[q],
            edgecolors="black",
            linewidths=0.4,
        )

    plt.xscale("log", base=2)
    plt.xlabel("Threads (log2 scale)")
    plt.ylabel("Producer Ratio (%)")
    plt.title(title)
    plt.legend(title="Queue")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_oracle_vs_prediction_side_by_side(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    queues: List[str],
    out_path: Path,
) -> None:
    color_map = make_queue_color_map(queues)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, col, title in [
        (axes[0], true_col, "Oracle Best Queue"),
        (axes[1], pred_col, "Model Predicted Queue"),
    ]:
        for q in queues:
            sub = df[df[col] == q]
            if sub.empty:
                continue
            ax.scatter(
                sub["threads"],
                sub["producer_ratio"],
                label=q,
                alpha=0.85,
                s=70,
                c=color_map[q],
                edgecolors="black",
                linewidths=0.4,
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads (log2 scale)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Producer Ratio (%)")

    handles = [
        Line2D([0], [0], marker="o", color="w", label=q,
               markerfacecolor=color_map[q], markeredgecolor="black", markersize=8)
        for q in queues
    ]
    fig.legend(handles=handles, labels=queues, loc="upper center", ncol=len(queues), title="Queue")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    y_true: pd.Series,
    y_pred: np.ndarray,
    queues: List[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=queues)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(queues)))
    ax.set_yticks(np.arange(len(queues)))
    ax.set_xticklabels(queues)
    ax.set_yticklabels(queues)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(queues)):
        for j in range(len(queues)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def choose_throughput_slice(df: pd.DataFrame, test_name: Optional[str], ops_per_thread: Optional[int]) -> Tuple[str, int]:
    if test_name is None:
        test_name = str(df["test"].mode().iloc[0])
    if ops_per_thread is None:
        ops_per_thread = int(df["ops_per_thread"].mode().iloc[0])
    return test_name, ops_per_thread


def plot_throughput_vs_threads(
    df: pd.DataFrame,
    queues: List[str],
    test_name: str,
    ops_per_thread: int,
    out_path: Path,
) -> None:
    sub = df[(df["test"] == test_name) & (df["ops_per_thread"] == ops_per_thread)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for test={test_name}, ops_per_thread={ops_per_thread}")

    # If multiple producer ratios exist, aggregate by median per thread.
    grouped = sub.groupby("threads", as_index=False).median(numeric_only=True).sort_values("threads")

    plt.figure(figsize=(8, 6))
    for q in queues:
        col = f"{q}_throughput_mops"
        if col not in grouped.columns:
            continue
        plt.plot(grouped["threads"], grouped[col], marker="o", label=q)

    plt.xscale("log", base=2)
    plt.xlabel("Threads (log2 scale)")
    plt.ylabel("Throughput (Mops/s)")
    plt.title(f"Throughput vs Threads\n(test={test_name}, ops_per_thread={ops_per_thread}, median over producer_ratio)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Queue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot oracle/predicted queue selection boundaries.")
    parser.add_argument("--input", required=True, help="Input aggregated ML CSV, e.g. *_agg_ml.csv")
    parser.add_argument("--queues", default="", help="Comma-separated queues, e.g. wfq,sfq,broker")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-name", default=None, help="For throughput plot; default = mode")
    parser.add_argument("--ops-per-thread", type=int, default=None, help="For throughput plot; default = mode")
    parser.add_argument("--out-dir", default="decision_plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_queues = parse_csv_strings(args.queues) if args.queues else None

    df = load_dataset(input_path)
    queues = infer_available_queues(df, requested_queues)
    validate_columns(df, queues)

    # Keep only rows with labels in selected queue set
    df = df[df["best_queue"].isin(queues)].copy().reset_index(drop=True)

    X = build_features(df)
    y = df["best_queue"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(y.unique()) > 1 else None,
    )

    model = build_model(args.model, args.random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    df_test["predicted_queue"] = y_pred

    # Plots
    plot_queue_map(
        df_test,
        label_col="best_queue",
        title="Oracle Best Queue on Test Set",
        queues=queues,
        out_path=out_dir / "oracle_queue_map.png",
    )

    plot_queue_map(
        df_test,
        label_col="predicted_queue",
        title=f"Predicted Queue on Test Set ({args.model})",
        queues=queues,
        out_path=out_dir / "predicted_queue_map.png",
    )

    plot_oracle_vs_prediction_side_by_side(
        df_test,
        true_col="best_queue",
        pred_col="predicted_queue",
        queues=queues,
        out_path=out_dir / "oracle_vs_prediction_side_by_side.png",
    )

    plot_confusion_matrix_heatmap(
        y_true=y_test.reset_index(drop=True),
        y_pred=y_pred,
        queues=queues,
        out_path=out_dir / "confusion_matrix.png",
    )

    tname, ops = choose_throughput_slice(df, args.test_name, args.ops_per_thread)
    plot_throughput_vs_threads(
        df=df,
        queues=queues,
        test_name=tname,
        ops_per_thread=ops,
        out_path=out_dir / "throughput_vs_threads.png",
    )

    print("Saved plots:")
    print(f"  {out_dir / 'oracle_queue_map.png'}")
    print(f"  {out_dir / 'predicted_queue_map.png'}")
    print(f"  {out_dir / 'oracle_vs_prediction_side_by_side.png'}")
    print(f"  {out_dir / 'confusion_matrix.png'}")
    print(f"  {out_dir / 'throughput_vs_threads.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())