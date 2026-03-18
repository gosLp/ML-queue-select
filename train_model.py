#!/usr/bin/env python3
"""
train_model.py

Train simple ML models to predict the best GPU queue for a workload.

Expected input:
  Aggregated ML dataset from aggregate_results.py, e.g.
    outputs/pilot_ml.csv

The dataset should contain columns like:
  - test
  - threads
  - ops_per_thread
  - producer_ratio
  - block_size
  - best_queue
  - best_throughput_mops
  - wfq_throughput_mops
  - sfq_throughput_mops
  - broker_throughput_mops   (optional if broker exists)

Example:
  python train_model.py --input outputs/pilot_ml.csv

Optional:
  python train_model.py \
    --input outputs/pilot_ml.csv \
    --queues wfq,sfq,broker \
    --test-size 0.2 \
    --random-state 42 \
    --out-dir ml_outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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
        if c.endswith("_throughput_mops"):
            q = c[: -len("_throughput_mops")]
            if q != "best":
                queues.append(q)
    queues = sorted(set(queues))
    if not queues:
        raise ValueError("Could not infer queue columns from dataset.")
    return queues


def validate_columns(df: pd.DataFrame, queues: List[str]) -> None:
    required = [
        "test",
        "threads",
        "ops_per_thread",
        "producer_ratio",
        "block_size",
        "best_queue",
        "best_throughput_mops",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_q = [f"{q}_throughput_mops" for q in queues if f"{q}_throughput_mops" not in df.columns]
    if missing_q:
        raise ValueError(f"Missing queue throughput columns: {missing_q}")


def build_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[["test", "threads", "ops_per_thread", "producer_ratio", "block_size"]].copy()
    y = df["best_queue"].copy()
    return X, y


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_models(random_state: int) -> Dict[str, Pipeline]:
    pre = build_preprocessor()

    rf = Pipeline(steps=[
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        )),
    ])

    lr = Pipeline(steps=[
        ("preprocess", pre),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        )),
    ])

    return {
        "random_forest": rf,
        "logistic_regression": lr,
    }


def best_fixed_queue(train_y: pd.Series) -> str:
    return train_y.mode().iloc[0]


def compute_near_optimal_and_regret(
    df_test: pd.DataFrame,
    pred_labels: np.ndarray,
    queues: List[str],
    threshold: float = 0.95,
) -> Tuple[float, float]:
    """
    Returns:
      near_optimal_rate, avg_regret_mops
    """
    near = 0
    regrets = []

    for i, pred_q in enumerate(pred_labels):
        row = df_test.iloc[i]
        best_thr = float(row["best_throughput_mops"])
        pred_col = f"{pred_q}_throughput_mops"

        pred_thr = row.get(pred_col, np.nan)
        if pd.isna(pred_thr):
            continue

        pred_thr = float(pred_thr)
        if best_thr > 0 and pred_thr >= threshold * best_thr:
            near += 1

        regrets.append(best_thr - pred_thr)

    near_rate = near / len(df_test) if len(df_test) > 0 else 0.0
    avg_regret = float(np.mean(regrets)) if regrets else float("nan")
    return near_rate, avg_regret


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    queues: List[str],
) -> Dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    near_rate, avg_regret = compute_near_optimal_and_regret(df_test, preds, queues)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=sorted(y_train.unique()))

    return {
        "name": name,
        "accuracy": float(acc),
        "near_optimal_rate": float(near_rate),
        "avg_regret_mops": float(avg_regret),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "model": model,
    }


def evaluate_best_fixed(
    fixed_q: str,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    queues: List[str],
) -> Dict:
    preds = np.array([fixed_q] * len(y_test))
    acc = accuracy_score(y_test, preds)
    near_rate, avg_regret = compute_near_optimal_and_regret(df_test, preds, queues)

    return {
        "name": "best_fixed",
        "accuracy": float(acc),
        "near_optimal_rate": float(near_rate),
        "avg_regret_mops": float(avg_regret),
        "fixed_queue": fixed_q,
        "predictions": preds.tolist(),
    }


def get_rf_feature_importance(rf_pipeline: Pipeline) -> pd.DataFrame:
    pre = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]

    feature_names = pre.get_feature_names_out()
    importances = model.feature_importances_

    out = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return out


def plot_accuracy(results: List[Dict], out_path: Path) -> None:
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, accs)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_near_optimal(results: List[Dict], out_path: Path) -> None:
    names = [r["name"] for r in results]
    vals = [r["near_optimal_rate"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, vals)
    plt.ylabel("Near-Optimal Rate")
    plt.title("Near-Optimal Performance Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_regret(results: List[Dict], out_path: Path) -> None:
    names = [r["name"] for r in results]
    vals = [r["avg_regret_mops"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, vals)
    plt.ylabel("Average Regret (Mops/s)")
    plt.title("Average Regret Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, out_path: Path, top_k: int = 10) -> None:
    top = fi_df.head(top_k).iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Random Forest Feature Importance (Top {top_k})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_metrics_json(results: List[Dict], out_path: Path) -> None:
    serializable = []
    for r in results:
        x = {k: v for k, v in r.items() if k not in {"model"}}
        serializable.append(x)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_summary_csv(results: List[Dict], out_path: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "model": r["name"],
            "accuracy": r["accuracy"],
            "near_optimal_rate": r["near_optimal_rate"],
            "avg_regret_mops": r["avg_regret_mops"],
            "fixed_queue": r.get("fixed_queue", ""),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train queue-selection ML models.")
    parser.add_argument("--input", required=True, help="Input aggregated ML CSV")
    parser.add_argument("--queues", default="", help="Comma-separated queue names, e.g. wfq,sfq,broker")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="ml_outputs", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    queues = infer_available_queues(df, parse_csv_strings(args.queues) if args.queues else None)
    validate_columns(df, queues)

    # Keep only rows whose best_queue is in the queue set
    df = df[df["best_queue"].isin(queues)].copy()

    X, y = build_features_and_labels(df)

    # Keep original test dataframe aligned with split indices
    idx = np.arange(len(df))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(y.unique()) > 1 else None,
    )

    df_test = df.iloc[idx_test].reset_index(drop=True)

    models = build_models(args.random_state)
    results = []

    # Best fixed baseline
    fixed_q = best_fixed_queue(y_train)
    fixed_result = evaluate_best_fixed(fixed_q, y_test.reset_index(drop=True), df_test, queues)
    results.append(fixed_result)

    # ML models
    for name, model in models.items():
        res = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test.reset_index(drop=True),
            df_test=df_test,
            queues=queues,
        )
        results.append(res)

    # Feature importance from RF
    rf_result = next(r for r in results if r["name"] == "random_forest")
    fi_df = get_rf_feature_importance(rf_result["model"])

    # Save artifacts
    save_metrics_json(results, out_dir / "metrics.json")
    save_summary_csv(results, out_dir / "summary.csv")
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)

    plot_accuracy(results, out_dir / "accuracy.png")
    plot_near_optimal(results, out_dir / "near_optimal.png")
    plot_regret(results, out_dir / "regret.png")
    plot_feature_importance(fi_df, out_dir / "feature_importance.png", top_k=10)

    # Print concise report
    print("\n=== Queue Selection ML Results ===")
    print(f"Input dataset: {input_path}")
    print(f"Rows used: {len(df)}")
    print(f"Queues: {queues}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print()

    for r in results:
        print(f"Model: {r['name']}")
        if r["name"] == "best_fixed":
            print(f"  Fixed queue:        {r['fixed_queue']}")
        print(f"  Accuracy:           {r['accuracy']:.4f}")
        print(f"  Near-optimal rate:  {r['near_optimal_rate']:.4f}")
        print(f"  Avg regret (Mops):  {r['avg_regret_mops']:.4f}")
        print()

    print("Top Random Forest Features:")
    print(fi_df.head(10).to_string(index=False))
    print()

    print("Saved outputs:")
    print(f"  {out_dir / 'metrics.json'}")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'feature_importance.csv'}")
    print(f"  {out_dir / 'accuracy.png'}")
    print(f"  {out_dir / 'near_optimal.png'}")
    print(f"  {out_dir / 'regret.png'}")
    print(f"  {out_dir / 'feature_importance.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())