import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "test",
    "threads",
    "ops_per_thread",
    "producer_ratio",
    "block_size",
]


def infer_available_queues(df: pd.DataFrame) -> List[str]:
    queues = []
    for col in df.columns:
        if col.endswith("_throughput_mops") and col != "best_throughput_mops":
            queues.append(col.replace("_throughput_mops", ""))
    return sorted(queues)


def build_preprocessor() -> ColumnTransformer:
    categorical_features = ["test"]
    numeric_features = ["threads", "ops_per_thread", "producer_ratio", "block_size"]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


def make_producer_ratio_bucket(series: pd.Series) -> pd.Series:
    # Proposal says stratified by producer-ratio bucket.
    # If producer_ratio already takes discrete values like 0,25,50,75,100 this preserves them.
    return series.astype(str)


def compute_metrics(
    df_eval: pd.DataFrame,
    predicted_queues: List[str],
    best_fixed_queue: str,
) -> Dict[str, float]:
    true_queues = df_eval["best_queue"].tolist()
    top1_accuracy = accuracy_score(true_queues, predicted_queues)

    regrets = []
    near_oracle = []
    speedups = []

    for (_, row), pred in zip(df_eval.iterrows(), predicted_queues):
        oracle_tput = float(row["best_throughput_mops"])
        chosen_tput = float(row[f"{pred}_throughput_mops"])
        best_fixed_tput = float(row[f"{best_fixed_queue}_throughput_mops"])

        regret = (
            (oracle_tput - chosen_tput) / oracle_tput
            if oracle_tput > 0
            else 0.0
        )
        regrets.append(regret)

        near_oracle.append(1 if chosen_tput >= 0.95 * oracle_tput else 0)

        speedup = (
            chosen_tput / best_fixed_tput
            if best_fixed_tput > 0
            else 0.0
        )
        speedups.append(speedup)

    return {
        "top1_accuracy": float(np.mean(top1_accuracy)),
        "near_oracle_rate": float(np.mean(near_oracle)),
        "avg_regret": float(np.mean(regrets)),
        "speedup_vs_best_fixed": float(np.mean(speedups)),
    }


def select_best_fixed_queue(df_train: pd.DataFrame, queues: List[str]) -> str:
    avg_tputs = {
        q: float(df_train[f"{q}_throughput_mops"].mean())
        for q in queues
    }
    return max(avg_tputs, key=avg_tputs.get)


def baseline_best_fixed(df_eval: pd.DataFrame, best_fixed_queue: str) -> List[str]:
    return [best_fixed_queue] * len(df_eval)


def baseline_random(df_eval: pd.DataFrame, queues: List[str], random_state: int) -> List[str]:
    rng = np.random.default_rng(random_state)
    return rng.choice(queues, size=len(df_eval)).tolist()


def baseline_rule_of_thumb(df_eval: pd.DataFrame, queues: List[str]) -> List[str]:
    # Simple threshold selector per proposal idea.
    # If both broker and sfq are present, choose broker for high producer ratio else sfq.
    # Otherwise fall back to first queue.
    if "broker" in queues and "sfq" in queues:
        return [
            "broker" if float(row["producer_ratio"]) > 50 else "sfq"
            for _, row in df_eval.iterrows()
        ]
    return [queues[0]] * len(df_eval)


def baseline_oracle(df_eval: pd.DataFrame) -> List[str]:
    return df_eval["best_queue"].tolist()


def build_models(random_state: int) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]
        ),
    }


def evaluate_supervised_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    df_eval: pd.DataFrame,
    best_fixed_queue: str,
) -> Dict[str, float]:
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    metrics = compute_metrics(df_eval, preds.tolist(), best_fixed_queue)
    metrics["model"] = name
    return metrics


def cross_validate_models(
    df: pd.DataFrame,
    queues: List[str],
    random_state: int,
    cv_folds: int,
) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS]
    y = df["best_queue"]
    strat = make_producer_ratio_bucket(df["producer_ratio"])

    # Combine y and producer-ratio bucket so CV stratification honors both label and proposal bucket idea.
    combined_strat = y.astype(str) + "__" + strat.astype(str)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, combined_strat), start=1):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        X_train = df_train[FEATURE_COLUMNS]
        X_test = df_test[FEATURE_COLUMNS]
        y_train = df_train["best_queue"]

        best_fixed_queue = select_best_fixed_queue(df_train, queues)

        # Baselines
        baseline_preds = {
            "oracle": baseline_oracle(df_test),
            "best_fixed": baseline_best_fixed(df_test, best_fixed_queue),
            "rule_of_thumb": baseline_rule_of_thumb(df_test, queues),
            "random": baseline_random(df_test, queues, random_state + fold),
        }

        for name, preds in baseline_preds.items():
            metrics = compute_metrics(df_test, preds, best_fixed_queue)
            metrics["model"] = name
            metrics["fold"] = fold
            fold_rows.append(metrics)

        # Supervised models
        for name, model in build_models(random_state + fold).items():
            metrics = evaluate_supervised_model(
                name=name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_test,
                df_eval=df_test,
                best_fixed_queue=best_fixed_queue,
            )
            metrics["fold"] = fold
            fold_rows.append(metrics)

    return pd.DataFrame(fold_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Proposal-aligned queue model training with Random Forest kept as primary model."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to aggregated ML CSV file.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Holdout test fraction (proposal uses 70/30 split).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds.")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    required_cols = FEATURE_COLUMNS + ["best_queue", "best_throughput_mops"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    queues = infer_available_queues(df)
    if not queues:
        raise ValueError("No queue throughput columns found in input CSV.")

    print("Available queues:", queues)

    strat = make_producer_ratio_bucket(df["producer_ratio"])
    combined_strat = df["best_queue"].astype(str) + "__" + strat.astype(str)

    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=combined_strat,
    )

    X_train = df_train[FEATURE_COLUMNS]
    X_test = df_test[FEATURE_COLUMNS]
    y_train = df_train["best_queue"]

    best_fixed_queue = select_best_fixed_queue(df_train, queues)
    print("Best fixed queue:", best_fixed_queue)

    holdout_rows = []

    # Baselines
    baseline_preds = {
        "oracle": baseline_oracle(df_test),
        "best_fixed": baseline_best_fixed(df_test, best_fixed_queue),
        "rule_of_thumb": baseline_rule_of_thumb(df_test, queues),
        "random": baseline_random(df_test, queues, args.random_state),
    }

    for name, preds in baseline_preds.items():
        metrics = compute_metrics(df_test, preds, best_fixed_queue)
        metrics["model"] = name
        holdout_rows.append(metrics)

    # Supervised models
    for name, model in build_models(args.random_state).items():
        metrics = evaluate_supervised_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            df_eval=df_test,
            best_fixed_queue=best_fixed_queue,
        )
        holdout_rows.append(metrics)

    holdout_df = pd.DataFrame(holdout_rows)

    print("\nHoldout Results")
    print(holdout_df.sort_values("model").to_string(index=False))

    cv_df = cross_validate_models(
        df=df,
        queues=queues,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )

    print("\nCross-Validation Results")
    print(cv_df.to_string(index=False))

    cv_summary = (
        cv_df.groupby("model", as_index=False)[
            ["top1_accuracy", "near_oracle_rate", "avg_regret", "speedup_vs_best_fixed"]
        ]
        .mean()
        .sort_values("model")
    )

    print("\nCross-Validation Means")
    print(cv_summary.to_string(index=False))

    out_dir = input_path.parent
    holdout_df.to_csv(out_dir / "proposal_holdout_results.csv", index=False)
    cv_df.to_csv(out_dir / "proposal_cv_results.csv", index=False)
    cv_summary.to_csv(out_dir / "proposal_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()