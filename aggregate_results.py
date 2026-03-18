#!/usr/bin/env python3
"""
aggregate_results.py

Aggregate raw run-level queue benchmark results into ML-ready datasets.

Input:
  CSV produced by run_experiments.py

Outputs:
  1. Repeat-aggregated per-queue dataset
  2. Pivoted ML dataset with one row per workload and one column per queue
  3. Optional summary CSV

Example:
  python aggregate_results.py --input outputs/queue_sweep_20260317_123456.csv

  python aggregate_results.py \
      --input outputs/pilot_20260317_123456.csv \
      --queues wfq,sfq \
      --agg median \
      --min-success-queues 2 \
      --out-prefix pilot_agg
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def maybe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def maybe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def safe_queue_name(row: Dict[str, Any]) -> str:
    q = (row.get("queue") or row.get("queue_label_expected") or "").strip().lower()
    return q


def safe_test_name(row: Dict[str, Any]) -> str:
    return str(row.get("test") or row.get("test_requested") or "").strip()


def safe_threads(row: Dict[str, Any]) -> Optional[int]:
    return maybe_int(row.get("threads", row.get("threads_requested")))


def safe_ops(row: Dict[str, Any]) -> Optional[int]:
    return maybe_int(row.get("ops_per_thread", row.get("ops_per_thread_requested")))


def safe_pr(row: Dict[str, Any]) -> Optional[int]:
    return maybe_int(row.get("producer_ratio", row.get("producer_ratio_requested")))


def safe_block(row: Dict[str, Any]) -> Optional[int]:
    return maybe_int(row.get("block_size", row.get("block_size_requested")))


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def agg_value(values: List[float], method: str) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    if method == "median":
        return float(statistics.median(vals))
    if method == "mean":
        return float(statistics.mean(vals))
    if method == "max":
        return float(max(vals))
    raise ValueError(f"Unknown aggregation method: {method}")


def group_successful_runs(
    rows: List[Dict[str, Any]],
    allowed_queues: Optional[set[str]],
) -> List[Dict[str, Any]]:
    kept = []
    for row in rows:
        queue = safe_queue_name(row)
        if allowed_queues is not None and queue not in allowed_queues:
            continue

        success = maybe_bool(row.get("success"))
        if success is not True:
            continue

        test = safe_test_name(row)
        threads = safe_threads(row)
        ops = safe_ops(row)
        pr = safe_pr(row)
        block = safe_block(row)

        thr = maybe_float(row.get("avg_throughput_mops"))
        elapsed = maybe_float(row.get("avg_elapsed_ms"))
        succ_ops = maybe_float(row.get("avg_successful_ops"))
        empty_deq = maybe_float(row.get("avg_empty_dequeues"))

        if not queue or not test:
            continue
        if threads is None or ops is None or pr is None or block is None:
            continue
        if thr is None:
            continue

        kept.append({
            "queue": queue,
            "test": test,
            "threads": threads,
            "ops_per_thread": ops,
            "producer_ratio": pr,
            "block_size": block,
            "avg_throughput_mops": thr,
            "avg_elapsed_ms": elapsed,
            "avg_successful_ops": succ_ops,
            "avg_empty_dequeues": empty_deq,
        })
    return kept


def aggregate_per_queue(
    rows: List[Dict[str, Any]],
    agg_method: str,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (
            row["queue"],
            row["test"],
            row["threads"],
            row["ops_per_thread"],
            row["producer_ratio"],
            row["block_size"],
        )
        groups[key].append(row)

    out: List[Dict[str, Any]] = []

    for key, items in sorted(groups.items()):
        queue, test, threads, ops, pr, block = key

        thr = agg_value([maybe_float(x["avg_throughput_mops"]) for x in items], agg_method)
        elapsed = agg_value([maybe_float(x["avg_elapsed_ms"]) for x in items], agg_method)
        succ_ops = agg_value([maybe_float(x["avg_successful_ops"]) for x in items], agg_method)
        empty_deq = agg_value([maybe_float(x["avg_empty_dequeues"]) for x in items], agg_method)

        out.append({
            "queue": queue,
            "test": test,
            "threads": threads,
            "ops_per_thread": ops,
            "producer_ratio": pr,
            "block_size": block,
            "num_runs_aggregated": len(items),
            "agg_method": agg_method,
            "throughput_mops": thr,
            "elapsed_ms": elapsed,
            "successful_ops": succ_ops,
            "empty_dequeues": empty_deq,
        })

    return out


def pivot_ml_dataset(
    agg_rows: List[Dict[str, Any]],
    queue_order: List[str],
    min_success_queues: int,
) -> List[Dict[str, Any]]:
    workload_groups: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for row in agg_rows:
        workload_key = (
            row["test"],
            row["threads"],
            row["ops_per_thread"],
            row["producer_ratio"],
            row["block_size"],
        )
        workload_groups[workload_key][row["queue"]] = row

    ml_rows: List[Dict[str, Any]] = []

    for workload_key, qmap in sorted(workload_groups.items()):
        test, threads, ops, pr, block = workload_key

        present_queues = [q for q in queue_order if q in qmap and qmap[q].get("throughput_mops") is not None]
        if len(present_queues) < min_success_queues:
            continue

        row: Dict[str, Any] = {
            "test": test,
            "threads": threads,
            "ops_per_thread": ops,
            "producer_ratio": pr,
            "block_size": block,
            "num_available_queues": len(present_queues),
        }

        best_queue = None
        best_thr = None

        for q in queue_order:
            qrow = qmap.get(q)
            thr = qrow.get("throughput_mops") if qrow else None
            elapsed = qrow.get("elapsed_ms") if qrow else None
            succ_ops = qrow.get("successful_ops") if qrow else None
            empty = qrow.get("empty_dequeues") if qrow else None
            nruns = qrow.get("num_runs_aggregated") if qrow else None

            row[f"{q}_throughput_mops"] = thr
            row[f"{q}_elapsed_ms"] = elapsed
            row[f"{q}_successful_ops"] = succ_ops
            row[f"{q}_empty_dequeues"] = empty
            row[f"{q}_num_runs"] = nruns

            if thr is not None and (best_thr is None or thr > best_thr):
                best_thr = thr
                best_queue = q

        row["best_queue"] = best_queue
        row["best_throughput_mops"] = best_thr

        if best_thr is not None:
            for q in queue_order:
                thr = row.get(f"{q}_throughput_mops")
                regret = None if thr is None else (best_thr - thr)
                near_oracle = None if thr is None else (thr / best_thr if best_thr > 0 else None)
                row[f"{q}_regret_mops"] = regret
                row[f"{q}_oracle_ratio"] = near_oracle

        ml_rows.append(row)

    return ml_rows


def build_summary(ml_rows: List[Dict[str, Any]], queue_order: List[str]) -> List[Dict[str, Any]]:
    counts = defaultdict(int)
    oracle_ratios = defaultdict(list)

    for row in ml_rows:
        best_q = row.get("best_queue")
        if best_q:
            counts[best_q] += 1

        for q in queue_order:
            v = maybe_float(row.get(f"{q}_oracle_ratio"))
            if v is not None:
                oracle_ratios[q].append(v)

    out = []
    total = len(ml_rows)

    for q in queue_order:
        wins = counts[q]
        avg_oracle_ratio = agg_value(oracle_ratios[q], "mean")
        out.append({
            "queue": q,
            "num_best": wins,
            "best_fraction": (wins / total) if total > 0 else None,
            "avg_oracle_ratio": avg_oracle_ratio,
            "num_workloads_seen": len(oracle_ratios[q]),
        })

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate queue benchmark results into ML datasets.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV from run_experiments.py",
    )
    parser.add_argument(
        "--queues",
        default="wfq,sfq",
        help="Comma-separated queue order for pivoted ML dataset",
    )
    parser.add_argument(
        "--agg",
        default="median",
        choices=["median", "mean", "max"],
        help="Aggregation method across repeated runs",
    )
    parser.add_argument(
        "--min-success-queues",
        type=int,
        default=2,
        help="Minimum number of queues present to keep a workload row in the ML dataset",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix (default: derived from input filename)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    queue_order = parse_csv_strings(args.queues)
    allowed_queues = set(queue_order)

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    default_prefix = input_path.stem + "_agg"
    out_prefix = args.out_prefix if args.out_prefix else default_prefix

    per_queue_path = out_dir / f"{out_prefix}_per_queue.csv"
    ml_path = out_dir / f"{out_prefix}_ml.csv"
    summary_path = out_dir / f"{out_prefix}_summary.csv"

    raw_rows = read_csv(input_path)
    kept_rows = group_successful_runs(raw_rows, allowed_queues)
    per_queue_rows = aggregate_per_queue(kept_rows, args.agg)
    ml_rows = pivot_ml_dataset(per_queue_rows, queue_order, args.min_success_queues)
    summary_rows = build_summary(ml_rows, queue_order)

    write_csv(per_queue_path, per_queue_rows)
    write_csv(ml_path, ml_rows)
    write_csv(summary_path, summary_rows)

    print(f"Input rows:                  {len(raw_rows)}")
    print(f"Successful filtered rows:    {len(kept_rows)}")
    print(f"Per-queue aggregated rows:   {len(per_queue_rows)}")
    print(f"ML workload rows:            {len(ml_rows)}")
    print(f"Wrote per-queue CSV:         {per_queue_path}")
    print(f"Wrote ML dataset CSV:        {ml_path}")
    print(f"Wrote summary CSV:           {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())