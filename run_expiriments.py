#!/usr/bin/env python3
"""
run_experiments.py

Sweep queue benchmark configurations and collect structured results.

Expected binaries:
  ./queue_bench_wfq
  ./queue_bench_sfq

Each binary should print one final JSON line like:
  {"queue":"wfq", ... }

Example:
  python run_experiments.py

  python run_experiments.py \
      --threads 64,256,1024,4096 \
      --ops 64,256,1024 \
      --producer-ratios 0,25,50,75,100 \
      --tests balanced,split_roles \
      --repeats 3 \
      --out-prefix pilot
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ExperimentSpec:
    binary: str
    queue_label: str
    threads: int
    ops_per_thread: int
    producer_ratio: int
    test: str
    block_size: int
    repeat_id: int


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_command(spec: ExperimentSpec) -> List[str]:
    return [
        spec.binary,
        "--threads", str(spec.threads),
        "--ops", str(spec.ops_per_thread),
        "--producer-ratio", str(spec.producer_ratio),
        "--test", spec.test,
        "--block", str(spec.block_size),
        "--repeats", "1",
    ]


def extract_last_json_line(stdout: str) -> Optional[Dict[str, Any]]:
    """
    Find the last line in stdout that parses as JSON object.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def run_one(spec: ExperimentSpec, timeout_sec: int) -> Dict[str, Any]:
    cmd = build_command(spec)
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        wall_time_sec = time.time() - t0

        parsed = extract_last_json_line(proc.stdout)

        row: Dict[str, Any] = {
            "binary": spec.binary,
            "queue_label_expected": spec.queue_label,
            "threads_requested": spec.threads,
            "ops_per_thread_requested": spec.ops_per_thread,
            "producer_ratio_requested": spec.producer_ratio,
            "test_requested": spec.test,
            "block_size_requested": spec.block_size,
            "repeat_id": spec.repeat_id,
            "command": " ".join(cmd),
            "return_code": proc.returncode,
            "timed_out": False,
            "wall_time_sec": wall_time_sec,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "parse_ok": parsed is not None,
        }

        if parsed is not None:
            row.update(parsed)
            row["success"] = bool(parsed.get("success", False)) and proc.returncode == 0
        else:
            row["success"] = False
            row["parse_error"] = "No JSON object found in stdout"

        return row

    except subprocess.TimeoutExpired as e:
        wall_time_sec = time.time() - t0
        return {
            "binary": spec.binary,
            "queue_label_expected": spec.queue_label,
            "threads_requested": spec.threads,
            "ops_per_thread_requested": spec.ops_per_thread,
            "producer_ratio_requested": spec.producer_ratio,
            "test_requested": spec.test,
            "block_size_requested": spec.block_size,
            "repeat_id": spec.repeat_id,
            "command": " ".join(cmd),
            "return_code": None,
            "timed_out": True,
            "wall_time_sec": wall_time_sec,
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else "",
            "parse_ok": False,
            "success": False,
            "parse_error": f"Timed out after {timeout_sec}s",
        }


def generate_specs(
    binaries: List[str],
    threads: List[int],
    ops: List[int],
    producer_ratios: List[int],
    tests: List[str],
    block_size: int,
    repeats: int,
) -> Iterable[ExperimentSpec]:
    for binary, th, op, pr, test, rep in itertools.product(
        binaries, threads, ops, producer_ratios, tests, range(repeats)
    ):
        queue_label = infer_queue_label(binary)
        yield ExperimentSpec(
            binary=binary,
            queue_label=queue_label,
            threads=th,
            ops_per_thread=op,
            producer_ratio=pr,
            test=test,
            block_size=block_size,
            repeat_id=rep,
        )


def infer_queue_label(binary: str) -> str:
    name = Path(binary).name.lower()
    if "wfq" in name:
        return "wfq"
    if "sfq" in name:
        return "sfq"
    if "broker" in name or "bq" in name:
        return "broker"
    return "unknown"


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compact_row_for_console(row: Dict[str, Any]) -> str:
    queue = row.get("queue", row.get("queue_label_expected", "?"))
    test = row.get("test", row.get("test_requested", "?"))
    threads = row.get("threads", row.get("threads_requested", "?"))
    ops = row.get("ops_per_thread", row.get("ops_per_thread_requested", "?"))
    pr = row.get("producer_ratio", row.get("producer_ratio_requested", "?"))
    thr = row.get("avg_throughput_mops", None)
    success = row.get("success", False)

    thr_s = f"{thr:.3f}" if isinstance(thr, (int, float)) else "NA"
    return (
        f"[{'OK' if success else 'FAIL'}] "
        f"queue={queue} test={test} threads={threads} ops={ops} "
        f"pr={pr} thr={thr_s} Mops/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU queue benchmark sweeps.")
    parser.add_argument(
        "--binaries",
        default="./queue_bench_wfq,./queue_bench_sfq",
        help="Comma-separated benchmark binaries",
    )
    parser.add_argument(
        "--threads",
        default="64,256,1024,4096",
        help="Comma-separated thread counts",
    )
    parser.add_argument(
        "--ops",
        default="64,256,1024",
        help="Comma-separated ops-per-thread values",
    )
    parser.add_argument(
        "--producer-ratios",
        default="0,25,50,75,100",
        help="Comma-separated producer ratio percentages",
    )
    parser.add_argument(
        "--tests",
        default="balanced,split_roles",
        help="Comma-separated test names",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size to pass to benchmark",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of run-level repeats per configuration",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-run timeout in seconds",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--out-prefix",
        default="queue_sweep",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop immediately on first failed run",
    )

    args = parser.parse_args()

    binaries = parse_csv_strings(args.binaries)
    threads = parse_csv_ints(args.threads)
    ops = parse_csv_ints(args.ops)
    producer_ratios = parse_csv_ints(args.producer_ratios)
    tests = parse_csv_strings(args.tests)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"{args.out_prefix}_{ts}.jsonl"
    csv_path = out_dir / f"{args.out_prefix}_{ts}.csv"

    rows: List[Dict[str, Any]] = []

    specs = list(generate_specs(
        binaries=binaries,
        threads=threads,
        ops=ops,
        producer_ratios=producer_ratios,
        tests=tests,
        block_size=args.block_size,
        repeats=args.repeats,
    ))

    print(f"Running {len(specs)} experiments...")
    print(f"JSONL output: {jsonl_path}")
    print(f"CSV output:   {csv_path}")

    for idx, spec in enumerate(specs, start=1):
        print(
            f"[{idx}/{len(specs)}] "
            f"{spec.queue_label} {spec.test} "
            f"threads={spec.threads} ops={spec.ops_per_thread} "
            f"pr={spec.producer_ratio} rep={spec.repeat_id}"
        )

        row = run_one(spec, timeout_sec=args.timeout)
        rows.append(row)
        print("   ", compact_row_for_console(row))

        if args.stop_on_fail and not row.get("success", False):
            print("Stopping on first failure as requested.")
            break

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    ok = sum(1 for r in rows if r.get("success", False))
    fail = len(rows) - ok

    print()
    print(f"Done. Successful runs: {ok}, failed runs: {fail}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())