# ML-Guided GPU Queue Selection Benchmark

This project benchmarks multiple GPU concurrent queue implementations and builds a dataset for **machine learning-based selection of the best queue** for a given workload.

Supported queues:
- **WFQ** (Wait-Free Queue)
- **SFQ** (Scogland-Feng Queue)
- **Broker Queue (BQ)**

The system provides:
1. A unified GPU benchmark harness
2. A Python experiment runner
3. A dataset aggregation pipeline for ML training

---

# Quick Start

## 1. Requirements

### Hardware
- AMD GPU (ROCm) **or**
- NVIDIA GPU (via HIP / CUDA compatibility)

### Software
- ROCm (recommended ≥ 5.0)
- `hipcc`
- Python 3.8+
- Python packages:
  ```bash
  pip install numpy

# Build Benchmarks

```
mkdir -p out/
```

compile each separately using `hipcc`

## WFQ 

```
hipcc -O3 -std=c++17 -DWFQ queue_bench.cpp -o out/queue_bench_wfq
```

## SFQ 

```
hipcc -O3 -std=c++17 -DSFQ queue_bench.cpp -o out/queue_bench_sfq
```

## BROKER 

```
hipcc -O3 -std=c++17 -DBROKER queue_bench.cpp -o out/queue_bench_broker
```

# Running a Single Benchmark

Example:

```
./queue_bench_wfq \
  --threads 1024 \
  --ops 256 \
  --producer-ratio 50 \
  --test balanced \
  --block 256
```

| Argument           | Description                        |
| ------------------ | ---------------------------------- |
| `--threads`        | Total GPU threads                  |
| `--ops`            | Operations per thread              |
| `--producer-ratio` | % enqueue vs dequeue               |
| `--test`           | `balanced`, `split_roles`, `burst` |
| `--block`          | Threads per block                  |


# Output
Each run prints a final json line:
```
{
  "queue": "wfq",
  "threads": 1024,
  "avg_throughput_mops": 123.45,
  ...
}
```
This is what the Python pipeline consumes

# Running Full Experiments
Use the experiments harness:
```
python run_experiments.py \
  --binaries ./out/queue_bench_wfq,./out/queue_bench_sfq,./out/queue_bench_broker
```

## Example : Small Pilot Run
```
python run_experiments.py \
  --binaries ./queue_bench_wfq,./queue_bench_sfq,./queue_bench_broker \
  --threads 64,256,1024 \
  --ops 64,256 \
  --producer-ratios 25,50,75 \
  --tests balanced,split_roles \
  --repeats 3 \
  --out-prefix pilot
```

# Output Files
Generated in outputs/
-- *.jsonl -> raw run logs
-- *.csv -> structured dataset (per run)


# Aggregated Results
convert raw runs into ML-ready dataset:

```
python aggregate_results.py \
  --input outputs/pilot_YYYYMMDD_HHMMSS.csv \
  --queues wfq,sfq,broker \
  --agg median \
  --min-success-queues 3 \
  --out-prefix pilot
```



# Project Structure 
.
├── queue_bench.cpp          # GPU benchmark harness
├── queue_api.hpp            # Unified queue interface
├── queues/
│   ├── wfq.hpp
│   ├── sfq.hpp / sfq.cpp
├── broker_queue_hip.hpp     # Broker wrapper
├── run_experiments.py       # Sweep runner
├── aggregate_results.py     # Dataset builder
└── outputs/