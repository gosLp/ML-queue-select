// queue_bench.cpp
//
// Unified benchmark driver for GPU queue variants.
// Compile one queue at a time:
//
//   hipcc -O3 -std=c++17 -DWFQ queue_bench.cpp -o queue_bench_wfq
//   hipcc -O3 -std=c++17 -DSFQ queue_bench.cpp sfqueue_hip.cpp -o queue_bench_sfq
//
// Example:
//   ./queue_bench_wfq --threads 4096 --ops 1024 --producer-ratio 50 --test balanced --block 256
//   ./queue_bench_sfq --threads 4096 --ops 1024 --producer-ratio 75 --test split_roles --block 256
//
// Expected companion header:
//   #include "queue_api.hpp"
//
// That header should define:
//   - queue_t / handle_t
//   - QUEUE_NAME
//   - queue_host_init(...)
//   - queue_reset(...)
//   - queue_destroy(...)
//   - bench_kernel(...)
//   - TEST_BALANCED / TEST_SPLIT_ROLES / TEST_BURST

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "queue_api.hpp"

struct BenchConfig {
    int threads = 1024;
    int ops_per_thread = 1024;
    int producer_ratio = 50;   // percent [0,100]
    int test_type = TEST_BALANCED;
    int block_size = 256;
    int repeats = 1;
    bool reset_between_repeats = true;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nOptions:\n"
        << "  --threads N            Total GPU threads (default: 1024)\n"
        << "  --ops N                Operations per thread (default: 1024)\n"
        << "  --producer-ratio N     Enqueue ratio percent [0..100] (default: 50)\n"
        << "  --test NAME            balanced | split_roles | burst (default: balanced)\n"
        << "  --block N              Threads per block (default: 256)\n"
        << "  --repeats N            Number of repeated runs (default: 1)\n"
        << "  --no-reset             Do not reset queue between repeats\n"
        << "  --help                 Show this message\n";
}

static bool parse_int_arg(const char* s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!s || *s == '\0' || (end && *end != '\0')) return false;
    if (v < INT32_MIN || v > INT32_MAX) return false;
    out = static_cast<int>(v);
    return true;
}

static int parse_test_type(const std::string& s) {
    if (s == "balanced") return TEST_BALANCED;
    if (s == "split_roles") return TEST_SPLIT_ROLES;
    if (s == "burst") return TEST_BURST;
    return -1;
}

static bool parse_args(int argc, char** argv, BenchConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--threads") {
            const char* v = need_value("--threads");
            if (!v || !parse_int_arg(v, cfg.threads) || cfg.threads <= 0) return false;
        } else if (arg == "--ops") {
            const char* v = need_value("--ops");
            if (!v || !parse_int_arg(v, cfg.ops_per_thread) || cfg.ops_per_thread <= 0) return false;
        } else if (arg == "--producer-ratio") {
            const char* v = need_value("--producer-ratio");
            if (!v || !parse_int_arg(v, cfg.producer_ratio) ||
                cfg.producer_ratio < 0 || cfg.producer_ratio > 100) {
                return false;
            }
        } else if (arg == "--test") {
            const char* v = need_value("--test");
            if (!v) return false;
            cfg.test_type = parse_test_type(v);
            if (cfg.test_type < 0) {
                std::cerr << "Unknown test type: " << v << "\n";
                return false;
            }
        } else if (arg == "--block") {
            const char* v = need_value("--block");
            if (!v || !parse_int_arg(v, cfg.block_size) || cfg.block_size <= 0) return false;
        } else if (arg == "--repeats") {
            const char* v = need_value("--repeats");
            if (!v || !parse_int_arg(v, cfg.repeats) || cfg.repeats <= 0) return false;
        } else if (arg == "--no-reset") {
            cfg.reset_between_repeats = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    return true;
}

static const char* test_type_name(int t) {
    switch (t) {
        case TEST_BALANCED:    return "balanced";
        case TEST_SPLIT_ROLES: return "split_roles";
        case TEST_BURST:       return "burst";
        default:               return "unknown";
    }
}

static inline void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::cerr << what << " failed: " << hipGetErrorString(err) << "\n";
        std::exit(1);
    }
}

struct RunResult {
    uint64_t successful_ops = 0;
    uint64_t empty_dequeues = 0;
    float elapsed_ms = 0.0f;
    double throughput_mops = 0.0;
    bool success = true;
};

static RunResult run_once(const BenchConfig& cfg,
                          queue_t* d_q,
                          handle_t* d_handles,
                          uint64_t* d_thread_counts,
                          uint64_t* d_empty_counts,
                          std::vector<uint64_t>& h_thread_counts,
                          std::vector<uint64_t>& h_empty_counts) {
    const int grid_size = (cfg.threads + cfg.block_size - 1) / cfg.block_size;

    check_hip(hipMemset(d_thread_counts, 0, sizeof(uint64_t) * cfg.threads), "hipMemset(thread_counts)");
    check_hip(hipMemset(d_empty_counts, 0, sizeof(uint64_t) * cfg.threads), "hipMemset(empty_counts)");

    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");

    check_hip(hipEventRecord(start), "hipEventRecord(start)");

    hipLaunchKernelGGL(
        bench_kernel,
        dim3(grid_size),
        dim3(cfg.block_size),
        0,
        0,
        d_q,
        d_handles,
        d_thread_counts,
        d_empty_counts,
        cfg.threads,
        cfg.ops_per_thread,
        cfg.producer_ratio,
        cfg.test_type
    );

    check_hip(hipGetLastError(), "bench_kernel launch");
    check_hip(hipEventRecord(stop), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");

    check_hip(hipMemcpy(h_thread_counts.data(),
                        d_thread_counts,
                        sizeof(uint64_t) * cfg.threads,
                        hipMemcpyDeviceToHost),
              "hipMemcpy(thread_counts)");

    check_hip(hipMemcpy(h_empty_counts.data(),
                        d_empty_counts,
                        sizeof(uint64_t) * cfg.threads,
                        hipMemcpyDeviceToHost),
              "hipMemcpy(empty_counts)");

    check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
    check_hip(hipEventDestroy(stop), "hipEventDestroy(stop)");

    RunResult rr;
    rr.elapsed_ms = elapsed_ms;

    for (int i = 0; i < cfg.threads; ++i) {
        rr.successful_ops += h_thread_counts[i];
        rr.empty_dequeues += h_empty_counts[i];
    }

    const double elapsed_sec = static_cast<double>(elapsed_ms) / 1000.0;
    rr.throughput_mops = (elapsed_sec > 0.0)
        ? (static_cast<double>(rr.successful_ops) / elapsed_sec) / 1.0e6
        : 0.0;

    return rr;
}

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    if (cfg.test_type == TEST_BURST && cfg.threads > cfg.block_size) {
        std::cerr
            << "Warning: TEST_BURST uses __syncthreads(), which is block-local.\n"
            << "For clean phase behavior, prefer threads <= block size for burst mode.\n";
    }

    queue_t* d_q = nullptr;
    handle_t* d_handles = nullptr;
    uint64_t* d_thread_counts = nullptr;
    uint64_t* d_empty_counts = nullptr;

    std::vector<uint64_t> h_thread_counts(cfg.threads, 0);
    std::vector<uint64_t> h_empty_counts(cfg.threads, 0);
    std::vector<RunResult> runs;
    runs.reserve(cfg.repeats);

    queue_host_init(&d_q, &d_handles, cfg.threads, cfg.ops_per_thread);

    check_hip(hipMalloc(reinterpret_cast<void**>(&d_thread_counts),
                        sizeof(uint64_t) * cfg.threads),
              "hipMalloc(thread_counts)");

    check_hip(hipMalloc(reinterpret_cast<void**>(&d_empty_counts),
                        sizeof(uint64_t) * cfg.threads),
              "hipMalloc(empty_counts)");

    for (int r = 0; r < cfg.repeats; ++r) {
        if (r > 0 && cfg.reset_between_repeats) {
            queue_reset(d_q, d_handles, cfg.threads);
        }

        RunResult rr = run_once(cfg,
                                d_q,
                                d_handles,
                                d_thread_counts,
                                d_empty_counts,
                                h_thread_counts,
                                h_empty_counts);
        runs.push_back(rr);

        std::cerr
            << "[run " << r
            << "] elapsed_ms=" << rr.elapsed_ms
            << " successful_ops=" << rr.successful_ops
            << " empty_dequeues=" << rr.empty_dequeues
            << " throughput_mops=" << rr.throughput_mops
            << "\n";
    }

    uint64_t total_successful_ops = 0;
    uint64_t total_empty_dequeues = 0;
    double total_elapsed_ms = 0.0;
    double best_throughput_mops = 0.0;
    double worst_throughput_mops = 0.0;

    if (!runs.empty()) {
        best_throughput_mops = runs.front().throughput_mops;
        worst_throughput_mops = runs.front().throughput_mops;
    }

    for (const auto& rr : runs) {
        total_successful_ops += rr.successful_ops;
        total_empty_dequeues += rr.empty_dequeues;
        total_elapsed_ms += rr.elapsed_ms;
        best_throughput_mops = std::max(best_throughput_mops, rr.throughput_mops);
        worst_throughput_mops = std::min(worst_throughput_mops, rr.throughput_mops);
    }

    const double avg_elapsed_ms = runs.empty() ? 0.0 : total_elapsed_ms / static_cast<double>(runs.size());
    const double avg_successful_ops = runs.empty() ? 0.0 : static_cast<double>(total_successful_ops) / static_cast<double>(runs.size());
    const double avg_empty_dequeues = runs.empty() ? 0.0 : static_cast<double>(total_empty_dequeues) / static_cast<double>(runs.size());
    const double avg_throughput_mops =
        (avg_elapsed_ms > 0.0)
            ? (avg_successful_ops / (avg_elapsed_ms / 1000.0)) / 1.0e6
            : 0.0;

    // Print one machine-readable JSON line as final output.
    std::cout
        << "{"
        << "\"queue\":\"" << QUEUE_NAME << "\","
        << "\"test\":\"" << test_type_name(cfg.test_type) << "\","
        << "\"threads\":" << cfg.threads << ","
        << "\"ops_per_thread\":" << cfg.ops_per_thread << ","
        << "\"producer_ratio\":" << cfg.producer_ratio << ","
        << "\"block_size\":" << cfg.block_size << ","
        << "\"repeats\":" << cfg.repeats << ","
        << "\"reset_between_repeats\":" << (cfg.reset_between_repeats ? "true" : "false") << ","
        << "\"avg_successful_ops\":" << avg_successful_ops << ","
        << "\"avg_empty_dequeues\":" << avg_empty_dequeues << ","
        << "\"avg_elapsed_ms\":" << avg_elapsed_ms << ","
        << "\"avg_throughput_mops\":" << avg_throughput_mops << ","
        << "\"best_throughput_mops\":" << best_throughput_mops << ","
        << "\"worst_throughput_mops\":" << worst_throughput_mops << ","
        << "\"success\":true"
        << "}"
        << "\n";

    if (d_thread_counts) hipFree(d_thread_counts);
    if (d_empty_counts) hipFree(d_empty_counts);
    queue_destroy(d_q, d_handles);

    return 0;
}