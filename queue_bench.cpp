// // queue_bench.cpp
// //
// // Unified benchmark driver for GPU queue variants.
// // Compile one queue at a time:
// //
// //   hipcc -O3 -std=c++17 -DWFQ queue_bench.cpp -o queue_bench_wfq
// //   hipcc -O3 -std=c++17 -DSFQ queue_bench.cpp sfqueue_hip.cpp -o queue_bench_sfq
// //
// // Example:
// //   ./queue_bench_wfq --threads 4096 --ops 1024 --producer-ratio 50 --test balanced --block 256
// //   ./queue_bench_sfq --threads 4096 --ops 1024 --producer-ratio 75 --test split_roles --block 256
// //
// // Expected companion header:
// //   #include "queue_api.hpp"
// //
// // That header should define:
// //   - queue_t / handle_t
// //   - QUEUE_NAME
// //   - queue_host_init(...)
// //   - queue_reset(...)
// //   - queue_destroy(...)
// //   - bench_kernel(...)
// //   - TEST_BALANCED / TEST_SPLIT_ROLES / TEST_BURST

// #include <hip/hip_runtime.h>

// #include <algorithm>
// #include <cstdint>
// #include <cstdlib>
// #include <cstring>
// #include <iostream>
// #include <string>
// #include <vector>

// #include "queue_api.hpp"

// struct BenchConfig {
//     int threads = 1024;
//     int ops_per_thread = 1024;
//     int producer_ratio = 50;   // percent [0,100]
//     int test_type = TEST_BALANCED;
//     int block_size = 256;
//     int repeats = 1;
//     bool reset_between_repeats = true;
// };

// static void print_usage(const char* prog) {
//     std::cerr
//         << "Usage: " << prog << " [options]\n"
//         << "\nOptions:\n"
//         << "  --threads N            Total GPU threads (default: 1024)\n"
//         << "  --ops N                Operations per thread (default: 1024)\n"
//         << "  --producer-ratio N     Enqueue ratio percent [0..100] (default: 50)\n"
//         << "  --test NAME            balanced | split_roles | burst (default: balanced)\n"
//         << "  --block N              Threads per block (default: 256)\n"
//         << "  --repeats N            Number of repeated runs (default: 1)\n"
//         << "  --no-reset             Do not reset queue between repeats\n"
//         << "  --help                 Show this message\n";
// }

// static bool parse_int_arg(const char* s, int& out) {
//     char* end = nullptr;
//     long v = std::strtol(s, &end, 10);
//     if (!s || *s == '\0' || (end && *end != '\0')) return false;
//     if (v < INT32_MIN || v > INT32_MAX) return false;
//     out = static_cast<int>(v);
//     return true;
// }

// static int parse_test_type(const std::string& s) {
//     if (s == "balanced") return TEST_BALANCED;
//     if (s == "split_roles") return TEST_SPLIT_ROLES;
//     if (s == "burst") return TEST_BURST;
//     return -1;
// }

// static bool parse_args(int argc, char** argv, BenchConfig& cfg) {
//     for (int i = 1; i < argc; ++i) {
//         const std::string arg = argv[i];

//         auto need_value = [&](const char* name) -> const char* {
//             if (i + 1 >= argc) {
//                 std::cerr << "Missing value for " << name << "\n";
//                 return nullptr;
//             }
//             return argv[++i];
//         };

//         if (arg == "--help" || arg == "-h") {
//             print_usage(argv[0]);
//             std::exit(0);
//         } else if (arg == "--threads") {
//             const char* v = need_value("--threads");
//             if (!v || !parse_int_arg(v, cfg.threads) || cfg.threads <= 0) return false;
//         } else if (arg == "--ops") {
//             const char* v = need_value("--ops");
//             if (!v || !parse_int_arg(v, cfg.ops_per_thread) || cfg.ops_per_thread <= 0) return false;
//         } else if (arg == "--producer-ratio") {
//             const char* v = need_value("--producer-ratio");
//             if (!v || !parse_int_arg(v, cfg.producer_ratio) ||
//                 cfg.producer_ratio < 0 || cfg.producer_ratio > 100) {
//                 return false;
//             }
//         } else if (arg == "--test") {
//             const char* v = need_value("--test");
//             if (!v) return false;
//             cfg.test_type = parse_test_type(v);
//             if (cfg.test_type < 0) {
//                 std::cerr << "Unknown test type: " << v << "\n";
//                 return false;
//             }
//         } else if (arg == "--block") {
//             const char* v = need_value("--block");
//             if (!v || !parse_int_arg(v, cfg.block_size) || cfg.block_size <= 0) return false;
//         } else if (arg == "--repeats") {
//             const char* v = need_value("--repeats");
//             if (!v || !parse_int_arg(v, cfg.repeats) || cfg.repeats <= 0) return false;
//         } else if (arg == "--no-reset") {
//             cfg.reset_between_repeats = false;
//         } else {
//             std::cerr << "Unknown argument: " << arg << "\n";
//             return false;
//         }
//     }

//     return true;
// }

// static const char* test_type_name(int t) {
//     switch (t) {
//         case TEST_BALANCED:    return "balanced";
//         case TEST_SPLIT_ROLES: return "split_roles";
//         case TEST_BURST:       return "burst";
//         default:               return "unknown";
//     }
// }

// static inline void check_hip(hipError_t err, const char* what) {
//     if (err != hipSuccess) {
//         std::cerr << what << " failed: " << hipGetErrorString(err) << "\n";
//         std::exit(1);
//     }
// }

// struct RunResult {
//     uint64_t successful_ops = 0;
//     uint64_t empty_dequeues = 0;
//     float elapsed_ms = 0.0f;
//     double throughput_mops = 0.0;
//     bool success = true;
// };

// static RunResult run_once(const BenchConfig& cfg,
//                           queue_t* d_q,
//                           handle_t* d_handles,
//                           uint64_t* d_thread_counts,
//                           uint64_t* d_empty_counts,
//                           std::vector<uint64_t>& h_thread_counts,
//                           std::vector<uint64_t>& h_empty_counts) {
//     const int grid_size = (cfg.threads + cfg.block_size - 1) / cfg.block_size;

//     check_hip(hipMemset(d_thread_counts, 0, sizeof(uint64_t) * cfg.threads), "hipMemset(thread_counts)");
//     check_hip(hipMemset(d_empty_counts, 0, sizeof(uint64_t) * cfg.threads), "hipMemset(empty_counts)");

//     hipEvent_t start, stop;
//     check_hip(hipEventCreate(&start), "hipEventCreate(start)");
//     check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");

//     check_hip(hipEventRecord(start), "hipEventRecord(start)");

//     hipLaunchKernelGGL(
//         bench_kernel,
//         dim3(grid_size),
//         dim3(cfg.block_size),
//         0,
//         0,
//         d_q,
//         d_handles,
//         d_thread_counts,
//         d_empty_counts,
//         cfg.threads,
//         cfg.ops_per_thread,
//         cfg.producer_ratio,
//         cfg.test_type
//     );

//     check_hip(hipGetLastError(), "bench_kernel launch");
//     check_hip(hipEventRecord(stop), "hipEventRecord(stop)");
//     check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");

//     float elapsed_ms = 0.0f;
//     check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");

//     check_hip(hipMemcpy(h_thread_counts.data(),
//                         d_thread_counts,
//                         sizeof(uint64_t) * cfg.threads,
//                         hipMemcpyDeviceToHost),
//               "hipMemcpy(thread_counts)");

//     check_hip(hipMemcpy(h_empty_counts.data(),
//                         d_empty_counts,
//                         sizeof(uint64_t) * cfg.threads,
//                         hipMemcpyDeviceToHost),
//               "hipMemcpy(empty_counts)");

//     check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
//     check_hip(hipEventDestroy(stop), "hipEventDestroy(stop)");

//     RunResult rr;
//     rr.elapsed_ms = elapsed_ms;

//     for (int i = 0; i < cfg.threads; ++i) {
//         rr.successful_ops += h_thread_counts[i];
//         rr.empty_dequeues += h_empty_counts[i];
//     }

//     const double elapsed_sec = static_cast<double>(elapsed_ms) / 1000.0;
//     rr.throughput_mops = (elapsed_sec > 0.0)
//         ? (static_cast<double>(rr.successful_ops) / elapsed_sec) / 1.0e6
//         : 0.0;

//     return rr;
// }

// int main(int argc, char** argv) {
//     BenchConfig cfg;
//     if (!parse_args(argc, argv, cfg)) {
//         print_usage(argv[0]);
//         return 1;
//     }

//     if (cfg.test_type == TEST_BURST && cfg.threads > cfg.block_size) {
//         std::cerr
//             << "Warning: TEST_BURST uses __syncthreads(), which is block-local.\n"
//             << "For clean phase behavior, prefer threads <= block size for burst mode.\n";
//     }

//     queue_t* d_q = nullptr;
//     handle_t* d_handles = nullptr;
//     uint64_t* d_thread_counts = nullptr;
//     uint64_t* d_empty_counts = nullptr;

//     std::vector<uint64_t> h_thread_counts(cfg.threads, 0);
//     std::vector<uint64_t> h_empty_counts(cfg.threads, 0);
//     std::vector<RunResult> runs;
//     runs.reserve(cfg.repeats);

//     queue_host_init(&d_q, &d_handles, cfg.threads, cfg.ops_per_thread);

//     check_hip(hipMalloc(reinterpret_cast<void**>(&d_thread_counts),
//                         sizeof(uint64_t) * cfg.threads),
//               "hipMalloc(thread_counts)");

//     check_hip(hipMalloc(reinterpret_cast<void**>(&d_empty_counts),
//                         sizeof(uint64_t) * cfg.threads),
//               "hipMalloc(empty_counts)");

//     for (int r = 0; r < cfg.repeats; ++r) {
//         if (r > 0 && cfg.reset_between_repeats) {
//             queue_reset(d_q, d_handles, cfg.threads);
//         }

//         RunResult rr = run_once(cfg,
//                                 d_q,
//                                 d_handles,
//                                 d_thread_counts,
//                                 d_empty_counts,
//                                 h_thread_counts,
//                                 h_empty_counts);
//         runs.push_back(rr);

//         std::cerr
//             << "[run " << r
//             << "] elapsed_ms=" << rr.elapsed_ms
//             << " successful_ops=" << rr.successful_ops
//             << " empty_dequeues=" << rr.empty_dequeues
//             << " throughput_mops=" << rr.throughput_mops
//             << "\n";
//     }

//     uint64_t total_successful_ops = 0;
//     uint64_t total_empty_dequeues = 0;
//     double total_elapsed_ms = 0.0;
//     double best_throughput_mops = 0.0;
//     double worst_throughput_mops = 0.0;

//     if (!runs.empty()) {
//         best_throughput_mops = runs.front().throughput_mops;
//         worst_throughput_mops = runs.front().throughput_mops;
//     }

//     for (const auto& rr : runs) {
//         total_successful_ops += rr.successful_ops;
//         total_empty_dequeues += rr.empty_dequeues;
//         total_elapsed_ms += rr.elapsed_ms;
//         best_throughput_mops = std::max(best_throughput_mops, rr.throughput_mops);
//         worst_throughput_mops = std::min(worst_throughput_mops, rr.throughput_mops);
//     }

//     const double avg_elapsed_ms = runs.empty() ? 0.0 : total_elapsed_ms / static_cast<double>(runs.size());
//     const double avg_successful_ops = runs.empty() ? 0.0 : static_cast<double>(total_successful_ops) / static_cast<double>(runs.size());
//     const double avg_empty_dequeues = runs.empty() ? 0.0 : static_cast<double>(total_empty_dequeues) / static_cast<double>(runs.size());
//     const double avg_throughput_mops =
//         (avg_elapsed_ms > 0.0)
//             ? (avg_successful_ops / (avg_elapsed_ms / 1000.0)) / 1.0e6
//             : 0.0;

//     // Print one machine-readable JSON line as final output.
//     std::cout
//         << "{"
//         << "\"queue\":\"" << QUEUE_NAME << "\","
//         << "\"test\":\"" << test_type_name(cfg.test_type) << "\","
//         << "\"threads\":" << cfg.threads << ","
//         << "\"ops_per_thread\":" << cfg.ops_per_thread << ","
//         << "\"producer_ratio\":" << cfg.producer_ratio << ","
//         << "\"block_size\":" << cfg.block_size << ","
//         << "\"repeats\":" << cfg.repeats << ","
//         << "\"reset_between_repeats\":" << (cfg.reset_between_repeats ? "true" : "false") << ","
//         << "\"avg_successful_ops\":" << avg_successful_ops << ","
//         << "\"avg_empty_dequeues\":" << avg_empty_dequeues << ","
//         << "\"avg_elapsed_ms\":" << avg_elapsed_ms << ","
//         << "\"avg_throughput_mops\":" << avg_throughput_mops << ","
//         << "\"best_throughput_mops\":" << best_throughput_mops << ","
//         << "\"worst_throughput_mops\":" << worst_throughput_mops << ","
//         << "\"success\":true"
//         << "}"
//         << "\n";

//     if (d_thread_counts) hipFree(d_thread_counts);
//     if (d_empty_counts) hipFree(d_empty_counts);
//     queue_destroy(d_q, d_handles);

//     return 0;
// }
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "queue_api.hpp"

#ifndef DEFAULT_RUN_MS
#define DEFAULT_RUN_MS 500
#endif

#ifndef DEFAULT_WARMUP_MS
#define DEFAULT_WARMUP_MS 100
#endif

struct BenchConfig {
    int threads = 1024;
    int chunk_ops = 64;
    int producer_ratio = 50;
    int test_type = TEST_BALANCED;
    int block_size = 256;
    int repeats = 1;
    int run_ms = DEFAULT_RUN_MS;
    int warmup_ms = DEFAULT_WARMUP_MS;
    bool reset_between_repeats = true;
};

static inline void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::cerr << what << " failed: " << hipGetErrorString(err) << "\n";
        std::exit(1);
    }
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\nOptions:\n"
        << "  --threads N            Total GPU threads default 1024\n"
        << "  --ops N                Chunk operations per active thread default 64\n"
        << "  --producer-ratio N     Producer ratio percent [0..100] default 50\n"
        << "  --test NAME            balanced | split_roles | burst\n"
        << "  --block N              Threads per block default 256\n"
        << "  --repeats N            Repeats default 1\n"
        << "  --run-ms N             Timed window default " << DEFAULT_RUN_MS << "\n"
        << "  --warmup-ms N          Warmup window default " << DEFAULT_WARMUP_MS << "\n"
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
    if (s == "balanced")    return TEST_BALANCED;
    if (s == "split_roles") return TEST_SPLIT_ROLES;
    if (s == "burst")       return TEST_BURST;
    return -1;
}

static const char* test_type_name(int t) {
    switch (t) {
        case TEST_BALANCED:    return "balanced";
        case TEST_SPLIT_ROLES: return "split_roles";
        case TEST_BURST:       return "burst";
        default:               return "unknown";
    }
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
            if (!v || !parse_int_arg(v, cfg.chunk_ops) || cfg.chunk_ops <= 0) return false;
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
        } else if (arg == "--run-ms") {
            const char* v = need_value("--run-ms");
            if (!v || !parse_int_arg(v, cfg.run_ms) || cfg.run_ms <= 0) return false;
        } else if (arg == "--warmup-ms") {
            const char* v = need_value("--warmup-ms");
            if (!v || !parse_int_arg(v, cfg.warmup_ms) || cfg.warmup_ms < 0) return false;
        } else if (arg == "--no-reset") {
            cfg.reset_between_repeats = false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    return true;
}

static inline void reset_stats() {
    unsigned long long zero = 0;
    check_hip(hipMemcpyToSymbol(HIP_SYMBOL(g_enq_success), &zero, sizeof(zero)),
              "hipMemcpyToSymbol(g_enq_success)");
    check_hip(hipMemcpyToSymbol(HIP_SYMBOL(g_enq_fail), &zero, sizeof(zero)),
              "hipMemcpyToSymbol(g_enq_fail)");
    check_hip(hipMemcpyToSymbol(HIP_SYMBOL(g_deq_success), &zero, sizeof(zero)),
              "hipMemcpyToSymbol(g_deq_success)");
    check_hip(hipMemcpyToSymbol(HIP_SYMBOL(g_deq_empty), &zero, sizeof(zero)),
              "hipMemcpyToSymbol(g_deq_empty)");
}

static inline Stats fetch_stats() {
    Stats s;

    check_hip(hipMemcpyFromSymbol(&s.enq_success,
                                  HIP_SYMBOL(g_enq_success),
                                  sizeof(s.enq_success)),
              "hipMemcpyFromSymbol(g_enq_success)");

    check_hip(hipMemcpyFromSymbol(&s.enq_fail,
                                  HIP_SYMBOL(g_enq_fail),
                                  sizeof(s.enq_fail)),
              "hipMemcpyFromSymbol(g_enq_fail)");

    check_hip(hipMemcpyFromSymbol(&s.deq_success,
                                  HIP_SYMBOL(g_deq_success),
                                  sizeof(s.deq_success)),
              "hipMemcpyFromSymbol(g_deq_success)");

    check_hip(hipMemcpyFromSymbol(&s.deq_empty,
                                  HIP_SYMBOL(g_deq_empty),
                                  sizeof(s.deq_empty)),
              "hipMemcpyFromSymbol(g_deq_empty)");

    return s;
}

struct QueueInstance {
    queue_t* d_q = nullptr;
    handle_t* d_h = nullptr;
};

static QueueInstance create_queue(int threads, int chunk_ops) {
    QueueInstance qi{};
    queue_host_init(&qi.d_q, &qi.d_h, threads, chunk_ops);
    check_hip(hipDeviceSynchronize(), "queue_host_init synchronize");
    return qi;
}

static void destroy_queue(QueueInstance& qi) {
    queue_destroy(qi.d_q, qi.d_h);
    qi.d_q = nullptr;
    qi.d_h = nullptr;
}

static void launch_one_chunk(const BenchConfig& cfg,
                             QueueInstance& qi,
                             int grid,
                             int block) {
    switch (cfg.test_type) {
        case TEST_BALANCED:
            balanced_chunk_kernel<<<grid, block>>>(
                qi.d_q,
                qi.d_h,
                cfg.threads,
                cfg.chunk_ops
            );
            break;

        case TEST_SPLIT_ROLES:
            split_chunk_kernel<<<grid, block>>>(
                qi.d_q,
                qi.d_h,
                cfg.threads,
                cfg.chunk_ops,
                cfg.producer_ratio
            );
            break;

        case TEST_BURST:
            burst_chunk_kernel<<<grid, block>>>(
                qi.d_q,
                qi.d_h,
                cfg.threads,
                cfg.chunk_ops
            );
            break;

        default:
            balanced_chunk_kernel<<<grid, block>>>(
                qi.d_q,
                qi.d_h,
                cfg.threads,
                cfg.chunk_ops
            );
            break;
    }

    check_hip(hipGetLastError(), "chunk kernel launch");
}

struct RunResult {
    Stats stats;
    float elapsed_ms = 0.0f;
    int chunks_launched = 0;

    unsigned long long successful_ops = 0;
    unsigned long long attempted_ops = 0;

    double succ_mops = 0.0;
    double attempted_mops = 0.0;
    double enq_mops = 0.0;
    double deq_mops = 0.0;
    double fail_mops = 0.0;
    double empty_mops = 0.0;
};

static RunResult run_once(const BenchConfig& cfg,
                          QueueInstance& qi) {
    const int block = cfg.block_size;
    const int grid = (cfg.threads + block - 1) / block;

    // Warmup: same as check_gwf style, but stats ignored.
    if (cfg.warmup_ms > 0) {
        auto warm_t0 = std::chrono::high_resolution_clock::now();

        while (true) {
            launch_one_chunk(cfg, qi, grid, block);
            check_hip(hipDeviceSynchronize(), "warmup chunk synchronize");

            auto warm_t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(warm_t1 - warm_t0).count();

            if (ms >= static_cast<double>(cfg.warmup_ms)) break;
        }
    }

    reset_stats();

    hipEvent_t ev_start, ev_stop;
    check_hip(hipEventCreate(&ev_start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&ev_stop), "hipEventCreate(stop)");

    check_hip(hipEventRecord(ev_start, 0), "hipEventRecord(start)");

    auto t0 = std::chrono::high_resolution_clock::now();
    int chunks = 0;

    while (true) {
        launch_one_chunk(cfg, qi, grid, block);
        check_hip(hipDeviceSynchronize(), "timed chunk synchronize");
        ++chunks;

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (ms >= static_cast<double>(cfg.run_ms)) break;
    }

    check_hip(hipEventRecord(ev_stop, 0), "hipEventRecord(stop)");
    check_hip(hipEventSynchronize(ev_stop), "hipEventSynchronize(stop)");

    float gpu_ms = 0.0f;
    check_hip(hipEventElapsedTime(&gpu_ms, ev_start, ev_stop), "hipEventElapsedTime");

    check_hip(hipEventDestroy(ev_start), "hipEventDestroy(start)");
    check_hip(hipEventDestroy(ev_stop), "hipEventDestroy(stop)");

    RunResult rr;
    rr.elapsed_ms = gpu_ms;
    rr.chunks_launched = chunks;
    rr.stats = fetch_stats();

    rr.successful_ops = rr.stats.enq_success + rr.stats.deq_success;
    rr.attempted_ops =
        rr.stats.enq_success +
        rr.stats.enq_fail +
        rr.stats.deq_success +
        rr.stats.deq_empty;

    const double sec = static_cast<double>(gpu_ms) / 1000.0;

    if (sec > 0.0) {
        rr.succ_mops      = static_cast<double>(rr.successful_ops) / sec / 1e6;
        rr.attempted_mops = static_cast<double>(rr.attempted_ops) / sec / 1e6;
        rr.enq_mops       = static_cast<double>(rr.stats.enq_success) / sec / 1e6;
        rr.deq_mops       = static_cast<double>(rr.stats.deq_success) / sec / 1e6;
        rr.fail_mops      = static_cast<double>(rr.stats.enq_fail) / sec / 1e6;
        rr.empty_mops     = static_cast<double>(rr.stats.deq_empty) / sec / 1e6;
    }

    return rr;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream os;
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:   os << c; break;
        }
    }
    return os.str();
}

int main(int argc, char** argv) {
    BenchConfig cfg;

    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    if (cfg.test_type == TEST_BURST && cfg.threads > cfg.block_size) {
        std::cerr
            << "Warning: TEST_BURST uses __syncthreads(), which is block-local. "
            << "For clean phase behavior, prefer threads <= block size.\n";
    }

    hipDeviceProp_t prop{};
    check_hip(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");

    QueueInstance qi = create_queue(cfg.threads, cfg.chunk_ops);

    std::vector<RunResult> runs;
    runs.reserve(static_cast<size_t>(cfg.repeats));

    for (int r = 0; r < cfg.repeats; ++r) {
        if (r > 0 && cfg.reset_between_repeats) {
            queue_reset(qi.d_q, qi.d_h, cfg.threads);
            check_hip(hipDeviceSynchronize(), "queue_reset synchronize");
        }

        RunResult rr = run_once(cfg, qi);
        runs.push_back(rr);

        std::cerr
            << "[run " << r << "]"
            << " gpu=\"" << prop.name << "\""
            << " elapsed_ms=" << rr.elapsed_ms
            << " chunks=" << rr.chunks_launched
            << " enq_success=" << rr.stats.enq_success
            << " enq_fail=" << rr.stats.enq_fail
            << " deq_success=" << rr.stats.deq_success
            << " deq_empty=" << rr.stats.deq_empty
            << " successful_ops=" << rr.successful_ops
            << " attempted_ops=" << rr.attempted_ops
            << " succ_mops=" << rr.succ_mops
            << " attempted_mops=" << rr.attempted_mops
            << "\n";
    }

    unsigned long long total_enq_success = 0;
    unsigned long long total_enq_fail = 0;
    unsigned long long total_deq_success = 0;
    unsigned long long total_deq_empty = 0;
    unsigned long long total_successful_ops = 0;
    unsigned long long total_attempted_ops = 0;
    unsigned long long total_chunks = 0;

    double total_elapsed_ms = 0.0;
    double best_throughput_mops = 0.0;
    double worst_throughput_mops = 0.0;

    if (!runs.empty()) {
        best_throughput_mops = runs.front().succ_mops;
        worst_throughput_mops = runs.front().succ_mops;
    }

    for (const auto& rr : runs) {
        total_enq_success += rr.stats.enq_success;
        total_enq_fail += rr.stats.enq_fail;
        total_deq_success += rr.stats.deq_success;
        total_deq_empty += rr.stats.deq_empty;
        total_successful_ops += rr.successful_ops;
        total_attempted_ops += rr.attempted_ops;
        total_elapsed_ms += rr.elapsed_ms;
        total_chunks += static_cast<unsigned long long>(rr.chunks_launched);

        best_throughput_mops = std::max(best_throughput_mops, rr.succ_mops);
        worst_throughput_mops = std::min(worst_throughput_mops, rr.succ_mops);
    }

    const double n = runs.empty() ? 1.0 : static_cast<double>(runs.size());

    const double avg_enq_success = static_cast<double>(total_enq_success) / n;
    const double avg_enq_fail = static_cast<double>(total_enq_fail) / n;
    const double avg_deq_success = static_cast<double>(total_deq_success) / n;
    const double avg_deq_empty = static_cast<double>(total_deq_empty) / n;
    const double avg_successful_ops = static_cast<double>(total_successful_ops) / n;
    const double avg_attempted_ops = static_cast<double>(total_attempted_ops) / n;
    const double avg_elapsed_ms = total_elapsed_ms / n;
    const double avg_chunks_launched = static_cast<double>(total_chunks) / n;

    const double sec = avg_elapsed_ms / 1000.0;

    const double avg_throughput_mops =
        sec > 0.0 ? avg_successful_ops / sec / 1e6 : 0.0;

    const double avg_attempted_mops =
        sec > 0.0 ? avg_attempted_ops / sec / 1e6 : 0.0;

    const double avg_enq_mops =
        sec > 0.0 ? avg_enq_success / sec / 1e6 : 0.0;

    const double avg_deq_mops =
        sec > 0.0 ? avg_deq_success / sec / 1e6 : 0.0;

    const double avg_fail_mops =
        sec > 0.0 ? avg_enq_fail / sec / 1e6 : 0.0;

    const double avg_empty_mops =
        sec > 0.0 ? avg_deq_empty / sec / 1e6 : 0.0;

    const double enq_fail_rate =
        (avg_enq_success + avg_enq_fail) > 0.0
            ? avg_enq_fail / (avg_enq_success + avg_enq_fail)
            : 0.0;

    const double deq_empty_rate =
        (avg_deq_success + avg_deq_empty) > 0.0
            ? avg_deq_empty / (avg_deq_success + avg_deq_empty)
            : 0.0;

    const double success_rate =
        avg_attempted_ops > 0.0
            ? avg_successful_ops / avg_attempted_ops
            : 0.0;

    std::cout
        << "{"
        << "\"gpu_name\":\"" << json_escape(prop.name) << "\","
        << "\"queue\":\"" << QUEUE_NAME << "\","
        << "\"test\":\"" << test_type_name(cfg.test_type) << "\","
        << "\"threads\":" << cfg.threads << ","
        << "\"ops_per_thread\":" << cfg.chunk_ops << ","
        << "\"chunk_ops\":" << cfg.chunk_ops << ","
        << "\"producer_ratio\":" << cfg.producer_ratio << ","
        << "\"block_size\":" << cfg.block_size << ","
        << "\"run_ms\":" << cfg.run_ms << ","
        << "\"warmup_ms\":" << cfg.warmup_ms << ","
        << "\"repeats\":" << cfg.repeats << ","
        << "\"reset_between_repeats\":" << (cfg.reset_between_repeats ? "true" : "false") << ","

        << "\"avg_chunks_launched\":" << avg_chunks_launched << ","
        << "\"avg_elapsed_ms\":" << avg_elapsed_ms << ","

        << "\"avg_successful_ops\":" << avg_successful_ops << ","
        << "\"avg_attempted_ops\":" << avg_attempted_ops << ","
        << "\"avg_empty_dequeues\":" << avg_deq_empty << ","

        << "\"avg_enq_success\":" << avg_enq_success << ","
        << "\"avg_enq_fail\":" << avg_enq_fail << ","
        << "\"avg_deq_success\":" << avg_deq_success << ","
        << "\"avg_deq_empty\":" << avg_deq_empty << ","

        << "\"avg_throughput_mops\":" << avg_throughput_mops << ","
        << "\"avg_attempted_mops\":" << avg_attempted_mops << ","
        << "\"avg_enq_mops\":" << avg_enq_mops << ","
        << "\"avg_deq_mops\":" << avg_deq_mops << ","
        << "\"avg_fail_mops\":" << avg_fail_mops << ","
        << "\"avg_empty_mops\":" << avg_empty_mops << ","

        << "\"enq_fail_rate\":" << enq_fail_rate << ","
        << "\"deq_empty_rate\":" << deq_empty_rate << ","
        << "\"success_rate\":" << success_rate << ","

        << "\"best_throughput_mops\":" << best_throughput_mops << ","
        << "\"worst_throughput_mops\":" << worst_throughput_mops << ","
        << "\"success\":true"
        << "}"
        << "\n";

    destroy_queue(qi);
    return 0;
}