// #pragma once

// #include <hip/hip_runtime.h>
// #include <stdint.h>

// // ============================================================
// // Queue selection
// // Compile with exactly one of:
// //   -DWFQ
// //   -DSFQ
// //   -DBROKER
// // ============================================================

// #if (defined(WFQ) + defined(SFQ) + defined(BROKER)) > 1
// #error "Define only one of WFQ, SFQ, or BROKER"
// #endif

// #if !defined(WFQ) && !defined(SFQ) && !defined(BROKER)
// #error "Must define one of WFQ, SFQ, or BROKER"
// #endif

// // ============================================================
// // Includes + unified aliases
// // ============================================================

// #ifdef WFQ
// #include "./queues/wfq.hpp"   // rename this if your actual WFQ header filename differs

// using queue_t  = wf_queue;
// using handle_t = wf_handle;

// static constexpr const char* QUEUE_NAME = "wfq";

// #define QUEUE_EMPTY_VALUE WF_EMPTY

// static inline void queue_host_init(queue_t** d_q,
//                                    handle_t** d_handles,
//                                    int num_threads,
//                                    int ops_per_thread) {
//     // WFQ actually uses ops_per_thread to size the pool
//     wf_queue_host_init_ex(d_q, d_handles, num_threads, ops_per_thread);
// }

// static inline void queue_reset(queue_t* d_q,
//                                handle_t* d_handles,
//                                int num_threads) {
//     wf_queue_reset_for_bfs(d_q, d_handles, num_threads);
// }

// static inline void queue_destroy(queue_t* d_q,
//                                  handle_t* d_handles) {
//     wf_queue_destroy(d_q, d_handles);
// }

// __device__ __forceinline__ void queue_enqueue(queue_t* q, handle_t* h, uint64_t v) {
//     wf_enqueue(q, h, v);
// }

// __device__ __forceinline__ uint64_t queue_dequeue(queue_t* q, handle_t* h) {
//     return wf_dequeue(q, h);
// }

// #endif // WFQ

// #ifdef SFQ
// #include "./queues/sfq.hpp"
// #include "./queues/sfq.cpp"   // SFQ is header+implementation for simplicity

// using queue_t  = sfq_queue;
// using handle_t = sfq_handle;

// static constexpr const char* QUEUE_NAME = "sfq";

// #define QUEUE_EMPTY_VALUE SFQ_EMPTY

// static inline void queue_host_init(queue_t** d_q,
//                                    handle_t** d_handles,
//                                    int num_threads,
//                                    int ops_per_thread) {
//     (void)ops_per_thread; // SFQ does not use this currently
//     sfq_queue_host_init(d_q, d_handles, num_threads);
// }

// static inline void queue_reset(queue_t* d_q,
//                                handle_t* d_handles,
//                                int num_threads) {
//     // For now, just reinitialize in-place.
//     // I recommend adding a proper sfq_queue_reset(...) helper; see notes below.
//     // sfq_init_kernel<<<1, 1>>>(d_q, d_handles, num_threads);
//     sfq_queue_reset(d_q, d_handles, num_threads);
//     hipDeviceSynchronize();
// }

// static inline void queue_destroy(queue_t* d_q,
//                                  handle_t* d_handles) {
//     sfq_queue_destroy(d_q, d_handles);
// }

// __device__ __forceinline__ bool queue_try_enqueue(queue_t* q, handle_t* h, uint64_t v) {
//     uint32_t item = static_cast<uint32_t>(v & 0xFFFFFFFFu);
//     if (item == 0u) item = 1u;
//     return sfq_enqueue_nb_u32(q, item) == SFQ_SUCCESS;
// }

// __device__ __forceinline__ bool queue_try_dequeue(queue_t* q, handle_t* h, uint64_t* out) {
//     uint32_t item = 0;
//     int rc = sfq_dequeue_nb_u32(q, &item);
//     if (rc == SFQ_SUCCESS) {
//         *out = static_cast<uint64_t>(item);
//         return true;
//     }
//     *out = SFQ_EMPTY;
//     return false;
// }
// #endif // SFQ


// #ifdef BROKER
// #include "queues/broker_queue_hip.hpp"

// using queue_t = broker_queue;
// using handle_t = broker_handle;

// static constexpr const char* QUEUE_NAME = "broker";
// #define QUEUE_EMPTY_VALUE BROKER_EMPTY

// static inline void queue_host_init(queue_t** d_q,
//                                    handle_t** d_handles,
//                                    int num_threads,
//                                    int ops_per_thread) {
//     (void)ops_per_thread;
//     broker_queue_host_init(d_q, d_handles, num_threads);
// }

// static inline void queue_reset(queue_t* d_q,
//                                handle_t* d_handles,
//                                int num_threads) {
//     broker_queue_reset(d_q, d_handles, num_threads);
// }

// static inline void queue_destroy(queue_t* d_q,
//                                  handle_t* d_handles) {
//     broker_queue_destroy(d_q, d_handles);
// }

// __device__ __forceinline__ bool queue_try_enqueue(queue_t* q, handle_t* h, uint64_t v) {
//     return q->enqueue(v) == bq::QueueStatus::Success;
// }

// __device__ __forceinline__ bool queue_try_dequeue(queue_t* q, handle_t* h, uint64_t* out) {
//     return q->dequeue(*out) == bq::QueueStatus::Success;
// }

// #endif
// // ============================================================
// // Common benchmark test modes
// // ============================================================

// enum BenchTestType : int {
//     TEST_BALANCED    = 0,  // each thread mixes enq/deq by producer ratio
//     TEST_SPLIT_ROLES = 1,  // some threads are producers, some consumers
//     TEST_BURST       = 2   // phase 1 enqueue-heavy, phase 2 dequeue-heavy
// };

// // ============================================================
// // Helpers
// // ============================================================

// __device__ __forceinline__ bool should_enqueue_balanced(int iter, int producer_ratio_percent) {
//     // Deterministic 100-step pattern:
//     // producer_ratio_percent = 75 => 75% enqueue attempts
//     int slot = iter % 100;
//     return slot < producer_ratio_percent;
// }

// __device__ __forceinline__ bool thread_is_producer(int tid,
//                                                    int total_threads,
//                                                    int producer_ratio_percent) {
//     // Deterministically assign first X% of threads as producers
//     // Example: 25% => first quarter of tids are producers
//     if (producer_ratio_percent <= 0) return false;
//     if (producer_ratio_percent >= 100) return true;

//     long long lhs = static_cast<long long>(tid) * 100LL;
//     long long rhs = static_cast<long long>(producer_ratio_percent) * static_cast<long long>(total_threads);
//     return lhs < rhs;
// }

// // ============================================================
// // Generic benchmark kernel
// // Writes per-thread successful op counts to thread_counts[tid]
// // Also optionally records empty dequeue counts.
// // ============================================================

// __global__ void bench_kernel(queue_t* q,
//                              handle_t* handles,
//                              uint64_t* thread_counts,
//                              uint64_t* empty_dequeues,
//                              int num_threads,
//                              int ops_per_thread,
//                              int producer_ratio_percent,
//                              int test_type) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_threads) return;

//     handle_t* h = &handles[tid];

//     uint64_t successful_ops = 0;
//     uint64_t empty_count = 0;

//     switch (test_type) {
//         case TEST_BALANCED: {
//             for (int i = 0; i < ops_per_thread; ++i) {
//                 bool do_enq = should_enqueue_balanced(i, producer_ratio_percent);

//                 if (do_enq) {
//                     uint64_t value = (static_cast<uint64_t>(tid) << 32) |
//                                      static_cast<uint64_t>(i + 1);
//                     queue_enqueue(q, h, value);
//                     successful_ops++;
//                 } else {
//                     uint64_t out = queue_dequeue(q, h);
//                     if (out != QUEUE_EMPTY_VALUE) {
//                         successful_ops++;
//                     } else {
//                         empty_count++;
//                     }
//                 }
//             }
//             break;
//         }

//         case TEST_SPLIT_ROLES: {
//             bool producer = thread_is_producer(tid, num_threads, producer_ratio_percent);

//             if (producer) {
//                 for (int i = 0; i < ops_per_thread; ++i) {
//                     uint64_t value = (static_cast<uint64_t>(tid) << 32) |
//                                      static_cast<uint64_t>(i + 1);
//                     queue_enqueue(q, h, value);
//                     successful_ops++;
//                 }
//             } else {
//                 for (int i = 0; i < ops_per_thread; ++i) {
//                     uint64_t out = queue_dequeue(q, h);
//                     if (out != QUEUE_EMPTY_VALUE) {
//                         successful_ops++;
//                     } else {
//                         empty_count++;
//                     }
//                 }
//             }
//             break;
//         }

//         case TEST_BURST: {
//             // Phase 1: enqueue-heavy
//             for (int i = 0; i < ops_per_thread; ++i) {
//                 uint64_t value = (static_cast<uint64_t>(tid) << 32) |
//                                  static_cast<uint64_t>(i + 1);
//                 queue_enqueue(q, h, value);
//                 successful_ops++;
//             }

//             __syncthreads();

//             // Phase 2: dequeue-heavy
//             for (int i = 0; i < ops_per_thread; ++i) {
//                 uint64_t out = queue_dequeue(q, h);
//                 if (out != QUEUE_EMPTY_VALUE) {
//                     successful_ops++;
//                 } else {
//                     empty_count++;
//                 }
//             }
//             break;
//         }

//         default: {
//             // Fallback to balanced
//             for (int i = 0; i < ops_per_thread; ++i) {
//                 bool do_enq = should_enqueue_balanced(i, producer_ratio_percent);

//                 if (do_enq) {
//                     uint64_t value = (static_cast<uint64_t>(tid) << 32) |
//                                      static_cast<uint64_t>(i + 1);
//                     queue_enqueue(q, h, value);
//                     successful_ops++;
//                 } else {
//                     uint64_t out = queue_dequeue(q, h);
//                     if (out != QUEUE_EMPTY_VALUE) {
//                         successful_ops++;
//                     } else {
//                         empty_count++;
//                     }
//                 }
//             }
//             break;
//         }
//     }

//     thread_counts[tid] = successful_ops;
//     if (empty_dequeues) empty_dequeues[tid] = empty_count;
// }


#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

// ============================================================
// Queue selection
// Compile with exactly one of:
//   -DWFQ
//   -DSFQ
//   -DBROKER
// Optional:
//   -DBROKER_BWD=1      use BWD non-linearizable Broker fast path
//   -DBQ_CAPACITY=65536
//   -DBQ_MAX_THREADS=65536
// ============================================================

#if (defined(WFQ) + defined(SFQ) + defined(BROKER)) > 1
#error "Define only one of WFQ, SFQ, or BROKER"
#endif

#if !defined(WFQ) && !defined(SFQ) && !defined(BROKER)
#error "Must define one of WFQ, SFQ, or BROKER"
#endif

#ifndef BROKER_BWD
#define BROKER_BWD 0
#endif

static inline void qapi_check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(e));
        std::abort();
    }
}

// ============================================================
// WFQ backend
// ============================================================

#ifdef WFQ

#include "./queues/wfq.hpp"

using queue_t  = wf_queue;
using handle_t = wf_handle;

static constexpr const char* QUEUE_NAME = "wfq";
static constexpr bool NEEDS_HANDLES = true;

#define QUEUE_EMPTY_VALUE WF_EMPTY

#ifndef WF_BENCH_PREALLOC_OPS_PER_THREAD
#define WF_BENCH_PREALLOC_OPS_PER_THREAD 65536
#endif

static inline void queue_host_init(queue_t** d_q,
                                   handle_t** d_handles,
                                   int num_threads,
                                   int ops_or_chunk_ops) {
    (void)ops_or_chunk_ops;

    wf_queue_host_init_ex(
        d_q,
        d_handles,
        num_threads,
        WF_BENCH_PREALLOC_OPS_PER_THREAD
    );
}

static inline void queue_reset(queue_t* d_q,
                               handle_t* d_handles,
                               int num_threads) {
    wf_queue_reset_for_bfs(d_q, d_handles, num_threads);
}

static inline void queue_destroy(queue_t* d_q,
                                 handle_t* d_handles) {
    // wf_queue_destroy(d_q, d_handles);
    if (d_q) hipFree(d_q);
    if (d_handles) hipFree(d_handles);
}

__device__ __forceinline__ bool queue_enqueue_op(queue_t* q,
                                                 handle_t* h,
                                                 uint64_t v) {
    return wf_enqueue(q, h, v);
}

__device__ __forceinline__ bool queue_dequeue_op(queue_t* q,
                                                 handle_t* h,
                                                 uint64_t* out) {
    uint64_t v = wf_dequeue(q, h);
    if (v == WF_EMPTY) {
        *out = WF_EMPTY;
        return false;
    }
    *out = v;
    return true;
}

#endif


// ============================================================
// SFQ backend
// ============================================================

#ifdef SFQ

#include "./queues/sfq.hpp"
#include "./queues/sfq.cpp"

using queue_t  = sfq_queue;
using handle_t = sfq_handle;

static constexpr const char* QUEUE_NAME = "sfq";
static constexpr bool NEEDS_HANDLES = true;

#define QUEUE_EMPTY_VALUE SFQ_EMPTY

static inline void queue_host_init(queue_t** d_q,
                                   handle_t** d_handles,
                                   int num_threads,
                                   int ops_or_chunk_ops) {
    (void)ops_or_chunk_ops;
    sfq_queue_host_init(d_q, d_handles, num_threads);
}

static inline void queue_reset(queue_t* d_q,
                               handle_t* d_handles,
                               int num_threads) {
#ifdef HAS_SFQ_QUEUE_RESET
    sfq_queue_reset(d_q, d_handles, num_threads);
#else
    // Reinitialize in place using the existing init kernel.
    constexpr int block = 256;
    int grid = (num_threads + block - 1) / block;
    if (grid < 1) grid = 1;
    hipLaunchKernelGGL(sfq_init_kernel, dim3(grid), dim3(block), 0, 0,
                       d_q, d_handles, num_threads);
    qapi_check_hip(hipGetLastError(), "sfq_init_kernel launch in queue_reset");
    qapi_check_hip(hipDeviceSynchronize(), "sfq reset synchronize");
#endif
}

static inline void queue_destroy(queue_t* d_q,
                                 handle_t* d_handles) {
    sfq_queue_destroy(d_q, d_handles);
}

__device__ __forceinline__ bool queue_enqueue_op(queue_t* q,
                                                 handle_t* h,
                                                 uint64_t v) {
    (void)h;
    uint32_t item = static_cast<uint32_t>(v & 0xFFFFFFFFu);
    if (item == 0u) item = 1u;

    // Match check_gwf/test_final style: SFQ is allowed to block/retry.
    return sfq_enqueue_blocking_u32(q, item) == SFQ_SUCCESS;
}

__device__ __forceinline__ bool queue_dequeue_op(queue_t* q,
                                                 handle_t* h,
                                                 uint64_t* out) {
    (void)h;
    uint32_t item = 0u;

    // Match your working SFQ: blocking dequeue is safe in pairwise balanced.
    // For split, the benchmark avoids deadlock by using finite chunk launches
    // and the queue state keeps moving across chunks.
    int rc = sfq_dequeue_blocking_u32(q, &item);

    if (rc == SFQ_SUCCESS) {
        *out = static_cast<uint64_t>(item);
        return true;
    }

    *out = SFQ_EMPTY;
    return false;
}

__device__ __forceinline__ bool queue_enqueue_nb_op(queue_t* q,
                                                    handle_t* h,
                                                    uint64_t v) {
    (void)h;
    uint32_t item = static_cast<uint32_t>(v & 0xFFFFFFFFu);
    if (item == 0u) item = 1u;
    return sfq_enqueue_nb_u32(q, item) == SFQ_SUCCESS;
}

__device__ __forceinline__ bool queue_dequeue_nb_op(queue_t* q,
                                                    handle_t* h,
                                                    uint64_t* out) {
    (void)h;
    uint32_t item = 0u;
    int rc = sfq_dequeue_nb_u32(q, &item);
    if (rc == SFQ_SUCCESS) {
        *out = static_cast<uint64_t>(item);
        return true;
    }
    *out = SFQ_EMPTY;
    return false;
}

#endif


// ============================================================
// Broker Queue backend
// ============================================================

#ifdef BROKER

#include "./queues/broker_queue_hip.hpp"

#ifndef BQ_CAPACITY
#define BQ_CAPACITY 65536u
#endif

#ifndef BQ_MAX_THREADS
#define BQ_MAX_THREADS 65536u
#endif

using value_t = uint64_t;
using queue_t = bq::BrokerQueue<value_t, BQ_CAPACITY, BQ_MAX_THREADS>;

struct handle_t {
    uint32_t tid;
};

static constexpr const char* QUEUE_NAME = BROKER_BWD ? "bwd" : "broker";
static constexpr bool NEEDS_HANDLES = true;

#define QUEUE_EMPTY_VALUE 0ull

static inline void queue_host_init(queue_t** d_q,
                                   handle_t** d_handles,
                                   int num_threads,
                                   int ops_or_chunk_ops) {
    (void)ops_or_chunk_ops;

    qapi_check_hip(hipMalloc(reinterpret_cast<void**>(d_q), sizeof(queue_t)),
                   "hipMalloc broker queue");

    qapi_check_hip(hipMalloc(reinterpret_cast<void**>(d_handles),
                             sizeof(handle_t) * static_cast<size_t>(num_threads)),
                   "hipMalloc broker handles");

    queue_t hq{};
    hq.host_init();

    qapi_check_hip(hipMemcpy(*d_q, &hq, sizeof(queue_t), hipMemcpyHostToDevice),
                   "hipMemcpy broker queue init");

    std::vector<handle_t> hh(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        hh[static_cast<size_t>(i)].tid = static_cast<uint32_t>(i);
    }

    qapi_check_hip(hipMemcpy(*d_handles,
                             hh.data(),
                             sizeof(handle_t) * static_cast<size_t>(num_threads),
                             hipMemcpyHostToDevice),
                   "hipMemcpy broker handles init");
}

static inline void queue_reset(queue_t* d_q,
                               handle_t* d_handles,
                               int num_threads) {
    queue_t hq{};
    hq.host_init();

    qapi_check_hip(hipMemcpy(d_q, &hq, sizeof(queue_t), hipMemcpyHostToDevice),
                   "hipMemcpy broker reset queue");

    std::vector<handle_t> hh(static_cast<size_t>(num_threads));
    for (int i = 0; i < num_threads; ++i) {
        hh[static_cast<size_t>(i)].tid = static_cast<uint32_t>(i);
    }

    qapi_check_hip(hipMemcpy(d_handles,
                             hh.data(),
                             sizeof(handle_t) * static_cast<size_t>(num_threads),
                             hipMemcpyHostToDevice),
                   "hipMemcpy broker reset handles");
}

static inline void queue_destroy(queue_t* d_q,
                                 handle_t* d_handles) {
    if (d_q) qapi_check_hip(hipFree(d_q), "hipFree broker queue");
    if (d_handles) qapi_check_hip(hipFree(d_handles), "hipFree broker handles");
}

__device__ __forceinline__ bool queue_enqueue_op(queue_t* q,
                                                 handle_t* h,
                                                 uint64_t v) {
    (void)h;
#if BROKER_BWD
    return q->enqueue_bwd(v) == bq::QueueStatus::Success;
#else
    return q->enqueue(v) == bq::QueueStatus::Success;
#endif
}

__device__ __forceinline__ bool queue_dequeue_op(queue_t* q,
                                                 handle_t* h,
                                                 uint64_t* out) {
    (void)h;
    value_t v = 0;

#if BROKER_BWD
    bq::QueueStatus st = q->dequeue_bwd(v);
#else
    bq::QueueStatus st = q->dequeue(v);
#endif

    if (st == bq::QueueStatus::Success) {
        *out = static_cast<uint64_t>(v);
        return true;
    }

    *out = QUEUE_EMPTY_VALUE;
    return false;
}

__device__ __forceinline__ bool queue_enqueue_nb_op(queue_t* q,
                                                    handle_t* h,
                                                    uint64_t v) {
    return queue_enqueue_op(q, h, v);
}

__device__ __forceinline__ bool queue_dequeue_nb_op(queue_t* q,
                                                    handle_t* h,
                                                    uint64_t* out) {
    return queue_dequeue_op(q, h, out);
}

#endif


// ============================================================
// Common test modes + stats
// ============================================================

enum BenchTestType : int {
    TEST_BALANCED    = 0,
    TEST_SPLIT_ROLES = 1,
    TEST_BURST       = 2
};

struct Stats {
    unsigned long long enq_success = 0;
    unsigned long long enq_fail    = 0;
    unsigned long long deq_success = 0;
    unsigned long long deq_empty   = 0;
};

__device__ unsigned long long g_enq_success = 0;
__device__ unsigned long long g_enq_fail    = 0;
__device__ unsigned long long g_deq_success = 0;
__device__ unsigned long long g_deq_empty   = 0;

__device__ __forceinline__ bool thread_is_producer(int tid,
                                                   int total_threads,
                                                   int producer_ratio_percent) {
    if (producer_ratio_percent <= 0) return false;
    if (producer_ratio_percent >= 100) return true;

    long long lhs = static_cast<long long>(tid) * 100LL;
    long long rhs = static_cast<long long>(producer_ratio_percent) *
                    static_cast<long long>(total_threads);

    return lhs < rhs;
}

// ============================================================
// Fixed-duration chunk kernels emulating check_gwf
// ============================================================

__global__ void balanced_chunk_kernel(queue_t* q,
                                      handle_t* handles,
                                      int num_threads,
                                      int chunk_ops) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* h = handles ? &handles[tid] : nullptr;

    unsigned long long local_enq_ok    = 0;
    unsigned long long local_enq_fail  = 0;
    unsigned long long local_deq_ok    = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; ++i) {
        uint64_t val = (((uint64_t)tid) << 32)
                     ^ (uint64_t)(i + 1)
                     ^ 0x9e3779b97f4a7c15ull;

        bool ok_enq = queue_enqueue_op(q, h, val);
        if (ok_enq) ++local_enq_ok;
        else        ++local_enq_fail;

        uint64_t out = 0;
        bool ok_deq = queue_dequeue_op(q, h, &out);
        if (ok_deq) ++local_deq_ok;
        else        ++local_deq_empty;
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}

__global__ void split_chunk_kernel(queue_t* q,
                                   handle_t* handles,
                                   int num_threads,
                                   int chunk_ops,
                                   int producer_percent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* h = handles ? &handles[tid] : nullptr;
    bool is_producer = thread_is_producer(tid, num_threads, producer_percent);

    unsigned long long local_enq_ok    = 0;
    unsigned long long local_enq_fail  = 0;
    unsigned long long local_deq_ok    = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; ++i) {
        if (is_producer) {
            uint64_t val = (((uint64_t)tid) << 32)
                         ^ (uint64_t)(i + 1)
                         ^ 0x517cc1b727220a95ull;

#ifdef SFQ
            // Split can deadlock with blocking-only SFQ under imbalance.
            // Use SFQ's nonblocking path here so the fixed-duration harness
            // never hangs and still exposes fail/empty behavior.
            bool ok_enq = queue_enqueue_nb_op(q, h, val);
#else
            bool ok_enq = queue_enqueue_op(q, h, val);
#endif

            if (ok_enq) ++local_enq_ok;
            else        ++local_enq_fail;
        } else {
            uint64_t out = 0;

#ifdef SFQ
            bool ok_deq = queue_dequeue_nb_op(q, h, &out);
#else
            bool ok_deq = queue_dequeue_op(q, h, &out);
#endif

            if (ok_deq) ++local_deq_ok;
            else        ++local_deq_empty;
        }
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}

__global__ void burst_chunk_kernel(queue_t* q,
                                   handle_t* handles,
                                   int num_threads,
                                   int chunk_ops) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* h = handles ? &handles[tid] : nullptr;

    unsigned long long local_enq_ok    = 0;
    unsigned long long local_enq_fail  = 0;
    unsigned long long local_deq_ok    = 0;
    unsigned long long local_deq_empty = 0;

    for (int i = 0; i < chunk_ops; ++i) {
        uint64_t val = (((uint64_t)tid) << 32)
                     ^ (uint64_t)(i + 1)
                     ^ 0x123456789abcdefull;

        bool ok_enq = queue_enqueue_op(q, h, val);
        if (ok_enq) ++local_enq_ok;
        else        ++local_enq_fail;
    }

    __syncthreads();

    for (int i = 0; i < chunk_ops; ++i) {
        uint64_t out = 0;
        bool ok_deq = queue_dequeue_op(q, h, &out);
        if (ok_deq) ++local_deq_ok;
        else        ++local_deq_empty;
    }

    atomicAdd(&g_enq_success, local_enq_ok);
    atomicAdd(&g_enq_fail,    local_enq_fail);
    atomicAdd(&g_deq_success, local_deq_ok);
    atomicAdd(&g_deq_empty,   local_deq_empty);
}