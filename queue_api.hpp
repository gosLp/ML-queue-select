#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

// ============================================================
// Queue selection
// Compile with exactly one of:
//   -DWFQ
//   -DSFQ
//   -DBROKER
// ============================================================

#if (defined(WFQ) + defined(SFQ) + defined(BROKER)) > 1
#error "Define only one of WFQ, SFQ, or BROKER"
#endif

#if !defined(WFQ) && !defined(SFQ) && !defined(BROKER)
#error "Must define one of WFQ, SFQ, or BROKER"
#endif

// ============================================================
// Includes + unified aliases
// ============================================================

#ifdef WFQ
#include "./queues/wfq.hpp"   // rename this if your actual WFQ header filename differs

using queue_t  = wf_queue;
using handle_t = wf_handle;

static constexpr const char* QUEUE_NAME = "wfq";

#define QUEUE_EMPTY_VALUE WF_EMPTY

static inline void queue_host_init(queue_t** d_q,
                                   handle_t** d_handles,
                                   int num_threads,
                                   int ops_per_thread) {
    // WFQ actually uses ops_per_thread to size the pool
    wf_queue_host_init_ex(d_q, d_handles, num_threads, ops_per_thread);
}

static inline void queue_reset(queue_t* d_q,
                               handle_t* d_handles,
                               int num_threads) {
    wf_queue_reset_for_bfs(d_q, d_handles, num_threads);
}

static inline void queue_destroy(queue_t* d_q,
                                 handle_t* d_handles) {
    wf_queue_destroy(d_q, d_handles);
}

__device__ __forceinline__ void queue_enqueue(queue_t* q, handle_t* h, uint64_t v) {
    wf_enqueue(q, h, v);
}

__device__ __forceinline__ uint64_t queue_dequeue(queue_t* q, handle_t* h) {
    return wf_dequeue(q, h);
}

#endif // WFQ

#ifdef SFQ
#include "./queues/sfq.hpp"
#include "./queues/sfq.cpp"   // SFQ is header+implementation for simplicity

using queue_t  = sfq_queue;
using handle_t = sfq_handle;

static constexpr const char* QUEUE_NAME = "sfq";

#define QUEUE_EMPTY_VALUE SFQ_EMPTY

static inline void queue_host_init(queue_t** d_q,
                                   handle_t** d_handles,
                                   int num_threads,
                                   int ops_per_thread) {
    (void)ops_per_thread; // SFQ does not use this currently
    sfq_queue_host_init(d_q, d_handles, num_threads);
}

static inline void queue_reset(queue_t* d_q,
                               handle_t* d_handles,
                               int num_threads) {
    // For now, just reinitialize in-place.
    // I recommend adding a proper sfq_queue_reset(...) helper; see notes below.
    // sfq_init_kernel<<<1, 1>>>(d_q, d_handles, num_threads);
    sfq_queue_reset(d_q, d_handles, num_threads);
    hipDeviceSynchronize();
}

static inline void queue_destroy(queue_t* d_q,
                                 handle_t* d_handles) {
    sfq_queue_destroy(d_q, d_handles);
}

__device__ __forceinline__ void queue_enqueue(queue_t* q, handle_t* h, uint64_t v) {
    sfq_enqueue(q, h, v);
}

__device__ __forceinline__ uint64_t queue_dequeue(queue_t* q, handle_t* h) {
    return sfq_dequeue(q, h);
}
#endif // SFQ


#ifdef BROKER
#include "queues/broker_queue_hip.hpp"

using queue_t = broker_queue;
using handle_t = broker_handle;

static constexpr const char* QUEUE_NAME = "broker";
#define QUEUE_EMPTY_VALUE BROKER_EMPTY

static inline void queue_host_init(queue_t** d_q,
                                   handle_t** d_handles,
                                   int num_threads,
                                   int ops_per_thread) {
    (void)ops_per_thread;
    broker_queue_host_init(d_q, d_handles, num_threads);
}

static inline void queue_reset(queue_t* d_q,
                               handle_t* d_handles,
                               int num_threads) {
    broker_queue_reset(d_q, d_handles, num_threads);
}

static inline void queue_destroy(queue_t* d_q,
                                 handle_t* d_handles) {
    broker_queue_destroy(d_q, d_handles);
}

__device__ __forceinline__ void queue_enqueue(queue_t* q, handle_t* h, uint64_t v) {
    broker_enqueue(q, h, v);
}

__device__ __forceinline__ uint64_t queue_dequeue(queue_t* q, handle_t* h) {
    return broker_dequeue(q, h);
}
#endif
// ============================================================
// Common benchmark test modes
// ============================================================

enum BenchTestType : int {
    TEST_BALANCED    = 0,  // each thread mixes enq/deq by producer ratio
    TEST_SPLIT_ROLES = 1,  // some threads are producers, some consumers
    TEST_BURST       = 2   // phase 1 enqueue-heavy, phase 2 dequeue-heavy
};

// ============================================================
// Helpers
// ============================================================

__device__ __forceinline__ bool should_enqueue_balanced(int iter, int producer_ratio_percent) {
    // Deterministic 100-step pattern:
    // producer_ratio_percent = 75 => 75% enqueue attempts
    int slot = iter % 100;
    return slot < producer_ratio_percent;
}

__device__ __forceinline__ bool thread_is_producer(int tid,
                                                   int total_threads,
                                                   int producer_ratio_percent) {
    // Deterministically assign first X% of threads as producers
    // Example: 25% => first quarter of tids are producers
    if (producer_ratio_percent <= 0) return false;
    if (producer_ratio_percent >= 100) return true;

    long long lhs = static_cast<long long>(tid) * 100LL;
    long long rhs = static_cast<long long>(producer_ratio_percent) * static_cast<long long>(total_threads);
    return lhs < rhs;
}

// ============================================================
// Generic benchmark kernel
// Writes per-thread successful op counts to thread_counts[tid]
// Also optionally records empty dequeue counts.
// ============================================================

__global__ void bench_kernel(queue_t* q,
                             handle_t* handles,
                             uint64_t* thread_counts,
                             uint64_t* empty_dequeues,
                             int num_threads,
                             int ops_per_thread,
                             int producer_ratio_percent,
                             int test_type) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    handle_t* h = &handles[tid];

    uint64_t successful_ops = 0;
    uint64_t empty_count = 0;

    switch (test_type) {
        case TEST_BALANCED: {
            for (int i = 0; i < ops_per_thread; ++i) {
                bool do_enq = should_enqueue_balanced(i, producer_ratio_percent);

                if (do_enq) {
                    uint64_t value = (static_cast<uint64_t>(tid) << 32) |
                                     static_cast<uint64_t>(i + 1);
                    queue_enqueue(q, h, value);
                    successful_ops++;
                } else {
                    uint64_t out = queue_dequeue(q, h);
                    if (out != QUEUE_EMPTY_VALUE) {
                        successful_ops++;
                    } else {
                        empty_count++;
                    }
                }
            }
            break;
        }

        case TEST_SPLIT_ROLES: {
            bool producer = thread_is_producer(tid, num_threads, producer_ratio_percent);

            if (producer) {
                for (int i = 0; i < ops_per_thread; ++i) {
                    uint64_t value = (static_cast<uint64_t>(tid) << 32) |
                                     static_cast<uint64_t>(i + 1);
                    queue_enqueue(q, h, value);
                    successful_ops++;
                }
            } else {
                for (int i = 0; i < ops_per_thread; ++i) {
                    uint64_t out = queue_dequeue(q, h);
                    if (out != QUEUE_EMPTY_VALUE) {
                        successful_ops++;
                    } else {
                        empty_count++;
                    }
                }
            }
            break;
        }

        case TEST_BURST: {
            // Phase 1: enqueue-heavy
            for (int i = 0; i < ops_per_thread; ++i) {
                uint64_t value = (static_cast<uint64_t>(tid) << 32) |
                                 static_cast<uint64_t>(i + 1);
                queue_enqueue(q, h, value);
                successful_ops++;
            }

            __syncthreads();

            // Phase 2: dequeue-heavy
            for (int i = 0; i < ops_per_thread; ++i) {
                uint64_t out = queue_dequeue(q, h);
                if (out != QUEUE_EMPTY_VALUE) {
                    successful_ops++;
                } else {
                    empty_count++;
                }
            }
            break;
        }

        default: {
            // Fallback to balanced
            for (int i = 0; i < ops_per_thread; ++i) {
                bool do_enq = should_enqueue_balanced(i, producer_ratio_percent);

                if (do_enq) {
                    uint64_t value = (static_cast<uint64_t>(tid) << 32) |
                                     static_cast<uint64_t>(i + 1);
                    queue_enqueue(q, h, value);
                    successful_ops++;
                } else {
                    uint64_t out = queue_dequeue(q, h);
                    if (out != QUEUE_EMPTY_VALUE) {
                        successful_ops++;
                    } else {
                        empty_count++;
                    }
                }
            }
            break;
        }
    }

    thread_counts[tid] = successful_ops;
    if (empty_dequeues) empty_dequeues[tid] = empty_count;
}