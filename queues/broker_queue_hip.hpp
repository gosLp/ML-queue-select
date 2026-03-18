#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <vector>
#include "bq.hpp"

#ifndef BROKER_CAPACITY
#define BROKER_CAPACITY 65536
#endif

#ifndef BROKER_MAX_THREADS
#define BROKER_MAX_THREADS 65536
#endif

#ifndef BROKER_EMPTY
#define BROKER_EMPTY 0ull
#endif

using broker_value_t = uint64_t;
using broker_queue = bq::BrokerQueue<broker_value_t, BROKER_CAPACITY, BROKER_MAX_THREADS>;

struct broker_handle {
    uint32_t thread_id;
};

static inline void broker_queue_host_init(broker_queue** d_q,
                                          broker_handle** d_handles,
                                          int num_threads) {
    hipMalloc((void**)d_q, sizeof(broker_queue));
    hipMalloc((void**)d_handles, sizeof(broker_handle) * (size_t)num_threads);

    broker_queue hq{};
    hq.host_init();
    hipMemcpy(*d_q, &hq, sizeof(broker_queue), hipMemcpyHostToDevice);

    std::vector<broker_handle> hh(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        hh[i].thread_id = static_cast<uint32_t>(i);
    }
    hipMemcpy(*d_handles, hh.data(),
              sizeof(broker_handle) * (size_t)num_threads,
              hipMemcpyHostToDevice);
}

static inline void broker_queue_reset(broker_queue* d_q,
                                      broker_handle* d_handles,
                                      int num_threads) {
    broker_queue hq{};
    hq.host_init();
    hipMemcpy(d_q, &hq, sizeof(broker_queue), hipMemcpyHostToDevice);

    std::vector<broker_handle> hh(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        hh[i].thread_id = static_cast<uint32_t>(i);
    }
    hipMemcpy(d_handles, hh.data(),
              sizeof(broker_handle) * (size_t)num_threads,
              hipMemcpyHostToDevice);
}

static inline void broker_queue_destroy(broker_queue* d_q,
                                        broker_handle* d_handles) {
    if (d_q) hipFree(d_q);
    if (d_handles) hipFree(d_handles);
}

__device__ __forceinline__ void broker_enqueue(broker_queue* q,
                                               broker_handle*,
                                               uint64_t v) {
    auto s = q->enqueue(v);
    (void)s;
}

__device__ __forceinline__ uint64_t broker_dequeue(broker_queue* q,
                                                   broker_handle*) {
    uint64_t out = BROKER_EMPTY;
    auto s = q->dequeue(out);
    return (s == bq::QueueStatus::Success) ? out : BROKER_EMPTY;
}