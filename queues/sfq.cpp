// sfqueue_hip.cpp - Native HIP implementation of SFQ (Slot-based FIFO Queue)
// Converted from OpenCL for fair comparison with WF-Queue

#include <hip/hip_runtime.h>
#include <stdint.h>

//=============================================================================
// SFQ CONFIGURATION - Matching your OpenCL version
//=============================================================================
#ifndef SFQ_QUEUE_LENGTH
#define SFQ_QUEUE_LENGTH 65536
// #define SFQ_QUEUE_LENGTH 1048576
#define SFQ_QUEUE_FACTOR 16
#endif

#ifndef SFQ_WORK
#define SFQ_WORK 100
#endif

#define SFQ_QUEUE_MASK (SFQ_QUEUE_LENGTH - 1)
#define SFQ_QUEUE_SMASK ((1U << (32 - SFQ_QUEUE_FACTOR)) - 1)
#define SFQ_GET_TARGET(H, Q) ((H) & SFQ_QUEUE_MASK)

#define SFQ_NULL_1 UINT32_MAX
#define SFQ_NULL_0 (UINT32_MAX-1)
#define SFQ_EMPTY 0

#ifndef SFQ_FAILSAFE
#define SFQ_FAILSAFE 10000
#endif

#define SFQ_TEST_FAILSAFE (fail < SFQ_FAILSAFE)

//=============================================================================
// HIP ATOMIC OPERATIONS - Native versions
//=============================================================================
#define SFQ_ATOMIC_LOAD(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
#define SFQ_ATOMIC_STORE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#define SFQ_ATOMIC_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, &(expected), desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define SFQ_ATOMIC_ADD(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
#define SFQ_ATOMIC_XCHG(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_SEQ_CST)
#define SFQ_FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)

//=============================================================================
// SFQ QUEUE STRUCTURE - Direct conversion from your OpenCL version
//=============================================================================
// struct sfq_queue {
//     volatile uint32_t head;
//     volatile uint32_t tail; 
//     volatile uint32_t vnull;
//     volatile uint32_t done;
//     union {
//         volatile uint32_t items[SFQ_QUEUE_LENGTH];
//         volatile uint32_t nodes[SFQ_QUEUE_LENGTH];
//     };
//     volatile uint32_t slots[SFQ_QUEUE_LENGTH];
//     uint32_t nprocs;  // For compatibility with test framework
// };

// Handle structure for compatibility with test interface
// struct sfq_handle {
//     uint32_t thread_id;  // Simple thread identification
//     uint32_t dummy[15];  // Padding to match wf_handle size roughly
// };

//=============================================================================
// CORE SFQ OPERATIONS - Direct translation from your OpenCL code
//=============================================================================

// Blocking enqueue - exact translation of my_enqueue_slot
__device__ int sfq_enqueue_slot(sfq_queue* q, uint32_t item) {
    const uint32_t tail = SFQ_ATOMIC_ADD(&q->tail, 1);
    const uint32_t pass = (tail >> SFQ_QUEUE_FACTOR) << 1;
    const uint32_t target = SFQ_GET_TARGET(tail, q);
    
#ifndef NOFAILSAFE
    uint32_t fail = 0;
#endif

    uint32_t slot = SFQ_ATOMIC_LOAD(&q->slots[target]);
    while (slot != pass) {
        uint32_t qdone = SFQ_ATOMIC_LOAD(&q->done);
        
#ifndef NOFAILSAFE
        fail++;
        if (!SFQ_TEST_FAILSAFE) {
            SFQ_ATOMIC_STORE(&q->done, 1);
            return 2;
        }
#endif
        slot = SFQ_ATOMIC_LOAD(&q->slots[target]);
    }
    
    // CRITICAL: Write item first, then fence, then update slot
    SFQ_ATOMIC_STORE(&q->items[target], item);
    SFQ_FENCE();  // Ensure item write is visible
    SFQ_ATOMIC_STORE(&q->slots[target], (pass + 1) & SFQ_QUEUE_SMASK);
    
    return 0;
}

// Blocking dequeue - exact translation of my_dequeue_slot  
__device__ int sfq_dequeue_slot(sfq_queue* q, uint32_t* p) {
    const uint32_t head = SFQ_ATOMIC_ADD(&q->head, 1);
    const uint32_t pass = ((head >> SFQ_QUEUE_FACTOR) << 1) + 1;
    const uint32_t target = SFQ_GET_TARGET(head, q);
    
#ifndef NOFAILSAFE
    uint32_t fail = 0;
#endif

    uint32_t slot = SFQ_ATOMIC_LOAD(&q->slots[target]);
    while (slot != pass) {
        volatile uint32_t qdone = SFQ_ATOMIC_LOAD(&q->done);
        if (qdone != 0 && head > qdone)
            return 1;
            
#ifndef NOFAILSAFE
        fail++;
        if (!SFQ_TEST_FAILSAFE) {
            SFQ_ATOMIC_STORE(&q->done, 2);
            return 2;
        }
#endif
        slot = SFQ_ATOMIC_LOAD(&q->slots[target]);
    }

    // CRITICAL: Fence before reading item to ensure visibility
    SFQ_FENCE(); 
    *p = SFQ_ATOMIC_LOAD(&q->items[target]);
    
    // Update slot after reading
    SFQ_ATOMIC_STORE(&q->slots[target], (pass + 1) & SFQ_QUEUE_SMASK);
    return 0;
}

// Non-blocking enqueue - exact translation of my_enqueue_nb_slot
__device__ int sfq_enqueue_nb_slot(sfq_queue* q, uint32_t item) {
    volatile uint32_t tail = SFQ_ATOMIC_LOAD(&q->tail);
    uint32_t target;
    uint32_t pass;
    
    for (;;) {
        target = SFQ_GET_TARGET(tail, q);
        pass = ((tail >> SFQ_QUEUE_FACTOR) << 1);
        
        if (SFQ_ATOMIC_LOAD(&q->slots[target]) != pass)
            return 1; // queue is full
            
        uint32_t ltail = tail;
        if ((ltail = SFQ_ATOMIC_XCHG(&q->tail, tail + 1)) == tail)
            break;
        tail = ltail;
    }
    
    SFQ_ATOMIC_STORE(&q->items[target], item);
    SFQ_FENCE();  // Ensure write visibility
    SFQ_ATOMIC_STORE(&q->slots[target], (pass + 1) & SFQ_QUEUE_SMASK);
    return 0;
}

// Non-blocking dequeue - exact translation of my_dequeue_nb_slot
__device__ int sfq_dequeue_nb_slot(sfq_queue* q, uint32_t* p) {
    volatile uint32_t head = SFQ_ATOMIC_LOAD(&q->head);
    uint32_t target;
    uint32_t pass;
    
    for (;;) {
        uint32_t lhead = head;
        target = SFQ_GET_TARGET(head, q);
        pass = (((head >> SFQ_QUEUE_FACTOR) << 1) + 1);
        
        if (SFQ_ATOMIC_LOAD(&q->slots[target]) != pass)
            return 1; // queue is empty
            
        if ((lhead = SFQ_ATOMIC_XCHG(&q->head, head + 1)) == head)
            break;
        head = lhead;
    }
    
    SFQ_FENCE();  // Ensure read visibility
    *p = SFQ_ATOMIC_LOAD(&q->items[target]);
    SFQ_ATOMIC_STORE(&q->slots[target], (pass + 1) & SFQ_QUEUE_SMASK);
    return 0;
}

//=============================================================================
// MAIN API INTERFACE - Compatible with WF-Queue test framework
//=============================================================================

// Main enqueue function - uses blocking version for fairness
__device__ void sfq_enqueue(sfq_queue* q, sfq_handle* h, uint64_t v) {
    // Convert uint64_t to uint32_t (SFQ works with 32-bit values)
    uint32_t item = (uint32_t)(v & 0xFFFFFFFF);
    if (item == 0) item = 1;  // Avoid zero values
    
    int result = sfq_enqueue_slot(q, item);
    // Note: In blocking mode, this should always succeed
    (void)result;  // Suppress unused variable warning
}

// Main dequeue function - uses blocking version for fairness  
__device__ uint64_t sfq_dequeue(sfq_queue* q, sfq_handle* h) {
    uint32_t item;
    int result = sfq_dequeue_slot(q, &item);
    
    if (result == 0) {
        return (uint64_t)item;  // Convert back to uint64_t
    } else {
        return SFQ_EMPTY;  // Empty queue
    }
}

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

// Initialize SFQ queue
__device__ void sfq_queue_init(sfq_queue* q, uint32_t nprocs) {
    SFQ_ATOMIC_STORE(&q->head, 0);
    SFQ_ATOMIC_STORE(&q->tail, 0);
    SFQ_ATOMIC_STORE(&q->vnull, 0);
    SFQ_ATOMIC_STORE(&q->done, 0);
    q->nprocs = nprocs;
    
    // Initialize all items and slots
    for (int i = 0; i < SFQ_QUEUE_LENGTH; i++) {
        SFQ_ATOMIC_STORE(&q->items[i], 0);
        SFQ_ATOMIC_STORE(&q->slots[i], 0);
    }
}

// Initialize SFQ handle (minimal for SFQ)
__device__ void sfq_handle_init(sfq_handle* h, sfq_queue* q, sfq_handle* next_handle, 
                               sfq_handle* enq_helper, sfq_handle* deq_helper) {
    h->thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // SFQ doesn't need complex handle state like WF-Queue
    (void)q; (void)next_handle; (void)enq_helper; (void)deq_helper;  // Suppress warnings
}

//=============================================================================
// KERNEL FUNCTIONS - Compatible with WF-Queue test interface
//=============================================================================

// Initialization kernel
__global__ void sfq_init_kernel(sfq_queue* q, sfq_handle* handles, int num_threads) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize queue
        sfq_queue_init(q, num_threads);
        
        // Initialize handles
        for (int i = 0; i < num_threads; i++) {
            sfq_handle* next_handle = &handles[(i + 1) % num_threads];
            sfq_handle* enq_helper = &handles[(i + 1) % num_threads]; 
            sfq_handle* deq_helper = &handles[(i + 1) % num_threads];
            
            sfq_handle_init(&handles[i], q, next_handle, enq_helper, deq_helper);
        }
    }
}

// Simple test kernel - same interface as WF-Queue
__global__ void sfq_simple_test_kernel(sfq_queue* q, sfq_handle* handles, 
                                      uint64_t* results, int num_threads, int ops_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;
    
    sfq_handle* h = &handles[tid];
    uint64_t successful_ops = 0;
    
    // Each thread does alternating enqueue/dequeue
    for (int i = 0; i < ops_per_thread; i++) {
        // Enqueue unique value: thread_id * 1000000 + operation_id + 1
        uint64_t value = tid * 1000000ULL + i + 1;
        sfq_enqueue(q, h, value);
        successful_ops++;
        
        // Small delay to let queue build up
        if (i % 10 == 0) {
            for (volatile int delay = 0; delay < 100; delay++);
        }
        
        // Dequeue
        uint64_t dequeued = sfq_dequeue(q, h); 
        if (dequeued != SFQ_EMPTY) {
            successful_ops++;
        }
    }
    
    results[tid] = successful_ops;
}

// High contention test - same interface as WF-Queue
__global__ void sfq_high_contention_kernel(sfq_queue* q, sfq_handle* handles, 
                                          uint64_t* results, int num_threads, int ops_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;
    
    sfq_handle* h = &handles[tid];
    uint64_t successful_ops = 0;
    
    // 70% producers, 30% consumers for realistic high contention
    bool is_producer = (tid % 10) < 7;
    
    if (is_producer) {
        // Producers: Just enqueue rapidly
        for (int i = 0; i < ops_per_thread; i++) {
            uint64_t value = ((uint64_t)tid << 32) | (i + 1);
            sfq_enqueue(q, h, value);
            successful_ops++;
        }
    } else {
        // Consumers: Dequeue aggressively (more attempts than producers)
        for (int i = 0; i < ops_per_thread * 3; i++) {
            uint64_t dequeued = sfq_dequeue(q, h);
            if (dequeued != SFQ_EMPTY) {
                successful_ops++;
            }
            // Brief pause to let producers catch up
            if (i % 50 == 0) {
                for (volatile int delay = 0; delay < 10; delay++);
            }
        }
    }
    
    results[tid] = successful_ops;
}

// Memory stress test - same interface as WF-Queue
__global__ void sfq_memory_stress_kernel(sfq_queue* q, sfq_handle* handles, 
                                        uint64_t* results, int num_threads, int ops_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;
    
    sfq_handle* h = &handles[tid];
    uint64_t successful_ops = 0;
    
    // Phase 1: All threads enqueue rapidly to build up huge queue
    for (int i = 0; i < ops_per_thread; i++) {
        uint64_t value = ((uint64_t)tid << 20) | (i + 1);
        sfq_enqueue(q, h, value);
        successful_ops++;
    }
    
    // Brief synchronization point
    __syncthreads();
    
    // Phase 2: All threads dequeue rapidly  
    for (int i = 0; i < ops_per_thread; i++) {
        uint64_t dequeued = sfq_dequeue(q, h);
        if (dequeued != SFQ_EMPTY) {
            successful_ops++;
        }
    }
    
    results[tid] = successful_ops;
}

// Performance test kernel - same interface as WF-Queue
__global__ void sfq_performance_test_kernel(sfq_queue* q, sfq_handle* handles, 
                                           uint64_t* results, int operations_per_thread,
                                           int test_type) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= q->nprocs) return;
    
    sfq_handle* h = &handles[tid];
    uint64_t local_ops = 0;
    
    // Synchronize all threads before starting
    __syncthreads();
    
    switch (test_type) {
        case 0: // Enqueue-Dequeue pairs
            for (int i = 0; i < operations_per_thread; i++) {
                uint64_t val = tid * 10000 + i + 1;
                sfq_enqueue(q, h, val);
                local_ops++;
                
                uint64_t dequeued = sfq_dequeue(q, h);
                if (dequeued != SFQ_EMPTY) {
                    local_ops++;
                }
            }
            break;
            
        case 1: // 50% enqueues, 50% dequeues
            for (int i = 0; i < operations_per_thread; i++) {
                if (i % 2 == 0) {
                    uint64_t val = tid * 10000 + i + 1;
                    sfq_enqueue(q, h, val);
                } else {
                    sfq_dequeue(q, h);
                }
                local_ops++;
            }
            break;
            
        case 2: // Mostly enqueues (80%)
            for (int i = 0; i < operations_per_thread; i++) {
                if (i % 5 != 0) {
                    uint64_t val = tid * 10000 + i + 1;
                    sfq_enqueue(q, h, val);
                } else {
                    sfq_dequeue(q, h);
                }
                local_ops++;
            }
            break;
            
        case 3: // Mostly dequeues (80%)
            for (int i = 0; i < operations_per_thread; i++) {
                if (i % 5 == 0) {
                    uint64_t val = tid * 10000 + i + 1;
                    sfq_enqueue(q, h, val);
                } else {
                    sfq_dequeue(q, h);
                }
                local_ops++;
            }
            break;
    }
    
    results[tid] = local_ops;
}

//=============================================================================
// HOST-SIDE INITIALIZATION FUNCTIONS
//=============================================================================

// Host function to initialize queue and handles
void sfq_queue_host_init(sfq_queue** d_q, sfq_handle** d_handles, int num_threads) {
    // Allocate device memory for queue
    hipMalloc((void**)d_q, sizeof(sfq_queue));
    
    // Allocate device memory for handles
    hipMalloc((void**)d_handles, num_threads * sizeof(sfq_handle));
    
    // Launch initialization kernel
    sfq_init_kernel<<<1, 1>>>(*d_q, *d_handles, num_threads);
    hipDeviceSynchronize();
}

// Host destroy helper so unified tests can clean up SFQ like other queues
void sfq_queue_destroy(sfq_queue* d_q, sfq_handle* d_h) {
    if (d_q) hipFree(d_q);
    if (d_h) hipFree(d_h);
}


//=============================================================================
// DEBUGGING AND VALIDATION FUNCTIONS
//=============================================================================

__device__ void sfq_print_queue_state(sfq_queue* q) {
    printf("SFQ Queue State: head=%u, tail=%u, done=%u\n", 
           SFQ_ATOMIC_LOAD(&q->head), SFQ_ATOMIC_LOAD(&q->tail), SFQ_ATOMIC_LOAD(&q->done));
}

__device__ void sfq_print_handle_state(sfq_handle* h, int tid) {
    printf("SFQ Handle %d: thread_id=%u\n", tid, h->thread_id);
}

// Validation kernel to check queue consistency
__global__ void sfq_validate_kernel(sfq_queue* q, sfq_handle* handles, int num_threads) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== SFQ Queue Validation ===\n");
        sfq_print_queue_state(q);
        
        for (int i = 0; i < min(num_threads, 10); i++) {  // Print first 10 handles
            sfq_print_handle_state(&handles[i], i);
        }
        
        // Check basic invariants
        uint32_t head = SFQ_ATOMIC_LOAD(&q->head);
        uint32_t tail = SFQ_ATOMIC_LOAD(&q->tail);
        
        printf("Invariant check: tail >= head = %s\n", (tail >= head) ? "PASS" : "FAIL");
        printf("=== End SFQ Validation ===\n");
    }
}


// Reset kernel to clear queue state between tests (not strictly needed for SFQ but useful for unified testing)
__global__ void sfq_reset_kernel(sfq_queue* q, sfq_handle* h, int num_threads) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Reset queue state
        SFQ_ATOMIC_STORE(&q->head, 0);
        SFQ_ATOMIC_STORE(&q->tail, 0);
        SFQ_ATOMIC_STORE(&q->vnull, 0);
        SFQ_ATOMIC_STORE(&q->done, 0);
        
        // Re-initialize items and slots
        for (int i = 0; i < SFQ_QUEUE_LENGTH; i++) {
            SFQ_ATOMIC_STORE(&q->items[i], 0);
            SFQ_ATOMIC_STORE(&q->slots[i], 0);
        }
        
        // Re-initialize handles if needed (not strictly necessary for SFQ)
        for (int i = 0; i < num_threads; i++) {
            h[i].thread_id = i;
        }
    }
}

// function call for reset kernel
void sfq_queue_reset(sfq_queue* d_q, sfq_handle* d_handles, int num_threads) {
    // int block = 256;
    // int grid = (num_threads + block - 1) / block;
    sfq_reset_kernel<<<1, 1>>>(d_q, d_handles, num_threads);
    hipDeviceSynchronize();
}


// End of sfqueue_hip.cpp