// sfqueue_hip.hpp - Native HIP implementation of SFQ (Slot-based FIFO Queue)
// This is the header file that matches your naming convention

#ifndef SFQUEUE_HIP_H
#define SFQUEUE_HIP_H

#include <hip/hip_runtime.h>
#include <stdint.h>

//=============================================================================
// SFQ CONFIGURATION - Matching your OpenCL version
//=============================================================================
#ifndef SFQ_QUEUE_LENGTH
// #define SFQ_QUEUE_LENGTH 65536
#define SFQ_QUEUE_LENGTH 131072
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
struct sfq_queue {
    volatile uint32_t head;
    volatile uint32_t tail; 
    volatile uint32_t vnull;
    volatile uint32_t done;
    union {
        volatile uint32_t items[SFQ_QUEUE_LENGTH];
        volatile uint32_t nodes[SFQ_QUEUE_LENGTH];
    };
    volatile uint32_t slots[SFQ_QUEUE_LENGTH];
    uint32_t nprocs;  // For compatibility with test framework
};

// Handle structure for compatibility with test interface
struct sfq_handle {
    uint32_t thread_id;  // Simple thread identification
    uint32_t dummy[15];  // Padding to match wf_handle size roughly
};

//=============================================================================
// DEVICE FUNCTION DECLARATIONS
//=============================================================================

// Main API functions
__device__ void sfq_enqueue(sfq_queue* q, sfq_handle* h, uint64_t v);
__device__ uint64_t sfq_dequeue(sfq_queue* q, sfq_handle* h);

// Core SFQ operations
__device__ int sfq_enqueue_slot(sfq_queue* q, uint32_t item);
__device__ int sfq_dequeue_slot(sfq_queue* q, uint32_t* p);
__device__ int sfq_enqueue_nb_slot(sfq_queue* q, uint32_t item);
__device__ int sfq_dequeue_nb_slot(sfq_queue* q, uint32_t* p);

// Initialization functions  
__device__ void sfq_queue_init(sfq_queue* q, uint32_t nprocs);
__device__ void sfq_handle_init(sfq_handle* h, sfq_queue* q, sfq_handle* next_handle, 
                               sfq_handle* enq_helper, sfq_handle* deq_helper);

//=============================================================================
// KERNEL FUNCTION DECLARATIONS
//=============================================================================

// Initialization kernels
// __global__ void sfq_init_kernel(sfq_queue* q, sfq_handle* handles, int num_threads);
void sfq_queue_reset(sfq_queue* d_q, sfq_handle* d_handles, int num_threads);

// Validation and debugging kernels
__global__ void sfq_validate_kernel(sfq_queue* q, sfq_handle* handles, int num_threads);

// Test kernels (these match the generic interface)
__global__ void sfq_simple_test_kernel(sfq_queue* q, sfq_handle* handles, 
                                      uint64_t* test_data, int num_ops);

__global__ void sfq_performance_test_kernel(sfq_queue* q, sfq_handle* handles, 
                                           uint64_t* results, int operations_per_thread,
                                           int test_type);

// reset kernels
__global__ void sfq_reset_kernel(sfq_queue* q, sfq_handle* h, int num_threads);

//=============================================================================
// HOST FUNCTION DECLARATIONS
//=============================================================================

// Host-side initialization
void sfq_queue_host_init(sfq_queue** d_q, sfq_handle** d_handles, int num_threads);
// Host-side destroy (used by unified test harness)
void sfq_queue_destroy(sfq_queue* d_q, sfq_handle* d_h);
void sfq_queue_reset(sfq_queue* d_q, sfq_handle* d_handles, int num_threads);


//=============================================================================
// DEBUG FUNCTIONS
//=============================================================================
__device__ void sfq_print_queue_state(sfq_queue* q);
__device__ void sfq_print_handle_state(sfq_handle* h, int tid);

#endif // SFQUEUE_HIP_H