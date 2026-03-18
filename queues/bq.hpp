#pragma once

// =============================================================================
// Broker Queue — HIP Implementation (AMD ROCm + NVIDIA CUDA)
//
// Based on: "The Broker Queue: A Fast, Linearizable FIFO Queue for
//            Fine-Granular Work Distribution on the GPU"
//            Kerbl et al., ICS 2018
//
// Three variants:
//   BrokerQueue<T,N>         — fully linearizable FIFO (BQ)
//   .enqueue_bwd / dequeue_bwd — non-linearizable fast path (BWD)
//   BrokerStealingQueue<T,N> — per-block queues with work stealing (BSQ)
//
// Requirements:
//   - N must be a power of 2
//   - MAX_THREADS must be ≥ actual concurrent threads touching the queue
//   - Elements must be trivially copyable
//   - hipcc (ROCm ≥ 5.0) or CUDA via HIP compatibility layer
// =============================================================================

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace bq {

// ---------------------------------------------------------------------------
// Compile-time log2 — only valid for powers of 2
// ---------------------------------------------------------------------------
template<uint32_t V> struct ConstLog2 {
    static constexpr uint32_t value = 1u + ConstLog2<V / 2u>::value;
};
template<> struct ConstLog2<1u> { static constexpr uint32_t value = 0u; };

// ---------------------------------------------------------------------------
// Status returned by enqueue / dequeue
// ---------------------------------------------------------------------------
enum class QueueStatus : uint32_t {
    Success = 0,
    Full    = 1,
    Empty   = 2
};

// ---------------------------------------------------------------------------
// BrokerQueue<T, N, MAX_THREADS>
//
// T           — element type (trivially copyable)
// N           — ring buffer capacity (power of 2; e.g. 1024, 4096, …)
// MAX_THREADS — safe upper bound on threads concurrently accessing the queue
//               Used to compute the head/tail distance margin for Full/Empty
//               detection.  Larger → more conservative; smaller → tighter
//               but risks missed detections.  Default 65536 is safe for full
//               GPU occupancy on current hardware.
//
// Memory: place this struct in GPU global memory (hipMalloc / __device__ global)
// ---------------------------------------------------------------------------
template<typename T, uint32_t N, uint32_t MAX_THREADS = 65536u>
struct BrokerQueue {

    static_assert(std::is_trivially_copyable<T>::value,
                  "BrokerQueue element type must be trivially copyable");
    static_assert((N & (N - 1u)) == 0u,
                  "BrokerQueue size N must be a power of 2");
    static_assert(N >= 2u,       "N must be at least 2");
    static_assert(MAX_THREADS >= 2u, "MAX_THREADS must be at least 2");
    static_assert(
        (uint64_t)N + (uint64_t)(MAX_THREADS / 2u) < (uint64_t)0x7FFF'FFFFu,
        "N + MAX_THREADS/2 must fit in int32_t for correct broker arithmetic");

    static constexpr uint32_t LOG2N    = ConstLog2<N>::value;
    static constexpr uint32_t MASK     = N - 1u;
    static constexpr uint32_t HALF_MAX = MAX_THREADS / 2u;

    // -------------------------------------------------------------------------
    // Data layout — aggressive alignment to avoid false sharing and maximise
    // L1/L2 cache efficiency on both GCN/RDNA and SM architectures.
    // -------------------------------------------------------------------------

    // Ring buffer: one cache line per 16 elements (for T=uint32_t).
    // Placed first so ring[0] is at the struct base address.
    alignas(128) T        ring[N];

    // Per-slot ticket array.
    // Even value 2k  → slot is ready for the k-th write.
    // Odd value  2k+1 → slot is ready for the k-th read.
    // After a read, slot is bumped to 2(k+1) for the next write cycle.
    // Initialised to 0 (= ready for first write).
    alignas(128) uint32_t tickets[N];

    // head and tail packed into a single 64-bit word for atomic snapshot reads.
    // layout (little-endian): bits[31:0]=head, bits[63:32]=tail
    //
    // Rationale: Full/Empty detection needs a consistent (head,tail) pair.
    // We do a single atomicAdd(&packed,0) to get a coherent 64-bit snapshot.
    // Individual 32-bit atomicAdds on head/tail still work because GPU hardware
    // guarantees sub-word atomicity at natural widths — a 32-bit atomic on tail
    // does not disturb head and vice-versa.
    alignas(8) union {
        struct { volatile uint32_t head;   // dequeue pointer (lo 32)
                 volatile uint32_t tail; }; // enqueue pointer (hi 32)
        unsigned long long head_tail_packed;
    };

    // Broker counter — models outstanding enqueue/dequeue commitments.
    // Range: [-(int)HALF_MAX, (int)(N + HALF_MAX)]
    // Kept on its own cache line to avoid false sharing with head/tail.
    alignas(64) int32_t count;

    // =========================================================================
    // Initialisation
    // =========================================================================

    // Call on host before first kernel launch.
    void host_init() noexcept {
        memset(tickets, 0, sizeof(tickets));
        head_tail_packed = 0ULL;
        count            = 0;
    }

    // Call once from device (single thread, or cooperatively from a block).
    __device__ void device_init_cooperative() {
        for (uint32_t i = threadIdx.x; i < N; i += blockDim.x)
            tickets[i] = 0u;
        if (threadIdx.x == 0u) {
            head_tail_packed = 0ULL;
            count            = 0;
        }
        __syncthreads();
    }

    // =========================================================================
    // Public interface — BQ (linearizable FIFO)
    // =========================================================================

    // Enqueue element.  Blocks until:
    //   (a) a slot is definitively available (returns Success), or
    //   (b) queue is observably full (returns Full).
    __device__ QueueStatus enqueue(const T& elem) {
        while (!ensure_enqueue()) {
            uint32_t h, t;
            load_ht(h, t);
            uint32_t d = t - h;
            // Full: tail advanced ≥N ahead of head, within the safety margin
            if (d >= N && d < N + HALF_MAX)
                return QueueStatus::Full;
        }
        put_data(elem);
        return QueueStatus::Success;
    }

    // Dequeue element.  Blocks until:
    //   (a) an element is definitively available (returns Success), or
    //   (b) queue is observably empty (returns Empty).
    __device__ QueueStatus dequeue(T& elem) {
        while (!ensure_dequeue()) {
            uint32_t h, t;
            load_ht(h, t);
            // Empty: head ≥ tail in logical sense.
            // In unsigned: t - h - 1 wraps to a large value when h ≥ t.
            uint32_t d = t - h - 1u;
            if (d >= N + HALF_MAX)
                return QueueStatus::Empty;
        }
        elem = read_data();
        return QueueStatus::Success;
    }

    // =========================================================================
    // Public interface — BWD (Broker Work Distributor, non-linearizable)
    //
    // Omits the retry loop on ensure_*; returns Full/Empty immediately if the
    // broker sees no available slot.  Faster but Count may transiently lie
    // about the true state.  Fine for work distribution; do NOT use when
    // strict FIFO ordering matters.
    // =========================================================================

    __device__ QueueStatus enqueue_bwd(const T& elem) {
        if (!ensure_enqueue()) return QueueStatus::Full;
        put_data(elem);
        return QueueStatus::Success;
    }

    __device__ QueueStatus dequeue_bwd(T& elem) {
        if (!ensure_dequeue()) return QueueStatus::Empty;
        elem = read_data();
        return QueueStatus::Success;
    }

    // =========================================================================
    // Batch dequeue — drain up to `n` elements into caller-provided array.
    // Returns the number of elements actually dequeued.
    // Useful in cooperative block execution (one thread drains for the warp).
    // =========================================================================
    __device__ uint32_t dequeue_batch(T* out, uint32_t n) {
        uint32_t got = 0u;
        for (uint32_t i = 0u; i < n; ++i) {
            T elem;
            if (dequeue(elem) != QueueStatus::Success) break;
            out[got++] = elem;
        }
        return got;
    }

    // Approximate element count (not precise under heavy contention).
    __device__ int32_t approx_size() const {
        return load_i32(&count);
    }

    // =========================================================================
    // Private helpers
    // =========================================================================
private:

    // Atomic load helpers — use atomicAdd(x,0) / atomicAdd(x,0ull) as the
    // standard GPU idiom for an acquire-style atomic read.
    __device__ __forceinline__
    static uint32_t load_u32(const volatile uint32_t* addr) {
        return atomicAdd(const_cast<uint32_t*>(addr), 0u);
    }
    __device__ __forceinline__
    static int32_t load_i32(const int32_t* addr) {
        return atomicAdd(const_cast<int32_t*>(addr), 0);
    }

    // Consistent 64-bit snapshot of (head, tail).
    __device__ __forceinline__
    void load_ht(uint32_t& h, uint32_t& t) const {
        unsigned long long ht =
            atomicAdd(const_cast<unsigned long long*>(&head_tail_packed), 0ULL);
        h = static_cast<uint32_t>(ht         & 0xFFFF'FFFFull);
        t = static_cast<uint32_t>(ht >> 32u  & 0xFFFF'FFFFull);
    }

    // Spin until tickets[p] reaches the expected value.
    // __threadfence() prevents the compiler / hardware from serving the load
    // from a stale register and ensures forward progress on both RDNA and SM.
    __device__ __forceinline__
    void wait_for_ticket(uint32_t p, uint32_t expected) const {
        while (load_u32(&tickets[p]) != expected)
            __threadfence();
    }

    // -------------------------------------------------------------------------
    // put_data: write an element into the ring buffer.
    // Called only after ensure_enqueue() returned true.
    // -------------------------------------------------------------------------
    __device__ __noinline__ void put_data(const T& elem) {
        // Reserve a tail slot.  FAA gives us a globally unique position.
        uint32_t pos = atomicAdd(const_cast<uint32_t*>(&tail), 1u);
        uint32_t p   = pos & MASK;
        uint32_t lap = pos >> LOG2N;          // which lap of the ring buffer

        // Wait for the slot to finish its previous read cycle (even ticket).
        wait_for_ticket(p, 2u * lap);

        // Write the element (non-atomic; ticket acts as the synchronisation gate).
        ring[p] = elem;

        // Ensure the write reaches L2 / global memory before we advance the
        // ticket, which is the release signal to the waiting reader.
        __threadfence();

        // Advance ticket to odd → slot is ready for dequeue.
        atomicExch(const_cast<uint32_t*>(&tickets[p]), 2u * lap + 1u);
    }

    // -------------------------------------------------------------------------
    // read_data: consume an element from the ring buffer.
    // Called only after ensure_dequeue() returned true.
    // -------------------------------------------------------------------------
    __device__ __noinline__ T read_data() {
        uint32_t pos = atomicAdd(const_cast<uint32_t*>(&head), 1u);
        uint32_t p   = pos & MASK;
        uint32_t lap = pos >> LOG2N;

        // Wait for the slot's enqueue to complete (odd ticket).
        wait_for_ticket(p, 2u * lap + 1u);

        // Read the element.
        T elem = ring[p];

        __threadfence();

        // Advance ticket to even (next lap) → slot ready for next enqueue.
        atomicExch(const_cast<uint32_t*>(&tickets[p]), 2u * (lap + 1u));

        return elem;
    }

    // -------------------------------------------------------------------------
    // ensure_enqueue: broker — atomically reserve a slot for enqueue.
    //
    // Increments Count optimistically; rolls back if Count ≥ N (full).
    // Returns true iff this thread is now committed to enqueue one element.
    //
    // Loop is O(1) amortised: at most two CAS-free atomic ops per iteration.
    // -------------------------------------------------------------------------
    __device__ __forceinline__ bool ensure_enqueue() {
        int32_t num = load_i32(&count);
        for (;;) {
            if (num >= static_cast<int32_t>(N)) return false;
            int32_t old = atomicAdd(&count, 1);
            if (old < static_cast<int32_t>(N))  return true;
            // Rollback: another thread snuck in and filled the queue.
            num = atomicAdd(&count, -1) - 1;
        }
    }

    // -------------------------------------------------------------------------
    // ensure_dequeue: broker — atomically reserve an item for dequeue.
    //
    // Decrements Count optimistically; rolls back if Count ≤ 0 (empty).
    // -------------------------------------------------------------------------
    __device__ __forceinline__ bool ensure_dequeue() {
        int32_t num = load_i32(&count);
        for (;;) {
            if (num <= 0) return false;
            int32_t old = atomicAdd(&count, -1);
            if (old > 0) return true;
            // Rollback.
            num = atomicAdd(&count, 1) + 1;
        }
    }
};

// =============================================================================
// BrokerStealingQueue<T, N, NUM_QUEUES, MAX_THREADS>
//
// One BrokerQueue per logical worker (typically one per GPU block).
// Threads first try their own queue; on Empty they walk all queues to steal.
//
// Usage pattern:
//   uint32_t my_q = blockIdx.x % NUM_QUEUES;
//   bsq.enqueue(elem, my_q);
//   bsq.dequeue(elem, my_q);
// =============================================================================
template<typename T, uint32_t N, uint32_t NUM_QUEUES, uint32_t MAX_THREADS = 65536u>
struct BrokerStealingQueue {

    BrokerQueue<T, N, MAX_THREADS> queues[NUM_QUEUES];

    void host_init() {
        for (uint32_t i = 0u; i < NUM_QUEUES; ++i)
            queues[i].host_init();
    }

    __device__ QueueStatus enqueue(const T& elem, uint32_t my_queue) {
        return queues[my_queue].enqueue(elem);
    }

    // Try own queue; if empty, attempt to steal from all other queues.
    __device__ QueueStatus dequeue(T& elem, uint32_t my_queue) {
        QueueStatus s = queues[my_queue].dequeue_bwd(elem);
        if (s == QueueStatus::Success) return s;

        // Work stealing: probe queues in round-robin order.
        for (uint32_t offset = 1u; offset < NUM_QUEUES; ++offset) {
            uint32_t q = (my_queue + offset) % NUM_QUEUES;
            s = queues[q].dequeue_bwd(elem);
            if (s == QueueStatus::Success) return s;
        }
        return QueueStatus::Empty;
    }

    // Use BQ (linearizable) variant for dequeue — slower but correct ordering.
    __device__ QueueStatus dequeue_linearizable(T& elem, uint32_t my_queue) {
        QueueStatus s = queues[my_queue].dequeue(elem);
        if (s == QueueStatus::Success) return s;

        for (uint32_t offset = 1u; offset < NUM_QUEUES; ++offset) {
            uint32_t q = (my_queue + offset) % NUM_QUEUES;
            s = queues[q].dequeue(elem);
            if (s == QueueStatus::Success) return s;
        }
        return QueueStatus::Empty;
    }
};

} // namespace bq