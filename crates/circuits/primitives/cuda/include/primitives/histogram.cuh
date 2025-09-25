#pragma once

#include "launcher.cuh"
#include "utils.cuh"
#include "trace_access.h"

/**
 * @file histogram.cuh
 * @brief Device-side helpers for local histogram accumulation in CUDA.
 */

static constexpr uint WARP_MASK = WARP_SIZE - 1;

namespace lookup {

struct Histogram {
    uint32_t *global_hist;
    uint32_t num_bins;

    __device__ Histogram() : global_hist(nullptr), num_bins(0) {}

    __device__ Histogram(uint32_t *global_hist, uint32_t num_bins)
        : global_hist(global_hist), num_bins(num_bins) {}

    __device__ void add_count(uint32_t idx) {
        if (idx < num_bins) {
            // Warp-level deduplicated atomicAdd
            auto curr_mask = __activemask();
            auto same_mask = __match_any_sync(curr_mask, idx);
            auto leader = __ffs(same_mask) - 1;

            // Only the leader does atomicAdd
            if ((threadIdx.x & WARP_MASK) == leader) {
                atomicAdd(&global_hist[idx], __popc(same_mask));
            }
        }
    }
};

} // namespace lookup

struct VariableRangeChecker {
    lookup::Histogram hist;

    __device__ VariableRangeChecker(uint32_t *global_hist, uint32_t num_bins)
        : hist(global_hist, num_bins) {}

    // Used by VariableRangeChecker to constrain value that can be represented
    // using max_bits bits.
    __device__ void add_count(uint32_t value, size_t max_bits) {
        uint32_t idx = (1 << max_bits) + value;
        hist.add_count(idx);
    }

    __device__ void merge(uint32_t *global_hist) {
        // Does nothing if we don't use shared or local histogram
    }

    __device__ uint32_t max_bits() const { return 30 - __clz(hist.num_bins); }

    __device__ __forceinline__ void decompose(
        uint32_t x,
        size_t bits,
        RowSlice limbs,
        const size_t limbs_len
    ) {
        size_t range_max_bits = max_bits();
#ifdef CUDA_DEBUG
        assert(limbs_len >= d_div_ceil(bits, range_max_bits));
#endif
        uint32_t mask = (1 << range_max_bits) - 1;
        size_t bits_remaining = bits;
#pragma unroll
        for (int i = 0; i < limbs_len; i++) {
            uint32_t limb_u32 = x & mask;
            limbs[i] = limb_u32;
            add_count(limb_u32, min(bits_remaining, range_max_bits));
            x >>= range_max_bits;
            bits_remaining -= min(bits_remaining, range_max_bits);
        }
#ifdef CUDA_DEBUG
        assert(bits_remaining == 0 && x == 0);
#endif
    }
};

template <uint32_t N> struct RangeTupleChecker {
    uint32_t sizes[N];
    lookup::Histogram hist;

    __device__ RangeTupleChecker(uint32_t *global_hist, uint32_t sizes[N]) {
        uint32_t num_bins = 1;
        for (int i = 0; i < N; i++) {
            this->sizes[i] = sizes[i];
            num_bins *= this->sizes[i];
        }
        hist = lookup::Histogram(global_hist, num_bins);
    }

    __device__ void add_count(uint32_t values[N]) {
        uint32_t idx = 0;
        for (int i = 0; i < N; i++) {
            idx = idx * sizes[i] + values[i];
        }
        hist.add_count(idx);
    }

    __device__ void add_count(RowSlice values) {
        uint32_t idx = 0;
        for (int i = 0; i < N; i++) {
            idx = idx * sizes[i] + values[i].asUInt32();
        }
        hist.add_count(idx);
    }

    __device__ void merge(uint32_t *global_hist) {
        // Does nothing if we don't use shared or local histogram
    }
};

// Histogram for BitwiseOperationLookup, which either does a range check
// or an XOR check for two field elements at a time. We expect global_hist
// to be of size 2 * 2^num_bits, where the first 2^num_bits elements store
// the range check histogram and the rest store for XOR.
struct BitwiseOperationLookup {
    uint32_t num_bits;
    uint32_t num_rows;
    lookup::Histogram hist;

    __device__ BitwiseOperationLookup(uint32_t *global_hist, uint32_t num_bits)
        : num_bits(num_bits), num_rows(1 << (num_bits << 1)), hist(global_hist, num_rows << 1) {}

    __device__ void add_range(uint32_t x, uint32_t y) {
        uint32_t idx = x * (1 << num_bits) + y;
        if (idx < num_rows) {
            hist.add_count(idx);
        }
    }

    __device__ void add_xor(uint32_t x, uint32_t y) {
        uint32_t idx = x * (1 << num_bits) + y;
        if (idx < num_rows) {
            hist.add_count(idx + num_rows);
        }
    }

    __device__ void merge(uint32_t *global_hist) {
        // Does nothing if we don't use shared or local histogram
    }
};
