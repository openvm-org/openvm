#pragma once

#include "trace_access.h"

__host__ __device__ static inline uint32_t compute_k(
    uint32_t num_flags,
    uint32_t max_degree,
    bool reserve_invalid
) {
    if (reserve_invalid) {
        num_flags++;
    }
    uint64_t b = 1;
    uint32_t k = 0;
    while (b < num_flags) {
        ++k;
        b = (b * (max_degree + k)) / k;
    }
    return k;
}

// k is the dimension needed to represent some number of flags, including
// reserved (0, ..., 0) point if reserve_valid is true. To use this struct
// in a kernel, best practice in most cases is to use compute_k above in
// the launcher code and pass it into the kernel by value.
struct Encoder {
    uint32_t num_flags;
    uint32_t max_degree;
    bool reserve_invalid;
    uint32_t k;

    __device__ constexpr Encoder(
        uint32_t num_flags,
        uint32_t max_degree,
        bool reserve_invalid,
        uint32_t k
    )
        : num_flags(num_flags), max_degree(max_degree), reserve_invalid(reserve_invalid), k(k) {}

    __device__ Encoder(uint32_t num_flags, uint32_t max_degree, bool reserve_invalid)
        : Encoder(
              num_flags,
              max_degree,
              reserve_invalid,
              compute_k(num_flags, max_degree, reserve_invalid)
          ) {}

    __device__ uint32_t width() const { return k; }

    __device__ void write_flag_pt(RowSlice pt, uint32_t idx) const {
#ifdef CUDA_DEBUG
        assert(idx < num_flags);
        assert(this->k > 0);
#endif
        if (reserve_invalid) {
            idx++;
        }

        uint32_t d = this->max_degree;
        uint32_t k = this->k - 1;
        uint32_t binom = 1;
        for (uint32_t i = 1; i <= k; i++) {
            binom = (binom * (d + i)) / i;
        }

        // While processing pt[i], let k be the number of indices left
        // to be processed (i.e. bins) and d the number of times we can
        // increment any remaining index (i.e. balls). Whenever we have
        // binom = (d + k) choose k <= idx, we subtract binom from idx,
        // increment pt[i], and decrement d.
        for (uint32_t i = 0; i < this->k; i++) {
            uint32_t current = 0;
            while (binom <= idx) {
                current++;
                idx -= binom;
                binom = (d + k == 0) ? 0 : (binom * d) / (d + k);
                d--;
            }
            pt[i] = Fp(current);
            binom = (d + k == 0) ? 0 : (binom * k) / (d + k);
            k--;
        }
    }
};
