#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <device_atomic_functions.h>

#include "def_types.h"
#include "primitives/fp_array.cuh"
#include "primitives/trace_access.h"

namespace deferral {

struct DeferralPoseidon2Count {
    uint32_t compress_mult;
    uint32_t capacity_mult;
};

struct DeferralPoseidon2Buffer {
    FpArray<16> *records;
    DeferralPoseidon2Count *counts;
    uint32_t *idx;
    size_t capacity;

    __device__ DeferralPoseidon2Buffer(
        FpArray<16> *records,
        DeferralPoseidon2Count *counts,
        uint32_t *idx,
        size_t capacity
    )
        : records(records), counts(counts), idx(idx), capacity(capacity) {}

    __device__ void record(RowSlice left, RowSlice right, bool is_compress) {
        FpArray<16> value;
        FpArray<8> left_array = FpArray<8>::from_row(left, DIGEST_SIZE);
        FpArray<8> right_array = FpArray<8>::from_row(right, DIGEST_SIZE);

#pragma unroll
        for (size_t i = 0; i < DIGEST_SIZE; ++i) {
            value.v[i] = left_array.v[i];
            value.v[i + DIGEST_SIZE] = right_array.v[i];
        }

        const uint32_t record_idx = atomicAdd(idx, 1);
        assert(record_idx < capacity && "DeferralPoseidon2Buffer overflow");
        records[record_idx] = value;
        counts[record_idx] = {
            is_compress ? 1u : 0u,
            is_compress ? 0u : 1u,
        };
    }
};

} // namespace deferral
