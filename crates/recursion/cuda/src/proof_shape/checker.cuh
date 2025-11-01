#pragma once

#include "primitives/histogram.cuh"

#include <cstddef>
#include <cstdint>

struct RangeChecker {
    lookup::Histogram hist;

    __device__ RangeChecker(uint32_t *global_hist, size_t num_bits)
        : hist(global_hist, 1 << num_bits) {}

    __device__ void add_count(uint32_t val) { hist.add_count(val); }
};

template <size_t N> struct PowerChecker {
    lookup::Histogram pow_hist;
    lookup::Histogram range_hist;

    __device__ PowerChecker(uint32_t *pow_hist, uint32_t *range_hist)
        : pow_hist(pow_hist, N), range_hist(range_hist, N) {}

    __device__ void add_pow_count(uint32_t log) { pow_hist.add_count(log); }
    __device__ void add_range_count(uint32_t val) { range_hist.add_count(val); }
};
