#pragma once

#include <cstddef>

// Returns the first index i in [0, len] such that arr[i] > value.
// Assumes arr is non-decreasing and has length at least len.
template <typename T>
__device__ __forceinline__ size_t partition_point_leq(
    const T *arr,
    size_t len,
    T value
) {
    size_t lo = 0;
    size_t hi = len;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (arr[mid] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

