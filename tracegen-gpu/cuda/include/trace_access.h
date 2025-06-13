#pragma once

#include "fp.h"
#include <cstddef>
#include <cstdint>

/// A RowSlice is a contiguous section of a row in col-based trace.
struct RowSlice {
    Fp *ptr;
    uint32_t stride;

    __device__ RowSlice(Fp *ptr, uint32_t stride) : ptr(ptr), stride(stride) {}

    __device__ __forceinline__ Fp &operator[](size_t column_index) const {
        return ptr[column_index * stride];
    }

    template <typename T>
    __device__ __forceinline__ void write(size_t column_index, T value) const {
        ptr[column_index * stride] = value;
    }

    template <typename T>
    __device__ __forceinline__ void write_array(size_t column_index, size_t length, const T *values)
        const {
#pragma unroll
        for (size_t i = 0; i < length; i++) {
            ptr[column_index * stride + i] = values[i];
        }
    }

    __device__ __forceinline__ RowSlice slice_from(size_t column_index) const {
        return RowSlice(ptr + column_index * stride, stride);
    }
};

/// Compute the 0-based column index of member `FIELD` within struct template `STRUCT<T>`,
/// by instantiating it as `STRUCT<uint8_t>` so that offsetof yields the element index.
#define COL_INDEX(STRUCT, FIELD) (offsetof(STRUCT<uint8_t>, FIELD))

/// Compute the fixed array length of `FIELD` within `STRUCT<T>`
#define COL_ARRAY_LEN(STRUCT, FIELD)                                                               \
    (sizeof(STRUCT<uint8_t>::FIELD) / sizeof(STRUCT<uint8_t>::FIELD[0]))

/// Write a single value into `FIELD` of struct `STRUCT<T>` at a given row.
#define COL_WRITE_VALUE(ROW, STRUCT, FIELD, VALUE) (ROW).write(COL_INDEX(STRUCT, FIELD), VALUE)

/// Write an array of values into the fixed‐length `FIELD` array of `STRUCT<T>` for one row.
#define COL_WRITE_ARRAY(ROW, STRUCT, FIELD, VALUES)                                                \
    (ROW).write_array(COL_INDEX(STRUCT, FIELD), COL_ARRAY_LEN(STRUCT, FIELD), VALUES)
