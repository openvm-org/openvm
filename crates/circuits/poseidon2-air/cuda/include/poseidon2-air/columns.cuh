#pragma once

#include "fp.h"
#include "primitives/trace_access.h"

namespace poseidon2 {

template <typename T, size_t DEGREE, size_t REGISTERS> struct SBox {
    T registers[REGISTERS];
};

template <typename T, size_t WIDTH, size_t SBOX_DEGREE, size_t SBOX_REGISTERS> struct FullRound {
    /// Possible intermediate results within each S-box.
    SBox<T, SBOX_DEGREE, SBOX_REGISTERS> sbox[WIDTH];
    /// The post-state, i.e. the entire layer after this full round.
    T post[WIDTH];
};

template <typename T, size_t WIDTH, size_t SBOX_DEGREE, size_t SBOX_REGISTERS> struct PartialRound {
    /// Possible intermediate results within the S-box.
    SBox<T, SBOX_DEGREE, SBOX_REGISTERS> sbox;
    /// The output of the S-box.
    T post;
};

template <
    typename T,
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGISTERS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
struct Poseidon2SubCols {
    T export_col;
    T inputs[WIDTH];
    FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS> beginning_full_rounds[HALF_FULL_ROUNDS];
    PartialRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS> partial_rounds[PARTIAL_ROUNDS];
    FullRound<T, WIDTH, SBOX_DEGREE, SBOX_REGISTERS> ending_full_rounds[HALF_FULL_ROUNDS];
};

template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
struct Poseidon2Row {
    // Single RowSlice for entire Poseidon2 subrow
    RowSlice slice;

    template <typename T>
    using Cols =
        Poseidon2SubCols<T, WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

    __device__ Poseidon2Row(Fp *input, uint32_t n) : slice(input, n) {}

    __device__ Poseidon2Row(RowSlice slice) : slice(std::move(slice)) {}

    // Null Poseidon2Row definition
    __device__ static Poseidon2Row null() { return Poseidon2Row(RowSlice::null()); }

    __device__ bool is_valid() const { return slice.is_valid(); }

    // Basic accessors
    __device__ RowSlice export_col() const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, export_col));
    }

    __device__ RowSlice inputs() const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, inputs));
    }

    // Beginning full rounds accessors
    __device__ RowSlice beginning_full_sbox(size_t round, size_t lane) const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, beginning_full_rounds[round].sbox[lane]));
    }

    __device__ RowSlice beginning_full_post(size_t round) const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, beginning_full_rounds[round].post));
    }

    // Partial rounds accessors
    __device__ RowSlice partial_sbox(size_t round) const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, partial_rounds[round].sbox));
    }

    __device__ RowSlice partial_post(size_t round) const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, partial_rounds[round].post));
    }

    // Ending full rounds accessors
    __device__ RowSlice ending_full_sbox(size_t round, size_t lane) const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, ending_full_rounds[round].sbox[lane]));
    }

    __device__ RowSlice ending_full_post(size_t round) const {
        if (!is_valid()) {
            return RowSlice::null();
        }
        return slice.slice_from(COL_INDEX(Cols, ending_full_rounds[round].post));
    }
    __device__ RowSlice outputs() const { return ending_full_post(HALF_FULL_ROUNDS - 1); }

    // Helper to get total size needed for the buffer
    static constexpr size_t get_total_size() { return sizeof(Cols<uint8_t>); }
};

} // namespace poseidon2
