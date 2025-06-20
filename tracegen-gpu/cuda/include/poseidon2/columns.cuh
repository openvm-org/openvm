#pragma once
#include "fp.h"
#include "trace_access.h"

namespace poseidon2 {

template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
struct Poseidon2Row {
    /// Assumed that trace is contiguous in memory
    /// Assumed layout is [export_col, inputs, beginning_full, partial, ending_full]
    RowSlice slice; // Single RowSlice for all data

    // Memory layout constants
    static constexpr size_t EXPORT_COL_SIZE = 1;
    static constexpr size_t INPUTS_SIZE = WIDTH;
    static constexpr size_t BEGINNING_FULL_SIZE =
        WIDTH * SBOX_REGS * HALF_FULL_ROUNDS + WIDTH * HALF_FULL_ROUNDS;
    static constexpr size_t PARTIAL_SIZE = (SBOX_REGS + 1) * PARTIAL_ROUNDS;
    static constexpr size_t ENDING_FULL_SIZE =
        WIDTH * SBOX_REGS * HALF_FULL_ROUNDS + WIDTH * HALF_FULL_ROUNDS;

    static constexpr size_t TOTAL_SIZE =
        EXPORT_COL_SIZE + INPUTS_SIZE + BEGINNING_FULL_SIZE + PARTIAL_SIZE + ENDING_FULL_SIZE;

    __device__ Poseidon2Row(Fp *input, uint32_t n) : slice(input, n) {}

    // Basic accessors
    __device__ RowSlice export_col() const { return slice.slice_from(0); }

    __device__ RowSlice inputs() const { return slice.slice_from(EXPORT_COL_SIZE); }

    // Beginning full rounds accessors
    __device__ RowSlice beginning_full_sbox(size_t round, size_t lane) const {
        size_t offset = EXPORT_COL_SIZE + INPUTS_SIZE + (round * (WIDTH * SBOX_REGS + WIDTH)) +
                        (lane * SBOX_REGS);
        return slice.slice_from(offset);
    }

    __device__ RowSlice beginning_full_post(size_t round) const {
        size_t offset = EXPORT_COL_SIZE + INPUTS_SIZE + (round * (WIDTH * SBOX_REGS + WIDTH)) +
                        (WIDTH * SBOX_REGS);
        return slice.slice_from(offset);
    }

    // Partial rounds accessors
    __device__ RowSlice partial_sbox(size_t round) const {
        size_t offset =
            EXPORT_COL_SIZE + INPUTS_SIZE + BEGINNING_FULL_SIZE + (round * (SBOX_REGS + 1));
        return slice.slice_from(offset);
    }

    __device__ RowSlice partial_post(size_t round) const {
        size_t offset = EXPORT_COL_SIZE + INPUTS_SIZE + BEGINNING_FULL_SIZE +
                        (round * (SBOX_REGS + 1)) + SBOX_REGS;
        return slice.slice_from(offset);
    }

    // Ending full rounds accessors
    __device__ RowSlice ending_full_sbox(size_t round, size_t lane) const {
        size_t offset = EXPORT_COL_SIZE + INPUTS_SIZE + BEGINNING_FULL_SIZE + PARTIAL_SIZE +
                        (round * (WIDTH * SBOX_REGS + WIDTH)) + (lane * SBOX_REGS);
        return slice.slice_from(offset);
    }

    __device__ RowSlice ending_full_post(size_t round) const {
        size_t offset = EXPORT_COL_SIZE + INPUTS_SIZE + BEGINNING_FULL_SIZE + PARTIAL_SIZE +
                        (round * (WIDTH * SBOX_REGS + WIDTH)) + (WIDTH * SBOX_REGS);
        return slice.slice_from(offset);
    }

    // Helper to get total size needed for the buffer
    static constexpr size_t get_total_size() { return TOTAL_SIZE; }
};

} // namespace poseidon2