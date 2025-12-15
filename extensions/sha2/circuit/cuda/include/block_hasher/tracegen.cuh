#pragma once

#include "columns.cuh"
#include "fp.h"
#include "primitives/trace_access.h"
#include <cstddef>
#include <cstdint>

// NOTE: This is a stubbed tracegen implementation to get the CUDA pipeline compiling again.
// It fills rows with zeros and basic flags/request_id; the full round/digest population
// matching the Rust AIR still needs to be implemented.
namespace sha2 {

template <typename V>
__device__ inline void write_round_stub(
    RowSlice row,
    uint32_t request_id,
    uint32_t global_block_idx,
    uint32_t local_row_idx
) {
    row.fill_zero(0, Sha2Layout<V>::ROUND_WIDTH);
    RowSlice inner = row.slice_from(Sha2Layout<V>::INNER_OFFSET);
    // Mark round rows within the block
    if (local_row_idx < V::ROUND_ROWS) {
        SHA2INNER_WRITE_ROUND(V, inner, flags.is_round_row, Fp::one());
        SHA2INNER_WRITE_ROUND(
            V,
            inner,
            flags.is_first_4_rows,
            (local_row_idx < static_cast<uint32_t>(V::MESSAGE_ROWS)) ? Fp::one() : Fp::zero()
        );
        SHA2INNER_WRITE_ROUND(V, inner, flags.is_digest_row, Fp::zero());
        SHA2INNER_WRITE_ROUND(V, inner, flags.global_block_idx, global_block_idx);
    } else {
        // digest rows
        SHA2INNER_WRITE_DIGEST(V, inner, flags.is_round_row, Fp::zero());
        SHA2INNER_WRITE_DIGEST(V, inner, flags.is_first_4_rows, Fp::zero());
        SHA2INNER_WRITE_DIGEST(V, inner, flags.is_digest_row, Fp::one());
        SHA2INNER_WRITE_DIGEST(V, inner, flags.global_block_idx, global_block_idx);
    }
    // Write request_id in the wrapper column
    SHA2_WRITE_ROUND(V, row, request_id, Fp(request_id));
}

template <typename V>
__global__ void sha2_block_tracegen_stub(
    Fp *trace,
    size_t trace_height,
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= trace_height) {
        return;
    }

    RowSlice row(trace + row_idx, trace_height);
    row.fill_zero(0, Sha2Layout<V>::WIDTH);

    uint32_t record_idx = row_idx / V::ROWS_PER_BLOCK;
    uint32_t local_row = row_idx % V::ROWS_PER_BLOCK;
    if (record_idx >= num_records) {
        return;
    }

    // Basic request_id and flags; actual round data is left zeroed for now.
    write_round_stub<V>(row, record_idx, record_idx + 1, local_row);
}

} // namespace sha2
