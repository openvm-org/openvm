#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "ptr_array.h"
#include "switch_macro.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename T> struct TranscriptCols {
    T proof_idx;
    T is_proof_start;

    T tidx;
    T is_sample;
    T mask[CHUNK];

    T permuted;
    T prev_state[WIDTH];
    T post_state[WIDTH];
};

struct TranscriptAirRecord {
    bool is_sample;
    bool permuted;
    uint8_t num_ops;
    uint32_t tidx;
    uint32_t state_idx;
};

template <size_t NUM_PROOFS>
__global__ void transcript_air_tracegen_kernel(
    Fp *trace,
    size_t height,
    const Array<uint32_t, NUM_PROOFS> row_bounds,
    const PtrArray<Fp, NUM_PROOFS> transcript_values,
    const PtrArray<Fp, NUM_PROOFS> start_states,
    const PtrArray<TranscriptAirRecord, NUM_PROOFS> records,
    Fp *poseidon2_buffer,
    const Array<uint32_t, NUM_PROOFS> poseidon2_buffer_offsets
) {
    uint32_t global_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + global_row_idx, height);

    uint32_t row_idx, num_rows, last_row_bound = 0;
    uint32_t proof_idx = NUM_PROOFS;
#pragma unroll
    for (uint32_t i = 0; i < NUM_PROOFS; i++) {
        uint32_t row_bound_i = row_bounds[i];
        num_rows = row_bound_i - last_row_bound;
        if (row_bound_i > global_row_idx) {
            proof_idx = i;
            row_idx = global_row_idx - last_row_bound;
            break;
        }
        last_row_bound = row_bound_i;
    }

    if (proof_idx == NUM_PROOFS) {
        row.fill_zero(0, sizeof(TranscriptCols<uint8_t>));
        return;
    }

    auto [is_sample, permuted, num_ops, tidx, state_idx] = records[proof_idx][row_idx];

    COL_WRITE_VALUE(row, TranscriptCols, proof_idx, proof_idx);
    COL_WRITE_VALUE(row, TranscriptCols, is_proof_start, row_idx == 0);
    COL_WRITE_VALUE(row, TranscriptCols, is_sample, is_sample);
    COL_WRITE_VALUE(row, TranscriptCols, tidx, tidx);
    COL_WRITE_VALUE(row, TranscriptCols, permuted, permuted);

#pragma unroll
    for (uint8_t i = 0; i < CHUNK; i++) {
        COL_WRITE_VALUE(row, TranscriptCols, mask[i], i < num_ops);
    }

    auto states = reinterpret_cast<const Array<Fp, WIDTH> *>(start_states[proof_idx]);
    Array<Fp, WIDTH> prev_state = states[state_idx];
    if (!is_sample) {
        for (uint8_t i = 0; i < num_ops; i++) {
            prev_state.arr[i] = transcript_values[proof_idx][tidx + i];
        }
    }
    COL_WRITE_ARRAY(row, TranscriptCols, prev_state, prev_state.arr);
    COL_WRITE_ARRAY(
        row, TranscriptCols, post_state, permuted ? states[state_idx + 1].arr : prev_state.arr
    );

    if (permuted) {
        auto array_buffer = reinterpret_cast<Array<Fp, WIDTH> *>(poseidon2_buffer);
        array_buffer[poseidon2_buffer_offsets[proof_idx] + state_idx] = prev_state;
    }
}

extern "C" int _transcript_air_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint32_t *h_row_bounds,
    Fp **d_transcript_values,
    Fp **d_start_states,
    TranscriptAirRecord **d_records,
    Fp *d_poseidon2_buffer,
    uint32_t *h_poseidon2_offsets,
    uint32_t num_proofs
) {
    assert(width == sizeof(TranscriptCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 256);

    SWITCH_BLOCK(
        num_proofs,
        NUM_PROOFS,
        (transcript_air_tracegen_kernel<NUM_PROOFS><<<grid, block>>>(
             d_trace,
             height,
             Array<uint32_t, NUM_PROOFS>(h_row_bounds),
             PtrArray<Fp, NUM_PROOFS>(d_transcript_values),
             PtrArray<Fp, NUM_PROOFS>(d_start_states),
             PtrArray<TranscriptAirRecord, NUM_PROOFS>(d_records),
             d_poseidon2_buffer,
             Array<uint32_t, NUM_PROOFS>(h_poseidon2_offsets)
        );),
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8
    )

    return CHECK_KERNEL();
}
