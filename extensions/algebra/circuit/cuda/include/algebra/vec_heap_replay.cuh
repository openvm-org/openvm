#pragma once

#include "primitives/trace_access.h"
#include "riscv-adapters/vec_heap.cuh"
#include "riscv-adapters/vec_heap_replay.cuh"

#include <cstddef>
#include <cstdint>

template <size_t NUM_READS, size_t BLOCKS>
static __device__ void fill_vec_heap_adapter_from_projection(
    RowSlice row,
    VecHeapTraceInput<NUM_READS, BLOCKS> const &input,
    VariableRangeChecker range_checker,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS> record = {};
    record.from_pc = input.from_pc;
    record.from_timestamp = input.from_timestamp;
    record.rd_ptr = input.rd_ptr;
    record.rd_val = input.rd_val;
    record.rd_read_aux.prev_timestamp = input.rd_prev_timestamp;
    for (size_t read = 0; read < NUM_READS; read++) {
        record.rs_ptrs[read] = input.rs_ptrs[read];
        record.rs_vals[read] = input.rs_vals[read];
        record.rs_read_aux[read].prev_timestamp = input.rs_prev_timestamps[read];
        for (size_t block = 0; block < BLOCKS; block++) {
            record.reads_aux[read][block].prev_timestamp =
                input.heap_prev_timestamps[read][block];
        }
    }
    for (size_t block = 0; block < BLOCKS; block++) {
        record.writes_aux[block].prev_timestamp = input.write_prev_timestamps[block];
        for (size_t limb = 0; limb < BLOCK_FE_WIDTH; limb++) {
            uint16_t packed = input.write_predecessors[block][limb];
            record.writes_aux[block].prev_data[2 * limb] = static_cast<uint8_t>(packed);
            record.writes_aux[block].prev_data[2 * limb + 1] =
                static_cast<uint8_t>(packed >> 8);
        }
    }
    VecHeapAdapter<NUM_READS, BLOCKS, BLOCKS> adapter(
        pointer_max_bits, range_checker, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record);
}
