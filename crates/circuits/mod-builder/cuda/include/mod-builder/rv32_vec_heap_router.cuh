#pragma once

#include "meta.cuh"
#include "primitives/trace_access.h"
#include "rv32-adapters/vec_heap.cuh"

struct Rv32VecHeapConfig {
    size_t num_reads;
    size_t blocks_per_read;
    size_t blocks_per_write;
    size_t read_size;
    size_t write_size;

    __device__ Rv32VecHeapConfig(size_t nr, size_t bpr, size_t bpw, size_t rs, size_t ws)
        : num_reads(nr), blocks_per_read(bpr), blocks_per_write(bpw), read_size(rs),
          write_size(ws) {}
};

__device__ inline Rv32VecHeapConfig get_rv32_vec_heap_config(const FieldExprMeta *meta) {
    if (meta->num_limbs <= 32) {
        if (meta->adapter_blocks == 1) {
            return Rv32VecHeapConfig(2, 1, 1, 32, 32);
        } else if (meta->adapter_blocks == 2) {
            if (meta->num_inputs == 2) {
                return Rv32VecHeapConfig(1, 2, 2, 32, 32);
            } else {
                return Rv32VecHeapConfig(2, 2, 2, 32, 32);
            }
        }
    } else if (meta->adapter_blocks == 3) {
        return Rv32VecHeapConfig(2, 3, 3, 16, 16);
    } else if (meta->adapter_blocks == 6) {
        if (meta->num_inputs == 2) {
            return Rv32VecHeapConfig(1, 6, 6, 16, 16);
        } else {
            return Rv32VecHeapConfig(2, 6, 6, 16, 16);
        }
    }

    return Rv32VecHeapConfig(2, 1, 1, 32, 32);
}

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
__device__ inline void instantiate_rv32_vec_heap_adapter(
    RowSlice &row,
    const uint8_t *rec_bytes,
    size_t pointer_max_bits,
    VariableRangeChecker &range_checker,
    BitwiseOperationLookup &bitwise_lookup,
    uint32_t timestamp_max_bits,
    size_t &adapter_size
) {
    using AdapterRecord = Rv32VecHeapAdapterRecord<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE>;
    using AdapterExecutor =
        Rv32VecHeapAdapter<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, READ_SIZE, WRITE_SIZE>;

    const auto *adapter_rec = reinterpret_cast<const AdapterRecord *>(rec_bytes);
    AdapterExecutor adapter_step(
        pointer_max_bits, range_checker, bitwise_lookup, timestamp_max_bits
    );
    adapter_step.fill_trace_row(row, *adapter_rec);

    adapter_size = sizeof(AdapterRecord);
}

__device__ inline void route_rv32_vec_heap_adapter(
    RowSlice &row,
    const uint8_t *rec_bytes,
    const FieldExprMeta *meta,
    size_t pointer_max_bits,
    VariableRangeChecker &range_checker,
    BitwiseOperationLookup &bitwise_lookup,
    uint32_t timestamp_max_bits,
    size_t &adapter_size
) {
    Rv32VecHeapConfig config = get_rv32_vec_heap_config(meta);

    if (config.num_reads == 1 && config.blocks_per_read == 2 && config.blocks_per_write == 2 &&
        config.read_size == 32 && config.write_size == 32) {
        instantiate_rv32_vec_heap_adapter<1, 2, 2, 32, 32>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    } else if (config.num_reads == 2 && config.blocks_per_read == 1 &&
               config.blocks_per_write == 1 && config.read_size == 32 && config.write_size == 32) {
        instantiate_rv32_vec_heap_adapter<2, 1, 1, 32, 32>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    } else if (config.num_reads == 2 && config.blocks_per_read == 2 &&
               config.blocks_per_write == 2 && config.read_size == 32 && config.write_size == 32) {
        instantiate_rv32_vec_heap_adapter<2, 2, 2, 32, 32>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    } else if (config.num_reads == 2 && config.blocks_per_read == 3 &&
               config.blocks_per_write == 3 && config.read_size == 16 && config.write_size == 16) {
        instantiate_rv32_vec_heap_adapter<2, 3, 3, 16, 16>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    } else if (config.num_reads == 2 && config.blocks_per_read == 6 &&
               config.blocks_per_write == 6 && config.read_size == 16 && config.write_size == 16) {
        instantiate_rv32_vec_heap_adapter<2, 6, 6, 16, 16>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    } else if (config.num_reads == 1 && config.blocks_per_read == 6 &&
               config.blocks_per_write == 6 && config.read_size == 16 && config.write_size == 16) {
        instantiate_rv32_vec_heap_adapter<1, 6, 6, 16, 16>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    } else {
        assert(
            config.num_reads == 2 && config.blocks_per_read == 1 && config.blocks_per_write == 1 &&
            config.read_size == 32 && config.write_size == 32
        );
        instantiate_rv32_vec_heap_adapter<2, 1, 1, 32, 32>(
            row,
            rec_bytes,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );
    }
}