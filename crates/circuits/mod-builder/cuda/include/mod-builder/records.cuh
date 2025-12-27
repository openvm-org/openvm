#pragma once

#include "rv32-adapters/vec_heap.cuh"

struct FieldExprCoreRecord {
    uint8_t opcode;
    uint8_t input_limbs[];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct FieldExprRecord {
    Rv32VecHeapAdapterRecord<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, READ_SIZE, WRITE_SIZE>
        adapter;
    FieldExprCoreRecord core;
};