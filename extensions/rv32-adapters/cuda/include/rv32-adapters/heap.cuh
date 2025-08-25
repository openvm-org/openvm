#pragma once

#include "vec_heap.cuh"

using namespace riscv;

// Simple heap adapter - just type aliases for vec_heap with BLOCKS_PER_READ=1, BLOCKS_PER_WRITE=1
template <typename T, size_t NUM_READS, size_t READ_SIZE, size_t WRITE_SIZE>
using Rv32HeapAdapterCols = Rv32VecHeapAdapterCols<T, NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>;

template <size_t NUM_READS, size_t READ_SIZE, size_t WRITE_SIZE>
using Rv32HeapAdapterRecord = Rv32VecHeapAdapterRecord<NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>;

template <size_t NUM_READS, size_t READ_SIZE, size_t WRITE_SIZE>
using Rv32HeapAdapterExecutor = Rv32VecHeapAdapter<NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>;