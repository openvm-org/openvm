#pragma once
#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

using namespace poseidon2;

template <typename T> struct VerifyBatchCellCols {
    MemoryReadAuxCols<T> read;
    T opened_index;
    MemoryReadAuxCols<T> read_row_pointer_and_length;
    T row_pointer;
    T row_end;
    T is_first_in_row;
};

template <typename T> struct TopLevelSpecificCols {
    T pc;
    T end_timestamp;
    T dim_register;
    T opened_register;
    T opened_length_register;
    T proof_id;
    T index_register;
    T commit_register;
    T final_opened_index;
    T log_height;
    T opened_length;
    T dim_base_pointer;
    T index_base_pointer;
    MemoryReadAuxCols<T> dim_base_pointer_read;
    MemoryReadAuxCols<T> opened_base_pointer_read;
    MemoryReadAuxCols<T> opened_length_read;
    MemoryReadAuxCols<T> index_base_pointer_read;
    MemoryReadAuxCols<T> commit_pointer_read;
    T proof_index;
    MemoryReadAuxCols<T> read_initial_height_or_sibling_is_on_right;
    MemoryReadAuxCols<T> read_final_height;
    T sibling_is_on_right;
    T commit_pointer;
    MemoryReadAuxCols<T> commit_read;
};

template <typename T> struct InsideRowSpecificCols {
    VerifyBatchCellCols<T> cells[CHUNK];
};

template <typename T> struct SimplePoseidonSpecificCols {
    T pc;
    T is_compress;
    T output_register;
    T input_register_1;
    T input_register_2;
    T output_pointer;
    T input_pointer_1;
    T input_pointer_2;
    MemoryReadAuxCols<T> read_output_pointer;
    MemoryReadAuxCols<T> read_input_pointer_1;
    MemoryReadAuxCols<T> read_input_pointer_2;
    MemoryReadAuxCols<T> read_data_1;
    MemoryReadAuxCols<T> read_data_2;
    MemoryWriteAuxCols<T, CHUNK> write_data_1;
    MemoryWriteAuxCols<T, CHUNK> write_data_2;
};

template <typename T> constexpr T constexpr_max(T a, T b) { return a > b ? a : b; }

constexpr size_t COL_SPECIFIC_WIDTH = constexpr_max(
    sizeof(TopLevelSpecificCols<uint8_t>),
    constexpr_max(
        sizeof(InsideRowSpecificCols<uint8_t>),
        sizeof(SimplePoseidonSpecificCols<uint8_t>)
    )
);
