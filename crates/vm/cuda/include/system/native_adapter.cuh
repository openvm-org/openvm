#pragma once

#include "memory/address.cuh"
#include "memory/controller.cuh"
#include "primitives/execution.h"
#include "primitives/trace_access.h"

template <typename T> struct NativeAdapterReadCols {
    MemoryAddress<T> address;
    MemoryReadOrImmediateAuxCols<T> read_aux;
};

template <typename T> struct NativeAdapterWriteCols {
    MemoryAddress<T> address;
    MemoryWriteAuxCols<T, 1> write_aux;
};

template <typename T, size_t R, size_t W> struct NativeAdapterCols {
    ExecutionState<T> from_state;
    NativeAdapterReadCols<T> reads_aux[R];
    NativeAdapterWriteCols<T> writes_aux[W];
};

template <typename F, size_t R, size_t W> struct NativeAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    F read_ptr_or_imm[R];
    MemoryReadAuxRecord reads_aux[R];
    F write_ptr[W];
    MemoryWriteAuxRecord<F, 1> writes_aux[W];
};

template <typename F, size_t R, size_t W> struct NativeAdapter {
    MemoryAuxColsFactory mem_helper;

    template <typename T> using Cols = NativeAdapterCols<T, R, W>;

    __device__ NativeAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, NativeAdapterRecord<F, R, W> record) {
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);

        for (uint32_t i = 0; i < R; i++) {
            F ptr_or_imm = record.read_ptr_or_imm[i];
            if (record.reads_aux[i].prev_timestamp == UINT32_MAX) {
                COL_WRITE_VALUE(row, Cols, reads_aux[i].address.address_space, RV32_IMM_AS);
                COL_WRITE_VALUE(row, Cols, reads_aux[i].address.pointer, ptr_or_imm);
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, reads_aux[i].read_aux.base)),
                    0,
                    record.from_timestamp + i
                );
                COL_WRITE_VALUE(row, Cols, reads_aux[i].read_aux.is_immediate, Fp::one());
                COL_WRITE_VALUE(row, Cols, reads_aux[i].read_aux.is_zero_aux, Fp::zero());
            } else {
                COL_WRITE_VALUE(row, Cols, reads_aux[i].address.address_space, native::AS_NATIVE);
                COL_WRITE_VALUE(row, Cols, reads_aux[i].address.pointer, ptr_or_imm);
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, reads_aux[i].read_aux.base)),
                    record.reads_aux[i].prev_timestamp,
                    record.from_timestamp + i
                );
                COL_WRITE_VALUE(row, Cols, reads_aux[i].read_aux.is_immediate, Fp::zero());
                COL_WRITE_VALUE(
                    row, Cols, reads_aux[i].read_aux.is_zero_aux, inv(Fp(native::AS_NATIVE))
                );
            }
        }

        if (W > 0) {
            COL_WRITE_VALUE(row, Cols, writes_aux[0].address.address_space, native::AS_NATIVE);
            COL_WRITE_VALUE(row, Cols, writes_aux[0].address.pointer, record.write_ptr[0]);
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, writes_aux[0].write_aux.base)),
                record.writes_aux[0].prev_timestamp,
                record.from_timestamp + R
            );
            COL_WRITE_ARRAY(
                row, Cols, writes_aux[0].write_aux.prev_data, record.writes_aux[0].prev_data
            );
        }
    }
};