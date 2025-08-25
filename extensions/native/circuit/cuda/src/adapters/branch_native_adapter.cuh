#pragma once

#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/address.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace native;

template <typename T> struct BranchNativeAdapterReadCols {
    MemoryAddress<T> address;
    MemoryReadOrImmediateAuxCols<T> read_aux;
};

template <typename T> struct BranchNativeAdapterCols {
    ExecutionState<T> from_state;
    BranchNativeAdapterReadCols<T> reads_aux[2];
};

template <typename F> struct BranchNativeAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    F ptrs[2];
    // Will set prev_timestamp to `u32::MAX` if the read is an immediate
    MemoryReadAuxRecord reads_aux[2];
};

template <typename F> struct BranchNativeAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ BranchNativeAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, BranchNativeAdapterRecord<F> const &record) {
        COL_WRITE_VALUE(row, BranchNativeAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, BranchNativeAdapterCols, from_state.timestamp, record.from_timestamp);

        Fp native_as_inv = inv(Fp(AS_NATIVE));
#pragma unroll
        for (uint32_t i = 0; i < 2; i++) {
            auto read_slice = row.slice_from(COL_INDEX(BranchNativeAdapterCols, reads_aux[i]));
            COL_WRITE_VALUE(
                read_slice, BranchNativeAdapterReadCols, address.pointer, record.ptrs[i]
            );
            if (record.reads_aux[i].prev_timestamp == UINT32_MAX) {
                COL_WRITE_VALUE(
                    read_slice, BranchNativeAdapterReadCols, address.address_space, AS_IMMEDIATE
                );
                mem_helper.fill(
                    read_slice.slice_from(COL_INDEX(BranchNativeAdapterReadCols, read_aux.base)),
                    0,
                    record.from_timestamp + i
                );
                COL_WRITE_VALUE(read_slice, BranchNativeAdapterReadCols, read_aux.is_immediate, 1);
                COL_WRITE_VALUE(read_slice, BranchNativeAdapterReadCols, read_aux.is_zero_aux, 0);
            } else {
                COL_WRITE_VALUE(
                    read_slice, BranchNativeAdapterReadCols, address.address_space, AS_NATIVE
                );
                mem_helper.fill(
                    read_slice.slice_from(COL_INDEX(BranchNativeAdapterReadCols, read_aux.base)),
                    record.reads_aux[i].prev_timestamp,
                    record.from_timestamp + i
                );
                COL_WRITE_VALUE(read_slice, BranchNativeAdapterReadCols, read_aux.is_immediate, 0);
                COL_WRITE_VALUE(
                    read_slice, BranchNativeAdapterReadCols, read_aux.is_zero_aux, native_as_inv
                );
            }
        }
    }
};