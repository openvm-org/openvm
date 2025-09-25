#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace program;
using namespace native;

template <typename T> struct JalRangeCheckCols {
    T is_jal;
    T is_range_check;
    T a_pointer;
    ExecutionState<T> state;
    MemoryWriteAuxCols<T, 1> writes_aux;
    T b;
    T c;
    T y;
};

template <typename F> struct JalRangeCheckRecord {
    bool is_jal;
    F a;
    uint32_t from_pc;
    uint32_t from_timestamp;
    MemoryWriteAuxRecord<F, 1> write;
    F b;
    F c;
};

struct JalRangeCheck {
    MemoryAuxColsFactory mem_helper;
    VariableRangeChecker range_checker;

    __device__ JalRangeCheck(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, const JalRangeCheckRecord<Fp> &record) {
        COL_WRITE_VALUE(row, JalRangeCheckCols, is_jal, record.is_jal);
        COL_WRITE_VALUE(row, JalRangeCheckCols, is_range_check, !record.is_jal);
        COL_WRITE_VALUE(row, JalRangeCheckCols, a_pointer, record.a);
        COL_WRITE_VALUE(row, JalRangeCheckCols, state.pc, record.from_pc);
        COL_WRITE_VALUE(row, JalRangeCheckCols, state.timestamp, record.from_timestamp);

        COL_WRITE_ARRAY(row, JalRangeCheckCols, writes_aux.prev_data, record.write.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(JalRangeCheckCols, writes_aux.base)),
            record.write.prev_timestamp,
            record.from_timestamp
        );

        if (record.is_jal) {
            COL_WRITE_VALUE(row, JalRangeCheckCols, b, record.b);
            COL_WRITE_VALUE(row, JalRangeCheckCols, c, Fp::zero());
            COL_WRITE_VALUE(row, JalRangeCheckCols, y, Fp::zero());
        } else {
            // Decompose the value: a_val = x + y * 2^16
            uint32_t a_val = record.write.prev_data[0].asUInt32();
            uint32_t x = a_val & 0xffff;
            uint32_t y = a_val >> 16;

            COL_WRITE_VALUE(row, JalRangeCheckCols, b, record.b);
            COL_WRITE_VALUE(row, JalRangeCheckCols, c, record.c);
            COL_WRITE_VALUE(row, JalRangeCheckCols, y, y);

            uint32_t b_bits = record.b.asUInt32();
            uint32_t c_bits = record.c.asUInt32();

#ifdef CUDA_DEBUG
            assert(b_bits <= 16);
            assert(c_bits <= 14);
            assert(x < (1 << b_bits));
            assert(y < (1 << c_bits));
#endif

            range_checker.add_count(x, b_bits);
            range_checker.add_count(y, c_bits);
        }
    }
};

__global__ void jal_rangecheck_tracegen(
    Fp *__restrict__ trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<JalRangeCheckRecord<Fp>> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_max_bins,
    uint32_t timestamp_max_bits
) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &record = records[idx];

        VariableRangeChecker range_checker(range_checker_ptr, range_checker_max_bins);

        JalRangeCheck chip(range_checker, timestamp_max_bits);
        chip.fill_trace_row(row, record);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _native_jal_rangecheck_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<JalRangeCheckRecord<Fp>> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_max_bins,
    uint32_t timestamp_max_bits
) {

    assert((height & (height - 1)) == 0);
    assert(width == sizeof(JalRangeCheckCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);
    jal_rangecheck_tracegen<<<grid, block>>>(
        (Fp *)d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_max_bins,
        timestamp_max_bits
    );

    return CHECK_KERNEL();
}