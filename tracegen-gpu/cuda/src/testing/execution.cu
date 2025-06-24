#include "../system/execution.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "trace_access.h"

template <typename T> struct DummyExecutionInteractionCols {
    T count;
    ExecutionState<T> initial_state;
    ExecutionState<T> final_state;
};

__global__ void execution_testing_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        auto record = reinterpret_cast<DummyExecutionInteractionCols<Fp> *>(records)[idx];
        COL_WRITE_VALUE(row, DummyExecutionInteractionCols, count, record.count);
        COL_WRITE_VALUE(
            row, DummyExecutionInteractionCols, initial_state.pc, record.initial_state.pc
        );
        COL_WRITE_VALUE(
            row,
            DummyExecutionInteractionCols,
            initial_state.timestamp,
            record.initial_state.timestamp
        );
        COL_WRITE_VALUE(row, DummyExecutionInteractionCols, final_state.pc, record.final_state.pc);
        COL_WRITE_VALUE(
            row, DummyExecutionInteractionCols, final_state.timestamp, record.final_state.timestamp
        );
    } else if (idx < height) {
#pragma unroll
        for (size_t i = 0; i < sizeof(DummyExecutionInteractionCols<uint8_t>); i++) {
            row.write(i, 0);
        }
    }
}

extern "C" int _execution_testing_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t num_records
) {
    assert(width == sizeof(DummyExecutionInteractionCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    execution_testing_tracegen<<<grid, block>>>(d_trace, height, d_records, num_records);
    return cudaGetLastError();
}
