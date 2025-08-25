#include "fp.h"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "system/execution.cuh"

template <typename T> struct DummyExecutionInteractionCols {
    T count;
    ExecutionState<T> initial_state;
    ExecutionState<T> final_state;
};

__global__ void execution_testing_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<DummyExecutionInteractionCols<Fp>> records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
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
    DeviceBufferConstView<DummyExecutionInteractionCols<Fp>> d_records
) {
    assert(width == sizeof(DummyExecutionInteractionCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    execution_testing_tracegen<<<grid, block>>>(d_trace, height, d_records);
    return cudaGetLastError();
}
