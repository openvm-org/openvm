#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "system/program.cuh"
#include <cuda_runtime.h>

__global__ void program_testing_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        auto record = reinterpret_cast<ProgramExecutionCols<Fp> *>(records)[idx];
        COL_WRITE_VALUE(row, ProgramExecutionCols, pc, record.pc);
        COL_WRITE_VALUE(row, ProgramExecutionCols, opcode, record.opcode);
        COL_WRITE_VALUE(row, ProgramExecutionCols, a, record.a);
        COL_WRITE_VALUE(row, ProgramExecutionCols, b, record.b);
        COL_WRITE_VALUE(row, ProgramExecutionCols, c, record.c);
        COL_WRITE_VALUE(row, ProgramExecutionCols, d, record.d);
        COL_WRITE_VALUE(row, ProgramExecutionCols, e, record.e);
        COL_WRITE_VALUE(row, ProgramExecutionCols, f, record.f);
        COL_WRITE_VALUE(row, ProgramExecutionCols, g, record.g);
        COL_WRITE_VALUE(row, ProgramCols, exec_freq, Fp::one());
    } else if (idx < height) {
#pragma unroll
        for (size_t i = 0; i < sizeof(ProgramCols<uint8_t>); i++) {
            row.write(i, 0);
        }
    }
}

extern "C" int _program_testing_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t num_records
) {
    assert(width == sizeof(ProgramCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    program_testing_tracegen<<<grid, block>>>(d_trace, height, d_records, num_records);
    return cudaGetLastError();
}
