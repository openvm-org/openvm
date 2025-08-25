#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"
#include "system/program.cuh"

static constexpr uint32_t EXIT_CODE_FAIL = 1;

__global__ void program_cached_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ProgramExecutionCols<Fp>> records,
    uint32_t pc_base,
    uint32_t pc_step,
    size_t terminate_opcode
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        COL_WRITE_VALUE(row, ProgramExecutionCols, pc, rec.pc);
        COL_WRITE_VALUE(row, ProgramExecutionCols, opcode, rec.opcode);
        COL_WRITE_VALUE(row, ProgramExecutionCols, a, rec.a);
        COL_WRITE_VALUE(row, ProgramExecutionCols, b, rec.b);
        COL_WRITE_VALUE(row, ProgramExecutionCols, c, rec.c);
        COL_WRITE_VALUE(row, ProgramExecutionCols, d, rec.d);
        COL_WRITE_VALUE(row, ProgramExecutionCols, e, rec.e);
        COL_WRITE_VALUE(row, ProgramExecutionCols, f, rec.f);
        COL_WRITE_VALUE(row, ProgramExecutionCols, g, rec.g);
    } else {
        COL_WRITE_VALUE(row, ProgramExecutionCols, pc, pc_base + (idx * pc_step));
        COL_WRITE_VALUE(row, ProgramExecutionCols, opcode, terminate_opcode);
        COL_WRITE_VALUE(row, ProgramExecutionCols, a, Fp::zero());
        COL_WRITE_VALUE(row, ProgramExecutionCols, b, Fp::zero());
        COL_WRITE_VALUE(row, ProgramExecutionCols, c, EXIT_CODE_FAIL);
        COL_WRITE_VALUE(row, ProgramExecutionCols, d, Fp::zero());
        COL_WRITE_VALUE(row, ProgramExecutionCols, e, Fp::zero());
        COL_WRITE_VALUE(row, ProgramExecutionCols, f, Fp::zero());
        COL_WRITE_VALUE(row, ProgramExecutionCols, g, Fp::zero());
    }
}

extern "C" int _program_cached_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ProgramExecutionCols<Fp>> d_records,
    uint32_t pc_base,
    uint32_t pc_step,
    size_t terminate_opcode
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(ProgramExecutionCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    program_cached_tracegen<<<grid, block>>>(
        d_trace, height, width, d_records, pc_base, pc_step, terminate_opcode
    );
    return cudaGetLastError();
}
