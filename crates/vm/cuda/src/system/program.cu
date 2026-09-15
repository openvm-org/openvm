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
    if (idx >= height) {
        return;
    }
    RowSlice row(trace + idx, height);
    COL_WRITE_VALUE(row, ProgramCachedCols, exec_end, Fp(1 + (idx == height - 1)));
    COL_WRITE_VALUE(row, ProgramCachedCols, exec_start, Fp(idx == 0));
    if (idx < records.len()) {
        auto const &rec = records[idx];
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.pc, rec.pc);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.opcode, rec.opcode);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.a, rec.a);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.b, rec.b);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.c, rec.c);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.d, rec.d);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.e, rec.e);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.f, rec.f);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.g, rec.g);
    } else {
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.pc, pc_base + (idx * pc_step));
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.opcode, terminate_opcode);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.a, Fp::zero());
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.b, Fp::zero());
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.c, EXIT_CODE_FAIL);
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.d, Fp::zero());
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.e, Fp::zero());
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.f, Fp::zero());
        COL_WRITE_VALUE(row, ProgramCachedCols, exec.g, Fp::zero());
    }
}

extern "C" int _program_cached_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ProgramExecutionCols<Fp>> d_records,
    uint32_t pc_base,
    uint32_t pc_step,
    size_t terminate_opcode,
    cudaStream_t stream
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(ProgramCachedCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    program_cached_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, width, d_records, pc_base, pc_step, terminate_opcode
    );
    return CHECK_KERNEL();
}
