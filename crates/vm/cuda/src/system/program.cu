#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "system/program.cuh"

static constexpr uint32_t EXIT_CODE_FAIL = 1;

__global__ void program_cached_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ProgramExecutionCols<Fp>> records,
    uint32_t pc_base,
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
        COL_WRITE_VALUE(row, ProgramExecutionCols, pc, pc_base + (idx * program::DEFAULT_PC_STEP));
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
    size_t terminate_opcode,
    cudaStream_t stream
) {
    assert(width == sizeof(ProgramExecutionCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    program_cached_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, width, d_records, pc_base, terminate_opcode
    );
    return CHECK_KERNEL();
}

/// Converts raw u32 execution frequencies to field elements in place on
/// device, zero-filling [filtered_len, height).
__global__ void program_fill_frequencies(
    const uint32_t *__restrict__ freqs,
    size_t filtered_len,
    Fp *__restrict__ out,
    size_t height
) {
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < height; i += stride) {
        out[i] = i < filtered_len ? Fp(freqs[i]) : Fp(0);
    }
}

extern "C" int _program_fill_frequencies(
    const uint32_t *d_freqs,
    size_t filtered_len,
    Fp *d_out,
    size_t height,
    cudaStream_t stream
) {
    if (height == 0) {
        return 0;
    }
    const int threads = 256;
    const size_t want = (height + threads - 1) / threads;
    const int blocks = (int)(want < 2048 ? want : 2048);
    program_fill_frequencies<<<blocks, threads, 0, stream>>>(d_freqs, filtered_len, d_out, height);
    return CHECK_KERNEL();
}
