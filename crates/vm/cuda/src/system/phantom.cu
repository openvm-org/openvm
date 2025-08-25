#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"

static constexpr uint32_t NUM_PHANTOM_OPERANDS = 3;

struct PhantomRecord {
    uint32_t pc;
    uint32_t operands[NUM_PHANTOM_OPERANDS];
    uint32_t timestamp;
};

template <typename T> struct PhantomCols {
    T pc;
    T operands[NUM_PHANTOM_OPERANDS];
    T timestamp;
    T is_valid;
};

__global__ void phantom_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PhantomRecord> records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        COL_WRITE_VALUE(row, PhantomCols, pc, rec.pc);
        COL_WRITE_ARRAY(row, PhantomCols, operands, rec.operands);
        COL_WRITE_VALUE(row, PhantomCols, timestamp, rec.timestamp);
        COL_WRITE_VALUE(row, PhantomCols, is_valid, Fp::one());
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _phantom_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PhantomRecord> d_records
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(PhantomCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    phantom_tracegen<<<grid, block>>>(d_trace, height, width, d_records);
    return cudaGetLastError();
}
