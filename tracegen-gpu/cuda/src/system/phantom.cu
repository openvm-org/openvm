#include "launcher.cuh"
#include "trace_access.h"

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
    uint8_t *records,
    size_t num_records
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        PhantomRecord rec = reinterpret_cast<PhantomRecord *>(records)[idx];
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
    uint8_t *d_records,
    size_t num_records
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(PhantomCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    phantom_tracegen<<<grid, block>>>(d_trace, height, width, d_records, num_records);
    return cudaGetLastError();
}
