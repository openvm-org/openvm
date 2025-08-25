#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"

template <typename T> struct DummyChipCols {
    T count;
    T x;
    T y;
    T z;
    T op;
};

struct DummyRecord {
    uint32_t x;
    uint32_t y;
    uint32_t op;
};

__global__ void bitwise_dummy_tracegen(
    Fp *trace,
    DeviceBufferConstView<DummyRecord> records,
    uint32_t *bitwise_count,
    uint32_t bitwise_num_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto const height = records.len();

    if (idx < height) {
        auto const &record = records[idx];
        BitwiseOperationLookup bitwise(bitwise_count, bitwise_num_bits);
        RowSlice row(trace + idx, height);

        COL_WRITE_VALUE(row, DummyChipCols, count, 1);
        COL_WRITE_VALUE(row, DummyChipCols, op, record.op);
        COL_WRITE_VALUE(row, DummyChipCols, x, record.x);
        COL_WRITE_VALUE(row, DummyChipCols, y, record.y);

        if (record.op == 0) {
            bitwise.add_range(record.x, record.y);
            COL_WRITE_VALUE(row, DummyChipCols, z, 0);
        } else {
            bitwise.add_xor(record.x, record.y);
            COL_WRITE_VALUE(row, DummyChipCols, z, record.x ^ record.y);
        }
    }
}

extern "C" int _bitwise_dummy_tracegen(
    Fp *d_trace,
    DeviceBufferConstView<DummyRecord> records,
    uint32_t *bitwise_count,
    uint32_t bitwise_num_bits
) {
    auto [grid, block] = kernel_launch_params(records.len());
    bitwise_dummy_tracegen<<<grid, block>>>(d_trace, records, bitwise_count, bitwise_num_bits);
    return cudaGetLastError();
}
