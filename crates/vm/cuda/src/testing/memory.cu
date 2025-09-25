#include "launcher.cuh"
#include "primitives/trace_access.h"
#include "system/memory/address.cuh"

template <typename T, size_t BLOCK_SIZE> struct DummyMemoryInteractionCols {
    MemoryAddress<T> address;
    T data[BLOCK_SIZE];
    T timestamp;
    T count;
};

template <size_t BLOCK_SIZE>
__global__ void memory_testing_tracegen(Fp *trace, size_t height, Fp *records, size_t num_records) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        auto record = reinterpret_cast<DummyMemoryInteractionCols<Fp, BLOCK_SIZE> *>(records)[idx];
        COL_WRITE_VALUE(row, MemoryAddress, address_space, record.address.address_space);
        COL_WRITE_VALUE(row, MemoryAddress, pointer, record.address.pointer);
        row.write_array(2, BLOCK_SIZE, record.data);
        row.write(2 + BLOCK_SIZE, record.timestamp);
        row.write(2 + BLOCK_SIZE + 1, record.count);
    } else if (idx < height) {
#pragma unroll
        for (size_t i = 0; i < sizeof(DummyMemoryInteractionCols<uint8_t, BLOCK_SIZE>); i++) {
            row.write(i, 0);
        }
    }
}

extern "C" int _memory_testing_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    Fp *d_records,
    size_t num_records,
    size_t block_size
) {
    auto [grid, block] = kernel_launch_params(height);
    switch (block_size) {
    case 1:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 1>));
        memory_testing_tracegen<1><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    case 2:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 2>));
        memory_testing_tracegen<2><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    case 4:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 4>));
        memory_testing_tracegen<4><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    case 8:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 8>));
        memory_testing_tracegen<8><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    case 16:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 16>));
        memory_testing_tracegen<16><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    case 32:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 32>));
        memory_testing_tracegen<32><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    case 64:
        assert(width == sizeof(DummyMemoryInteractionCols<uint8_t, 64>));
        memory_testing_tracegen<64><<<grid, block>>>(d_trace, height, d_records, num_records);
        break;
    default:
        assert(false && "Invalid block size");
    }
    return CHECK_KERNEL();
}
