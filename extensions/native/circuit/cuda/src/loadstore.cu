#include "launcher.cuh"
#include "native/adapters/loadstore_native_adapter.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/trace_access.h"

using namespace riscv;
using namespace program;
using namespace native;

// Native LoadStore opcodes
#define LOADW 0
#define STOREW 1
#define HINT_STOREW 2

template <typename T, uint32_t NUM_CELLS> struct NativeLoadStoreCoreCols {
    T is_loadw;
    T is_storew;
    T is_hint_storew;

    T pointer_read;
    T data[NUM_CELLS];
};

template <typename F, uint32_t NUM_CELLS> struct NativeLoadStoreCoreRecord {
    F pointer_read;
    F data[NUM_CELLS];
    uint8_t local_opcode;
};

template <uint32_t NUM_CELLS> struct NativeLoadStoreCore {
    template <typename T> using Cols = NativeLoadStoreCoreCols<T, NUM_CELLS>;

    __device__ void fill_trace_row(RowSlice row, NativeLoadStoreCoreRecord<Fp, NUM_CELLS> record) {
        COL_WRITE_VALUE(row, Cols, is_loadw, record.local_opcode == LOADW);
        COL_WRITE_VALUE(row, Cols, is_storew, record.local_opcode == STOREW);
        COL_WRITE_VALUE(row, Cols, is_hint_storew, record.local_opcode == HINT_STOREW);

        COL_WRITE_VALUE(row, Cols, pointer_read, record.pointer_read);

        COL_WRITE_ARRAY(row, Cols, data, record.data);
    }
};

template <typename T, uint32_t NUM_CELLS> struct NativeLoadStoreCols {
    NativeLoadStoreAdapterCols<T, NUM_CELLS> adapter;
    NativeLoadStoreCoreCols<T, NUM_CELLS> core;
};

template <typename F, uint32_t NUM_CELLS> struct NativeLoadStoreRecord {
    NativeLoadStoreAdapterRecord<F, NUM_CELLS> adapter;
    NativeLoadStoreCoreRecord<F, NUM_CELLS> core;
};

template <uint32_t NUM_CELLS> struct NativeLoadStoreWrapper {
    template <typename T> using Cols = NativeLoadStoreCols<T, NUM_CELLS>;
};

template <uint32_t NUM_CELLS>
__global__ void native_loadstore_tracegen(
    Fp *trace,
    uint32_t height,
    uint32_t width,
    DeviceBufferConstView<NativeLoadStoreRecord<Fp, NUM_CELLS>> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = NativeLoadStoreAdapter<Fp, NUM_CELLS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = NativeLoadStoreCore<NUM_CELLS>();
        core.fill_trace_row(
            row.slice_from(
                COL_INDEX(typename NativeLoadStoreWrapper<NUM_CELLS>::template Cols, core)
            ),
            record.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _native_loadstore_tracegen(
    Fp *d_trace,
    uint32_t height,
    uint32_t width,
    DeviceRawBufferConstView d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t num_cells,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height);

    switch (num_cells) {
    case 1:
        assert(width == sizeof(NativeLoadStoreCols<uint8_t, 1>));
        native_loadstore_tracegen<<<grid, block>>>(
            d_trace,
            height,
            width,
            d_records.as_typed<NativeLoadStoreRecord<Fp, 1>>(),
            d_range_checker,
            range_checker_num_bins,
            timestamp_max_bits
        );
        break;
    case 4:
        assert(width == sizeof(NativeLoadStoreCols<uint8_t, 4>));
        native_loadstore_tracegen<<<grid, block>>>(
            d_trace,
            height,
            width,
            d_records.as_typed<NativeLoadStoreRecord<Fp, 4>>(),
            d_range_checker,
            range_checker_num_bins,
            timestamp_max_bits
        );
        break;
    default:
        return cudaErrorInvalidValue;
    }

    return cudaGetLastError();
}