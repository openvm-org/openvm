#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/store.cuh"

using namespace riscv;
using namespace program;

enum Rv64StoreOpcode {
    STORED = 4,
    STOREW = 5,
    STOREH = 6,
    STOREB = 7,
};

constexpr size_t STORE_BYTE_SELECTOR_WIDTH = 3;
constexpr uint32_t STORE_BYTE_CASES = 8;
constexpr size_t STORE_HALFWORD_SELECTOR_WIDTH = 2;
constexpr uint32_t STORE_HALFWORD_CASES = 4;
constexpr size_t STORE_WORD_SELECTOR_WIDTH = 1;
constexpr uint32_t STORE_WORD_CASES = 2;
constexpr size_t STORE_DOUBLEWORD_SELECTOR_WIDTH = 1;
constexpr uint32_t STORE_DOUBLEWORD_CASES = 1;
constexpr uint32_t STORE_SELECTOR_MAX_DEGREE = 2;

struct StoreRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct StoreByteCoreCols {
    T selector[STORE_BYTE_SELECTOR_WIDTH];
    T is_valid;
    T read_cell_bytes[2];
    T prev_cell_bytes[2];
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct StoreHalfwordCoreCols {
    T selector[STORE_HALFWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct StoreWordCoreCols {
    T selector[STORE_WORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct StoreDoublewordCoreCols {
    T selector[STORE_DOUBLEWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64StoreByteCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreByteCoreCols<T> core;
};

template <typename T> struct Rv64StoreHalfwordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreHalfwordCoreCols<T> core;
};

template <typename T> struct Rv64StoreWordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreWordCoreCols<T> core;
};

template <typename T> struct Rv64StoreDoublewordCols {
    Rv64StoreAdapterCols<T> adapter;
    StoreDoublewordCoreCols<T> core;
};

constexpr size_t RV64_STORE_BYTE_WIDTH = sizeof(Rv64StoreByteCols<uint8_t>);
constexpr size_t RV64_STORE_HALFWORD_WIDTH = sizeof(Rv64StoreHalfwordCols<uint8_t>);
constexpr size_t RV64_STORE_WORD_WIDTH = sizeof(Rv64StoreWordCols<uint8_t>);
constexpr size_t RV64_STORE_DOUBLEWORD_WIDTH = sizeof(Rv64StoreDoublewordCols<uint8_t>);

struct Rv64StoreRecord {
    Rv64StoreAdapterRecord adapter;
    StoreRecord core;
};

static_assert(sizeof(Rv64StoreAdapterRecord) == 36);
static_assert(sizeof(StoreRecord) == 18);
static_assert(sizeof(Rv64StoreRecord) == 56);
static_assert(offsetof(Rv64StoreRecord, core) == 36);

static __device__ __forceinline__ uint16_t byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

struct StoreByteCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ StoreByteCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, StoreRecord record) {
        assert(record.local_opcode == STOREB);
        uint8_t shift = record.shift_amount;
        uint8_t cell_shift = shift >> 1;

        uint16_t read_cell_bytes[2] = {
            byte_from_cell(record.read_data[0], 0),
            byte_from_cell(record.read_data[0], 1),
        };
        uint16_t prev_cell_bytes[2] = {
            byte_from_cell(record.prev_data[cell_shift], 0),
            byte_from_cell(record.prev_data[cell_shift], 1),
        };
        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);
        bitwise_lookup.add_range(prev_cell_bytes[0], prev_cell_bytes[1]);

        Encoder encoder(
            STORE_BYTE_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_BYTE_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row.slice_from(COL_INDEX(StoreByteCoreCols, selector)), shift);
        COL_WRITE_VALUE(row, StoreByteCoreCols, is_valid, 1);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, read_cell_bytes, read_cell_bytes);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, prev_cell_bytes, prev_cell_bytes);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, StoreByteCoreCols, prev_data, record.prev_data);
    }
};

struct StoreHalfwordCore {
    __device__ void fill_trace_row(RowSlice row, StoreRecord record) {
        assert(record.local_opcode == STOREH);
        uint8_t shift = record.shift_amount;
        uint32_t case_idx = shift >> 1;

        Encoder encoder(
            STORE_HALFWORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_HALFWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, case_idx);
        row[STORE_HALFWORD_SELECTOR_WIDTH] = 1;
        row.write_array(STORE_HALFWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            STORE_HALFWORD_SELECTOR_WIDTH + 1 + BLOCK_FE_WIDTH,
            BLOCK_FE_WIDTH,
            record.prev_data
        );
    }
};

struct StoreWordCore {
    __device__ void fill_trace_row(RowSlice row, StoreRecord record) {
        assert(record.local_opcode == STOREW);
        uint8_t shift = record.shift_amount;
        uint32_t case_idx = shift >> 2;

        Encoder encoder(
            STORE_WORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_WORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, case_idx);
        row[STORE_WORD_SELECTOR_WIDTH] = 1;
        row.write_array(STORE_WORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            STORE_WORD_SELECTOR_WIDTH + 1 + BLOCK_FE_WIDTH, BLOCK_FE_WIDTH, record.prev_data
        );
    }
};

struct StoreDoublewordCore {
    __device__ void fill_trace_row(RowSlice row, StoreRecord record) {
        assert(record.local_opcode == STORED);
        assert(record.shift_amount == 0);

        Encoder encoder(
            STORE_DOUBLEWORD_CASES, STORE_SELECTOR_MAX_DEGREE, true, STORE_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, 0);
        row[STORE_DOUBLEWORD_SELECTOR_WIDTH] = 1;
        row.write_array(STORE_DOUBLEWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            STORE_DOUBLEWORD_SELECTOR_WIDTH + 1 + BLOCK_FE_WIDTH,
            BLOCK_FE_WIDTH,
            record.prev_data
        );
    }
};

__global__ void rv64_store_byte_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = StoreByteCore(BitwiseOperationLookup(bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64StoreByteCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreByteCols, adapter.mem_as, 2);
    }
}

__global__ void rv64_store_halfword_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        StoreHalfwordCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64StoreHalfwordCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreHalfwordCols, adapter.mem_as, 2);
    }
}

__global__ void rv64_store_word_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        StoreWordCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64StoreWordCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreWordCols, adapter.mem_as, 2);
    }
}

__global__ void rv64_store_doubleword_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64StoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        StoreDoublewordCore core;
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64StoreDoublewordCols, core)), record.core
        );
    } else {
        row.fill_zero(0, width);
        COL_WRITE_VALUE(row, Rv64StoreDoublewordCols, adapter.mem_as, 2);
    }
}

extern "C" int _rv64_store_byte_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_STORE_BYTE_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_byte_tracegen_kernel<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _rv64_store_halfword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_STORE_HALFWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_halfword_tracegen_kernel<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _rv64_store_word_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_STORE_WORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_word_tracegen_kernel<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _rv64_store_doubleword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64StoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_STORE_DOUBLEWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_store_doubleword_tracegen_kernel<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
