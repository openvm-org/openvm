#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/load.cuh"

using namespace riscv;
using namespace program;

enum Rv64UnsignedLoadOpcode {
    LOADD = 0,
    LOADBU = 1,
    LOADHU = 2,
    LOADWU = 3,
};

constexpr size_t LOAD_BYTE_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_BYTE_CASES = 8;
constexpr size_t LOAD_HALFWORD_SELECTOR_WIDTH = 2;
constexpr uint32_t LOAD_HALFWORD_CASES = 4;
constexpr size_t LOAD_WORD_SELECTOR_WIDTH = 1;
constexpr uint32_t LOAD_WORD_CASES = 2;
constexpr size_t LOAD_DOUBLEWORD_SELECTOR_WIDTH = 1;
constexpr uint32_t LOAD_DOUBLEWORD_CASES = 1;
constexpr uint32_t LOAD_SELECTOR_MAX_DEGREE = 2;

struct LoadRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint16_t read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadByteCoreCols {
    T selector[LOAD_BYTE_SELECTOR_WIDTH];
    T is_valid;
    T read_cell_bytes[2];
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadHalfwordCoreCols {
    T selector[LOAD_HALFWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadWordCoreCols {
    T selector[LOAD_WORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadDoublewordCoreCols {
    T selector[LOAD_DOUBLEWORD_SELECTOR_WIDTH];
    T is_valid;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadByteCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadByteCoreCols<T> core;
};

template <typename T> struct Rv64LoadHalfwordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadHalfwordCoreCols<T> core;
};

template <typename T> struct Rv64LoadWordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadWordCoreCols<T> core;
};

template <typename T> struct Rv64LoadDoublewordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadDoublewordCoreCols<T> core;
};

constexpr size_t RV64_LOAD_BYTE_WIDTH = sizeof(Rv64LoadByteCols<uint8_t>);
constexpr size_t RV64_LOAD_HALFWORD_WIDTH = sizeof(Rv64LoadHalfwordCols<uint8_t>);
constexpr size_t RV64_LOAD_WORD_WIDTH = sizeof(Rv64LoadWordCols<uint8_t>);
constexpr size_t RV64_LOAD_DOUBLEWORD_WIDTH = sizeof(Rv64LoadDoublewordCols<uint8_t>);

struct Rv64LoadRecord {
    Rv64LoadAdapterRecord adapter;
    LoadRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 44);
static_assert(sizeof(LoadRecord) == 10);
static_assert(sizeof(Rv64LoadRecord) == 56);
static_assert(offsetof(Rv64LoadRecord, core) == 44);

static __device__ __forceinline__ uint16_t byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

struct LoadByteCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadByteCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        assert(record.local_opcode == LOADBU);
        uint8_t shift = record.shift_amount;
        uint16_t read_cell = record.read_data[shift >> 1];
        uint16_t read_cell_bytes[2] = {
            byte_from_cell(read_cell, 0),
            byte_from_cell(read_cell, 1),
        };
        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);

        Encoder encoder(LOAD_BYTE_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_BYTE_SELECTOR_WIDTH);
        encoder.write_flag_pt(row.slice_from(COL_INDEX(LoadByteCoreCols, selector)), shift);
        COL_WRITE_VALUE(row, LoadByteCoreCols, is_valid, 1);
        COL_WRITE_ARRAY(row, LoadByteCoreCols, read_cell_bytes, read_cell_bytes);
        COL_WRITE_ARRAY(row, LoadByteCoreCols, read_data, record.read_data);
    }
};

struct LoadHalfwordCore {
    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        assert(record.local_opcode == LOADHU);
        uint8_t shift = record.shift_amount;
        uint32_t case_idx = shift >> 1;

        Encoder encoder(
            LOAD_HALFWORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_HALFWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, case_idx);
        row[LOAD_HALFWORD_SELECTOR_WIDTH] = 1;
        row.write_array(LOAD_HALFWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
    }
};

struct LoadWordCore {
    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        assert(record.local_opcode == LOADWU);
        uint8_t shift = record.shift_amount;
        uint32_t case_idx = shift >> 2;

        Encoder encoder(LOAD_WORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_WORD_SELECTOR_WIDTH);
        encoder.write_flag_pt(row, case_idx);
        row[LOAD_WORD_SELECTOR_WIDTH] = 1;
        row.write_array(LOAD_WORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
    }
};

struct LoadDoublewordCore {
    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        assert(record.local_opcode == LOADD);
        assert(record.shift_amount == 0);

        Encoder encoder(
            LOAD_DOUBLEWORD_CASES, LOAD_SELECTOR_MAX_DEGREE, true, LOAD_DOUBLEWORD_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(row, 0);
        row[LOAD_DOUBLEWORD_SELECTOR_WIDTH] = 1;
        row.write_array(LOAD_DOUBLEWORD_SELECTOR_WIDTH + 1, BLOCK_FE_WIDTH, record.read_data);
    }
};

__global__ void rv64_load_byte_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> records,
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
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        auto core = LoadByteCore(BitwiseOperationLookup(bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadByteCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_load_halfword_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        LoadHalfwordCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadHalfwordCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_load_word_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        LoadWordCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadWordCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_load_doubleword_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];
        auto adapter = Rv64LoadAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);
        LoadDoublewordCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadDoublewordCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_byte_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_BYTE_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_byte_tracegen_kernel<<<grid, block, 0, stream>>>(
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

extern "C" int _rv64_load_halfword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_HALFWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_halfword_tracegen_kernel<<<grid, block, 0, stream>>>(
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

extern "C" int _rv64_load_word_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_WORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_word_tracegen_kernel<<<grid, block, 0, stream>>>(
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

extern "C" int _rv64_load_doubleword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_DOUBLEWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_doubleword_tracegen_kernel<<<grid, block, 0, stream>>>(
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
