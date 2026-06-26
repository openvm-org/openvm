#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/load.cuh"

using namespace riscv;
using namespace program;

enum Rv64LoadSignExtendOpcode {
    LOADB = 8,
    LOADH = 9,
    LOADW = 10,
};

constexpr size_t LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH = 3;
constexpr uint32_t LOAD_SIGN_EXTEND_BYTE_CASES = 8;
constexpr size_t LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH = 2;
constexpr uint32_t LOAD_SIGN_EXTEND_HALFWORD_CASES = 4;
constexpr size_t LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH = 1;
constexpr uint32_t LOAD_SIGN_EXTEND_WORD_CASES = 2;
constexpr uint32_t LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE = 2;
constexpr uint16_t SIGN_BYTE = 1 << (RV64_BYTE_BITS - 1);
constexpr uint16_t SIGN_U16 = 1 << (U16_BITS - 1);
struct LoadRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint16_t read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadSignExtendByteCoreCols {
    T selector[LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH];
    T is_valid;
    T data_most_sig_bit;
    T read_cell_bytes[2];
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T, size_t SELECTOR_WIDTH> struct LoadSignExtendAlignedCoreCols {
    T selector[SELECTOR_WIDTH];
    T is_valid;
    T data_most_sig_bit;
    T read_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadSignExtendByteCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadSignExtendByteCoreCols<T> core;
};

template <typename T> struct Rv64LoadSignExtendHalfwordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadSignExtendAlignedCoreCols<T, LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH> core;
};

template <typename T> struct Rv64LoadSignExtendWordCols {
    Rv64LoadAdapterCols<T> adapter;
    LoadSignExtendAlignedCoreCols<T, LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH> core;
};

constexpr size_t RV64_LOAD_SIGN_EXTEND_BYTE_WIDTH =
    sizeof(Rv64LoadSignExtendByteCols<uint8_t>);
constexpr size_t RV64_LOAD_SIGN_EXTEND_HALFWORD_WIDTH =
    sizeof(Rv64LoadSignExtendHalfwordCols<uint8_t>);
constexpr size_t RV64_LOAD_SIGN_EXTEND_WORD_WIDTH =
    sizeof(Rv64LoadSignExtendWordCols<uint8_t>);

struct Rv64LoadSignExtendRecord {
    Rv64LoadAdapterRecord adapter;
    LoadRecord core;
};

static_assert(sizeof(Rv64LoadAdapterRecord) == 44);
static_assert(sizeof(LoadRecord) == 10);
static_assert(sizeof(Rv64LoadSignExtendRecord) == 56);
static_assert(offsetof(Rv64LoadSignExtendRecord, core) == 44);

static __device__ __forceinline__ uint16_t byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

struct LoadSignExtendByteCore {
    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadSignExtendByteCore(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : range_checker(range_checker), bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        assert(record.local_opcode == LOADB);
        uint8_t shift = record.shift_amount;
        uint16_t read_cell = record.read_data[shift >> 1];
        uint16_t read_cell_bytes[2] = {
            byte_from_cell(read_cell, 0),
            byte_from_cell(read_cell, 1),
        };
        uint16_t selected_byte = read_cell_bytes[shift & 1];
        uint16_t sign_bit = selected_byte & SIGN_BYTE;

        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);
        range_checker.add_count(selected_byte - sign_bit, RV64_BYTE_BITS - 1);

        Encoder encoder(
            LOAD_SIGN_EXTEND_BYTE_CASES,
            LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE,
            true,
            LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(LoadSignExtendByteCoreCols, selector)),
            shift
        );
        COL_WRITE_VALUE(row, LoadSignExtendByteCoreCols, is_valid, 1);
        COL_WRITE_VALUE(row, LoadSignExtendByteCoreCols, data_most_sig_bit, sign_bit != 0);
        COL_WRITE_ARRAY(row, LoadSignExtendByteCoreCols, read_cell_bytes, read_cell_bytes);
        COL_WRITE_ARRAY(row, LoadSignExtendByteCoreCols, read_data, record.read_data);
    }
};

template <size_t SELECTOR_WIDTH, uint32_t CASES> struct LoadSignExtendAlignedCore {
    VariableRangeChecker range_checker;

    __device__ LoadSignExtendAlignedCore(VariableRangeChecker range_checker)
        : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, LoadRecord record) {
        Rv64LoadSignExtendOpcode opcode =
            static_cast<Rv64LoadSignExtendOpcode>(record.local_opcode);
        uint8_t shift = record.shift_amount;
        uint32_t access_cells;
        uint32_t case_idx;
        if constexpr (CASES == LOAD_SIGN_EXTEND_HALFWORD_CASES) {
            assert(opcode == LOADH);
            access_cells = 1;
            case_idx = shift >> 1;
        } else {
            assert(opcode == LOADW);
            access_cells = 2;
            case_idx = shift >> 2;
        }
        uint16_t sign_cell = record.read_data[(shift >> 1) + access_cells - 1];
        uint16_t sign_bit = sign_cell & SIGN_U16;
        range_checker.add_count(sign_cell - sign_bit, U16_BITS - 1);

        Encoder encoder(CASES, LOAD_SIGN_EXTEND_SELECTOR_MAX_DEGREE, true, SELECTOR_WIDTH);
        encoder.write_flag_pt(row, case_idx);
        row[SELECTOR_WIDTH] = 1;
        row[SELECTOR_WIDTH + 1] = sign_bit != 0;
        row.write_array(SELECTOR_WIDTH + 2, BLOCK_FE_WIDTH, record.read_data);
    }
};

__global__ void rv64_load_sign_extend_byte_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> records,
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

        auto core = LoadSignExtendByteCore(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendByteCols, core)), record.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_load_sign_extend_halfword_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> records,
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

        using Core = LoadSignExtendAlignedCore<
            LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
            LOAD_SIGN_EXTEND_HALFWORD_CASES>;
        auto core = Core(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendHalfwordCols, core)), record.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_load_sign_extend_word_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> records,
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

        using Core = LoadSignExtendAlignedCore<
            LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
            LOAD_SIGN_EXTEND_WORD_CASES>;
        auto core = Core(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(Rv64LoadSignExtendWordCols, core)), record.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_sign_extend_byte_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_SIGN_EXTEND_BYTE_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_byte_tracegen_kernel<<<grid, block, 0, stream>>>(
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

extern "C" int _rv64_load_sign_extend_halfword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_SIGN_EXTEND_HALFWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_halfword_tracegen_kernel<<<grid, block, 0, stream>>>(
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

extern "C" int _rv64_load_sign_extend_word_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOAD_SIGN_EXTEND_WORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_load_sign_extend_word_tracegen_kernel<<<grid, block, 0, stream>>>(
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
