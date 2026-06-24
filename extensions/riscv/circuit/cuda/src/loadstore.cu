#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/loadstore.cuh"
#include <type_traits>

using namespace riscv;
using namespace program;

enum Rv64LoadStoreOpcode {
    LOADD,
    LOADBU,
    LOADHU,
    LOADWU,
    STORED,
    STOREW,
    STOREH,
    STOREB,
    LOADB,
    LOADH,
    LOADW,
};

constexpr size_t LOADSTORE_BYTE_SELECTOR_WIDTH = 5;
constexpr uint32_t LOADSTORE_BYTE_CASES = 16;
constexpr size_t LOADSTORE_HALFWORD_SELECTOR_WIDTH = 3;
constexpr uint32_t LOADSTORE_HALFWORD_CASES = 8;
constexpr size_t LOADSTORE_WORD_SELECTOR_WIDTH = 2;
constexpr uint32_t LOADSTORE_WORD_CASES = 4;
constexpr uint32_t LOADSTORE_SELECTOR_MAX_DEGREE = 2;
struct LoadStoreRecord {
    uint8_t local_opcode;
    uint8_t shift_amount;
    uint16_t read_data[BLOCK_FE_WIDTH];
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadStoreByteCoreCols {
    T selector[LOADSTORE_BYTE_SELECTOR_WIDTH];
    T is_valid;
    T is_load;
    T read_cell_bytes[2];
    T prev_cell_bytes[2];
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T, size_t SELECTOR_WIDTH> struct LoadStoreAlignedCoreCols {
    T selector[SELECTOR_WIDTH];
    T is_valid;
    T is_load;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct LoadStoreDoublewordCoreCols {
    T is_valid;
    T is_load;
    T read_data[BLOCK_FE_WIDTH];
    T prev_data[BLOCK_FE_WIDTH];
};

template <typename T> struct Rv64LoadStoreByteCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadStoreByteCoreCols<T> core;
};

template <typename T> struct Rv64LoadStoreHalfwordCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadStoreAlignedCoreCols<T, LOADSTORE_HALFWORD_SELECTOR_WIDTH> core;
};

template <typename T> struct Rv64LoadStoreWordCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadStoreAlignedCoreCols<T, LOADSTORE_WORD_SELECTOR_WIDTH> core;
};

template <typename T> struct Rv64LoadStoreDoublewordCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadStoreDoublewordCoreCols<T> core;
};

constexpr size_t RV64_LOADSTORE_BYTE_WIDTH = sizeof(Rv64LoadStoreByteCols<uint8_t>);
constexpr size_t RV64_LOADSTORE_HALFWORD_WIDTH =
    sizeof(Rv64LoadStoreHalfwordCols<uint8_t>);
constexpr size_t RV64_LOADSTORE_WORD_WIDTH = sizeof(Rv64LoadStoreWordCols<uint8_t>);
constexpr size_t RV64_LOADSTORE_DOUBLEWORD_WIDTH =
    sizeof(Rv64LoadStoreDoublewordCols<uint8_t>);

struct Rv64LoadStoreRecord {
    Rv64LoadStoreAdapterRecord adapter;
    LoadStoreRecord core;
};

static_assert(sizeof(Rv64LoadStoreAdapterRecord) == 36);
static_assert(sizeof(LoadStoreRecord) == 18);
static_assert(sizeof(Rv64LoadStoreRecord) == 56);
static_assert(offsetof(Rv64LoadStoreRecord, core) == 36);

__device__ __forceinline__ uint16_t byte_from_cell(uint16_t cell, uint8_t byte_idx) {
    return (cell >> (RV64_BYTE_BITS * byte_idx)) & 0xff;
}

__device__ __forceinline__ uint32_t loadstore_byte_case_idx(
    Rv64LoadStoreOpcode opcode,
    uint8_t shift
) {
    if (opcode == LOADBU) {
        return shift;
    }
    assert(opcode == STOREB);
    return 8 + shift;
}

__device__ __forceinline__ uint32_t loadstore_halfword_case_idx(
    Rv64LoadStoreOpcode opcode,
    uint8_t shift
) {
    if (opcode == LOADHU) {
        return shift >> 1;
    }
    assert(opcode == STOREH);
    return 4 + (shift >> 1);
}

__device__ __forceinline__ uint32_t loadstore_word_case_idx(
    Rv64LoadStoreOpcode opcode,
    uint8_t shift
) {
    if (opcode == LOADWU) {
        return shift >> 2;
    }
    assert(opcode == STOREW);
    return 2 + (shift >> 2);
}

struct LoadStoreByteCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ LoadStoreByteCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadStoreRecord record) {
        Rv64LoadStoreOpcode opcode = static_cast<Rv64LoadStoreOpcode>(record.local_opcode);
        uint8_t shift = record.shift_amount;
        uint8_t cell_shift = shift >> 1;
        bool is_store = opcode == STOREB;
        uint16_t read_cell = is_store ? record.read_data[0] : record.read_data[cell_shift];

        uint16_t read_cell_bytes[2] = {
            byte_from_cell(read_cell, 0),
            byte_from_cell(read_cell, 1),
        };
        bitwise_lookup.add_range(read_cell_bytes[0], read_cell_bytes[1]);

        uint16_t prev_cell_bytes[2] = {0, 0};
        if (is_store) {
            prev_cell_bytes[0] = byte_from_cell(record.prev_data[cell_shift], 0);
            prev_cell_bytes[1] = byte_from_cell(record.prev_data[cell_shift], 1);
            bitwise_lookup.add_range(prev_cell_bytes[0], prev_cell_bytes[1]);
        }

        Encoder encoder(
            LOADSTORE_BYTE_CASES,
            LOADSTORE_SELECTOR_MAX_DEGREE,
            true,
            LOADSTORE_BYTE_SELECTOR_WIDTH
        );
        encoder.write_flag_pt(
            row.slice_from(COL_INDEX(LoadStoreByteCoreCols, selector)),
            loadstore_byte_case_idx(opcode, shift)
        );
        COL_WRITE_VALUE(row, LoadStoreByteCoreCols, is_valid, 1);
        COL_WRITE_VALUE(row, LoadStoreByteCoreCols, is_load, opcode == LOADBU);
        COL_WRITE_ARRAY(row, LoadStoreByteCoreCols, read_cell_bytes, read_cell_bytes);
        COL_WRITE_ARRAY(row, LoadStoreByteCoreCols, prev_cell_bytes, prev_cell_bytes);
        COL_WRITE_ARRAY(row, LoadStoreByteCoreCols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, LoadStoreByteCoreCols, prev_data, record.prev_data);
    }
};

template <size_t SELECTOR_WIDTH, uint32_t CASES> struct LoadStoreAlignedCore {
    __device__ void fill_trace_row(RowSlice row, LoadStoreRecord record) {
        Rv64LoadStoreOpcode opcode = static_cast<Rv64LoadStoreOpcode>(record.local_opcode);
        uint8_t shift = record.shift_amount;
        uint32_t case_idx;
        if constexpr (CASES == LOADSTORE_HALFWORD_CASES) {
            case_idx = loadstore_halfword_case_idx(opcode, shift);
        } else {
            case_idx = loadstore_word_case_idx(opcode, shift);
        }

        Encoder encoder(CASES, LOADSTORE_SELECTOR_MAX_DEGREE, true, SELECTOR_WIDTH);
        encoder.write_flag_pt(row, case_idx);
        row[SELECTOR_WIDTH] = 1;
        row[SELECTOR_WIDTH + 1] = opcode == LOADHU || opcode == LOADWU;
        row.write_array(SELECTOR_WIDTH + 2, BLOCK_FE_WIDTH, record.read_data);
        row.write_array(
            SELECTOR_WIDTH + 2 + BLOCK_FE_WIDTH,
            BLOCK_FE_WIDTH,
            record.prev_data
        );
    }
};

struct LoadStoreDoublewordCore {
    __device__ void fill_trace_row(RowSlice row, LoadStoreRecord record) {
        Rv64LoadStoreOpcode opcode = static_cast<Rv64LoadStoreOpcode>(record.local_opcode);
        assert(record.shift_amount == 0);
        assert(opcode == LOADD || opcode == STORED);

        COL_WRITE_VALUE(row, LoadStoreDoublewordCoreCols, is_valid, 1);
        COL_WRITE_VALUE(row, LoadStoreDoublewordCoreCols, is_load, opcode == LOADD);
        COL_WRITE_ARRAY(row, LoadStoreDoublewordCoreCols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, LoadStoreDoublewordCoreCols, prev_data, record.prev_data);
    }
};

template <template <typename> typename Cols, typename Core>
__global__ void rv64_loadstore_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> records,
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

        auto adapter = Rv64LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        if constexpr (std::is_same_v<Core, LoadStoreByteCore>) {
            auto core = LoadStoreByteCore(BitwiseOperationLookup(bitwise_lookup_ptr));
            core.fill_trace_row(row.slice_from(COL_INDEX(Cols, core)), record.core);
        } else {
            Core core;
            core.fill_trace_row(row.slice_from(COL_INDEX(Cols, core)), record.core);
        }
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_load_store_byte_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOADSTORE_BYTE_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_loadstore_tracegen<Rv64LoadStoreByteCols, LoadStoreByteCore>
        <<<grid, block, 0, stream>>>(
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

extern "C" int _rv64_load_store_halfword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOADSTORE_HALFWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    using Core = LoadStoreAlignedCore<LOADSTORE_HALFWORD_SELECTOR_WIDTH, LOADSTORE_HALFWORD_CASES>;
    rv64_loadstore_tracegen<Rv64LoadStoreHalfwordCols, Core>
        <<<grid, block, 0, stream>>>(
            d_trace,
            height,
            width,
            d_records,
            pointer_max_bits,
            d_range_checker,
            range_checker_num_bins,
            nullptr,
            timestamp_max_bits
        );
    return CHECK_KERNEL();
}

extern "C" int _rv64_load_store_word_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOADSTORE_WORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    using Core = LoadStoreAlignedCore<LOADSTORE_WORD_SELECTOR_WIDTH, LOADSTORE_WORD_CASES>;
    rv64_loadstore_tracegen<Rv64LoadStoreWordCols, Core>
        <<<grid, block, 0, stream>>>(
            d_trace,
            height,
            width,
            d_records,
            pointer_max_bits,
            d_range_checker,
            range_checker_num_bins,
            nullptr,
            timestamp_max_bits
        );
    return CHECK_KERNEL();
}

extern "C" int _rv64_load_store_doubleword_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadStoreRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == RV64_LOADSTORE_DOUBLEWORD_WIDTH);
    auto [grid, block] = kernel_launch_params(height, 512);
    rv64_loadstore_tracegen<Rv64LoadStoreDoublewordCols, LoadStoreDoublewordCore>
        <<<grid, block, 0, stream>>>(
            d_trace,
            height,
            width,
            d_records,
            pointer_max_bits,
            d_range_checker,
            range_checker_num_bins,
            nullptr,
            timestamp_max_bits
        );
    return CHECK_KERNEL();
}
