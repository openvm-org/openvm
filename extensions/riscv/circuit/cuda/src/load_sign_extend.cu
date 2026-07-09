#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/loadstore.cuh"

using namespace riscv;
using namespace program;

// One case per (opcode, shift) pair, op order [LOADB, LOADH, LOADW], case index
// op_idx * 8 + shift. Access width is 1 << op_idx.
constexpr size_t LOAD_SIGN_EXTEND_CASES = 24;

template <typename T, size_t NUM_CELLS> struct LoadSignExtendCoreCols {
    /// One boolean flag per (opcode, shift) case.
    T flags[LOAD_SIGN_EXTEND_CASES];
    // The bit that is extended to the remaining bits
    T data_most_sig_bit;

    T read_data[NUM_CELLS];
    T read_data1[NUM_CELLS];
    T prev_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadSignExtendCoreRecord {
    bool is_byte;
    bool is_word;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
    uint8_t read_data1[NUM_CELLS];
    uint8_t prev_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadSignExtendCore {
    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = LoadSignExtendCoreCols<T, NUM_CELLS>;

    __device__ LoadSignExtendCore(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : range_checker(range_checker), bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LoadSignExtendCoreRecord<NUM_CELLS> record) {
        size_t shift = record.shift_amount;
        size_t op_idx = record.is_byte ? 0 : (record.is_word ? 2 : 1);
        size_t width = size_t(1) << op_idx;
        size_t case_idx = op_idx * NUM_CELLS + shift;
        bool crosses = shift + width > NUM_CELLS;

        // Byte at global position k in read_data ++ read_data1.
        size_t top = shift + width - 1;
        uint8_t most_sig_limb =
            top < NUM_CELLS ? record.read_data[top] : record.read_data1[top - NUM_CELLS];
        uint8_t most_sig_bit = most_sig_limb & (1u << (RV64_BYTE_BITS - 1));

        range_checker.add_count(most_sig_limb - most_sig_bit, RV64_BYTE_BITS - 1);
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i += 2) {
            bitwise_lookup.add_range(record.read_data[i], record.read_data[i + 1]);
        }
        if (crosses) {
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS; i += 2) {
                bitwise_lookup.add_range(record.read_data1[i], record.read_data1[i + 1]);
            }
        }

#pragma unroll
        for (size_t i = 0; i < LOAD_SIGN_EXTEND_CASES; i++) {
            row[COL_INDEX(Cols, flags) + i] = Fp(i == case_idx);
        }
        COL_WRITE_VALUE(row, Cols, data_most_sig_bit, most_sig_bit != 0);
        COL_WRITE_ARRAY(row, Cols, read_data, record.read_data);
        COL_WRITE_ARRAY(row, Cols, read_data1, record.read_data1);
        COL_WRITE_ARRAY(row, Cols, prev_data, record.prev_data);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct Rv64LoadSignExtendCols {
    Rv64LoadStoreAdapterCols<T> adapter;
    LoadSignExtendCoreCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct Rv64LoadSignExtendRecord {
    Rv64LoadStoreAdapterRecord adapter;
    LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS> core;
};

__global__ void rv64_load_sign_extend_tracegen(
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

        auto adapter = Rv64LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = LoadSignExtendCore<RV64_REGISTER_NUM_LIMBS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadSignExtendCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64LoadSignExtendCols<uint8_t>));
    }
}

extern "C" int _rv64_load_sign_extend_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendCols<uint8_t>));
    // Capped below the previous 512: the two-block sign-extension fill increased the
    // per-thread register footprint.
    auto [grid, block] = kernel_launch_params(height, 256);

    rv64_load_sign_extend_tracegen<<<grid, block, 0, stream>>>(
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
