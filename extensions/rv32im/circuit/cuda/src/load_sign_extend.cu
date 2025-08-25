#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/loadstore.cuh"

using namespace riscv;
using namespace program;

template <typename T, size_t NUM_CELLS> struct LoadSignExtendCoreCols {
    /// This chip treats loadb with 0 shift and loadb with 1 shift as different instructions
    T opcode_loadb_flag0;
    T opcode_loadb_flag1;
    T opcode_loadh_flag;

    T shift_most_sig_bit;
    // The bit that is extended to the remaining bits
    T data_most_sig_bit;

    T shifted_read_data[NUM_CELLS];
    T prev_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadSignExtendCoreRecord {
    bool is_byte;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
    uint8_t prev_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadSignExtendCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = LoadSignExtendCoreCols<T, NUM_CELLS>;

    __device__ LoadSignExtendCore(VariableRangeChecker range_checker)
        : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, LoadSignExtendCoreRecord<NUM_CELLS> record) {
        uint8_t shift = record.shift_amount;

        uint8_t most_sig_limb;
        if (record.is_byte) {
            most_sig_limb = record.read_data[shift];
        } else {
            most_sig_limb = record.read_data[NUM_CELLS / 2 - 1 + shift];
        }

        uint8_t most_sig_bit = most_sig_limb & 0x80;

        range_checker.add_count(most_sig_limb - most_sig_bit, 7);
        COL_WRITE_VALUE(row, Cols, opcode_loadb_flag0, record.is_byte && ((shift & 1) == 0));
        COL_WRITE_VALUE(row, Cols, opcode_loadb_flag1, record.is_byte && ((shift & 1) == 1));
        COL_WRITE_VALUE(row, Cols, opcode_loadh_flag, !record.is_byte);

        COL_WRITE_VALUE(row, Cols, data_most_sig_bit, most_sig_bit != 0);

        if ((shift & 2) != 0) {
            COL_WRITE_VALUE(row, Cols, shift_most_sig_bit, 1);
            // Shift the read data by 2 places to the left
#pragma unroll
            for (size_t i = 0; i < NUM_CELLS - 2; i++) {
                COL_WRITE_VALUE(row, Cols, shifted_read_data[i], record.read_data[i + 2]);
            }
            COL_WRITE_VALUE(row, Cols, shifted_read_data[NUM_CELLS - 2], record.read_data[0]);
            COL_WRITE_VALUE(row, Cols, shifted_read_data[NUM_CELLS - 1], record.read_data[1]);
        } else {
            COL_WRITE_VALUE(row, Cols, shift_most_sig_bit, 0);
            COL_WRITE_ARRAY(row, Cols, shifted_read_data, record.read_data);
        }

        COL_WRITE_ARRAY(row, Cols, prev_data, record.prev_data);
    }
};

// [Adapter + Core] columns and record
template <typename T> struct Rv32LoadSignExtendCols {
    Rv32LoadStoreAdapterCols<T> adapter;
    LoadSignExtendCoreCols<T, RV32_REGISTER_NUM_LIMBS> core;
};

struct Rv32LoadSignExtendRecord {
    Rv32LoadStoreAdapterRecord adapter;
    LoadSignExtendCoreRecord<RV32_REGISTER_NUM_LIMBS> core;
};

__global__ void rv32_load_sign_extend_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32LoadSignExtendRecord> records,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv32LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = LoadSignExtendCore<RV32_REGISTER_NUM_LIMBS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32LoadSignExtendCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv32LoadSignExtendCols<uint8_t>));
    }
}

extern "C" int _rv32_load_sign_extend_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32LoadSignExtendRecord> d_records,
    size_t pointer_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(Rv32LoadSignExtendCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv32_load_sign_extend_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return cudaGetLastError();
}
