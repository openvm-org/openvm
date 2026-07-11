#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/loadstore.cuh"
#include "riscv/rvr_compact.cuh"

using namespace riscv;
using namespace program;

template <typename T, size_t NUM_CELLS> struct LoadSignExtendCoreCols {
    /// This chip treats each (opcode, inner_shift) pair as a different instruction
    T opcode_loadb_flag0;
    T opcode_loadb_flag1;
    T opcode_loadb_flag2;
    T opcode_loadb_flag3;
    T opcode_loadh_flag0;
    T opcode_loadh_flag2;
    T opcode_loadw_flag;

    T shift_most_sig_bit;
    // The bit that is extended to the remaining bits
    T data_most_sig_bit;

    T shifted_read_data[NUM_CELLS];
    T prev_data[NUM_CELLS];
};

template <size_t NUM_CELLS> struct LoadSignExtendCoreRecord {
    bool is_byte;
    bool is_word;
    uint8_t shift_amount;
    uint8_t read_data[NUM_CELLS];
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
        uint8_t shift = record.shift_amount;
        uint8_t shift_most_sig_bit = (shift >> 2) & 1;
        uint8_t inner_shift = shift & 3;
        uint8_t rotate = shift_most_sig_bit * (NUM_CELLS / 2);

        uint8_t shifted_read_data[NUM_CELLS];
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i++) {
            shifted_read_data[i] = record.read_data[(i + rotate) % NUM_CELLS];
        }

        uint8_t most_sig_limb;
        if (record.is_byte) {
            most_sig_limb = shifted_read_data[inner_shift];
        } else if (record.is_word) {
            most_sig_limb = shifted_read_data[NUM_CELLS / 2 - 1];
        } else {
            most_sig_limb = shifted_read_data[inner_shift + 1];
        }

        uint8_t most_sig_bit = most_sig_limb & (1u << (RV64_BYTE_BITS - 1));
        bool is_word = record.is_word;
        bool is_half = !record.is_byte && !is_word;

        range_checker.add_count(most_sig_limb - most_sig_bit, RV64_BYTE_BITS - 1);
#pragma unroll
        for (size_t i = 0; i < NUM_CELLS; i += 2) {
            bitwise_lookup.add_range(shifted_read_data[i], shifted_read_data[i + 1]);
        }
        COL_WRITE_VALUE(row, Cols, opcode_loadb_flag0, record.is_byte && inner_shift == 0);
        COL_WRITE_VALUE(row, Cols, opcode_loadb_flag1, record.is_byte && inner_shift == 1);
        COL_WRITE_VALUE(row, Cols, opcode_loadb_flag2, record.is_byte && inner_shift == 2);
        COL_WRITE_VALUE(row, Cols, opcode_loadb_flag3, record.is_byte && inner_shift == 3);
        COL_WRITE_VALUE(row, Cols, opcode_loadh_flag0, is_half && inner_shift == 0);
        COL_WRITE_VALUE(row, Cols, opcode_loadh_flag2, is_half && inner_shift == 2);
        COL_WRITE_VALUE(row, Cols, opcode_loadw_flag, is_word);

        COL_WRITE_VALUE(row, Cols, data_most_sig_bit, most_sig_bit != 0);
        COL_WRITE_VALUE(row, Cols, shift_most_sig_bit, shift_most_sig_bit == 1);
        COL_WRITE_ARRAY(row, Cols, shifted_read_data, shifted_read_data);
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
    auto [grid, block] = kernel_launch_params(height, 512);

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

// M-GPUDEC (G2): compact-wire twin of the kernel above.
__global__ void rv64_load_sign_extend_tracegen_compact(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        RvrAlu3Compact const rec_c = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec_c.from_pc);
        Rv64LoadSignExtendRecord full;
        uint8_t shift_amount;
        full.adapter = rvr_decode_alu3_loadstore_start(rec_c, entry, shift_amount);
        bool const enabled = (entry.flags & RVR_OPERAND_FLAG_WRITE_ENABLED) != 0;
        if (enabled) {
            full.adapter.rd_rs2_ptr = entry.a;
            full.adapter.write_prev_timestamp = rec_c.write_prev_timestamp;
        }
#pragma unroll
        for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
            full.core.read_data[i] = rvr_u8_limb(rec_c.c, i);
            full.core.prev_data[i] = enabled ? rvr_u8_limb(rec_c.write_prev_data, i) : 0;
        }
        full.core.is_byte = (entry.flags & RVR_OPERAND_FLAG_LS_IS_BYTE) != 0;
        full.core.is_word = (entry.flags & RVR_OPERAND_FLAG_LS_IS_WORD) != 0;
        full.core.shift_amount = shift_amount;

        auto adapter = Rv64LoadStoreAdapter(
            pointer_max_bits,
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        auto core = LoadSignExtendCore<RV64_REGISTER_NUM_LIMBS>(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64LoadSignExtendCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64LoadSignExtendCols<uint8_t>));
    }
}

extern "C" int _rv64_load_sign_extend_tracegen_compact(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    size_t pointer_max_bits,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64LoadSignExtendCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_load_sign_extend_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_operand_table,
        pc_base,
        pointer_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
