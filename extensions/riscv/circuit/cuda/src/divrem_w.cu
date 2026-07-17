#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul_w.cuh"
#include "riscv/cores/divrem.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

using namespace riscv;

template <typename T> struct Rv64DivRemWCols {
    Rv64MultWAdapterCols<T> adapter;
    DivRemCoreCols<T, RV64_WORD_NUM_LIMBS> core;
};

struct Rv64DivRemWRecord {
    Rv64MultWAdapterRecord adapter;
    DivRemCoreRecords<RV64_WORD_NUM_LIMBS> core;
};

__global__ void rv64_div_rem_w_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64DivRemWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_bits,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &record = d_records[idx];

        Rv64MultWAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bits),
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        DivRemCore<RV64_WORD_NUM_LIMBS> core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            RangeTupleChecker<2>(
                d_range_tuple_checker_ptr,
                (uint32_t[2]){range_tuple_checker_sizes.x, range_tuple_checker_sizes.y}
            )
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64DivRemWCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64DivRemWCols<uint8_t>));
    }
}

extern "C" int _rv64_div_rem_w_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64DivRemWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64DivRemWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_div_rem_w_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        d_range_tuple_checker_ptr,
        range_tuple_checker_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): compact-wire twin of the kernel above; decodes in registers
// and calls the SAME fill methods.
template <typename RecordView>
__global__ void rv64_div_rem_w_tracegen_compact(
    Fp *d_trace,
    size_t height,
    RecordView d_records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_bits,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        RvrAlu3Compact const rec_c = d_records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec_c.from_pc);
        uint32_t const result_word =
            rvr_mul_div_w_result(entry.local_opcode, rec_c.b[0], rec_c.c[0]);
        Rv64DivRemWRecord full;
        full.adapter = rvr_decode_alu3_mult_w(rec_c, entry, result_word);
#pragma unroll
        for (size_t i = 0; i < RV64_WORD_NUM_LIMBS; i++) {
            full.core.b[i] = rvr_u8_limb(rec_c.b, i);
            full.core.c[i] = rvr_u8_limb(rec_c.c, i);
        }
        full.core.local_opcode = entry.local_opcode;

        Rv64MultWAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bits),
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        DivRemCore<RV64_WORD_NUM_LIMBS> core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            RangeTupleChecker<2>(
                d_range_tuple_checker_ptr,
                (uint32_t[2]){range_tuple_checker_sizes.x, range_tuple_checker_sizes.y}
            )
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64DivRemWCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64DivRemWCols<uint8_t>));
    }
}

extern "C" int _rv64_div_rem_w_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64DivRemWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_div_rem_w_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        operand_table,
        pc_base,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        d_range_tuple_checker_ptr,
        range_tuple_checker_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _rv64_div_rem_w_tracegen_g2, Rv64DivRemWCols, rv64_div_rem_w_tracegen_compact,
    RvrAlu3Compact, 512, operand_table, pc_base, range_checker,
    range_checker_num_bins, bitwise_lookup, range_tuple_checker,
    range_tuple_checker_sizes, timestamp_max_bits
)
