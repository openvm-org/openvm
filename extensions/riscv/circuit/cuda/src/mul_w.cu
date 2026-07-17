#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul_w.cuh"
#include "riscv/cores/mul.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64.
using Rv64MulWCoreRecord = MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS>;
using Rv64MulWCore = MultiplicationCore<RV64_WORD_NUM_LIMBS>;
template <typename T> using Rv64MulWCoreCols = MultiplicationCoreCols<T, RV64_WORD_NUM_LIMBS>;

template <typename T> struct Rv64MulWCols {
    Rv64MultWAdapterCols<T> adapter;
    Rv64MulWCoreCols<T> core;
};

struct Rv64MulWRecord {
    Rv64MultWAdapterRecord adapter;
    Rv64MulWCoreRecord core;
};

__global__ void rv64_mul_w_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64MulWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64MultWAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv64MulWCore core(range_tuple_checker, BitwiseOperationLookup(d_bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MulWCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MulWCols<uint8_t>));
    }
}

extern "C" int _rv64_mul_w_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64MulWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64MulWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_mul_w_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): compact-wire twin of the kernel above; decodes in registers
// and calls the SAME fill methods.
template <typename RecordView>
__global__ void rv64_mul_w_tracegen_compact(
    Fp *d_trace,
    size_t height,
    RecordView d_records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        RvrAlu3Compact const rec_c = d_records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec_c.from_pc);
        uint32_t const result_word =
            rvr_mul_div_w_result(0xFF, rec_c.b[0], rec_c.c[0]);
        Rv64MulWRecord full;
        full.adapter = rvr_decode_alu3_mult_w(rec_c, entry, result_word);
#pragma unroll
        for (size_t i = 0; i < RV64_WORD_NUM_LIMBS; i++) {
            full.core.b[i] = rvr_u8_limb(rec_c.b, i);
            full.core.c[i] = rvr_u8_limb(rec_c.c, i);
        }

        Rv64MultWAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv64MulWCore core(range_tuple_checker, BitwiseOperationLookup(d_bitwise_lookup_ptr));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MulWCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MulWCols<uint8_t>));
    }
}

extern "C" int _rv64_mul_w_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64MulWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv64_mul_w_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        operand_table,
        pc_base,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _rv64_mul_w_tracegen_g2, Rv64MulWCols, rv64_mul_w_tracegen_compact,
    RvrAlu3Compact, 256, operand_table, pc_base, range_checker,
    size_t(range_checker_num_bins), bitwise_lookup, range_tuple_checker,
    range_tuple_checker_sizes, timestamp_max_bits
)
