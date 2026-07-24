#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/mul.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

using namespace riscv;

// Concrete type aliases for 64-bit
using Rv64MultiplicationCoreRecord = MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64MultiplicationCore = MultiplicationCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv64MultiplicationCoreCols = MultiplicationCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64MultiplicationCols {
    Rv64MultAdapterCols<T> adapter;
    Rv64MultiplicationCoreCols<T> core;
};

struct Rv64MultiplicationRecord {
    Rv64MultAdapterRecord adapter;
    Rv64MultiplicationCoreRecord core;
};

__global__ void mul_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64MultiplicationRecord> d_records,
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

        Rv64MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr);
        Rv64MultiplicationCore core(range_tuple_checker, bitwise_lookup);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MultiplicationCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MultiplicationCols<uint8_t>));
    }
}

extern "C" int _mul_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64MultiplicationCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    mul_tracegen<<<grid, block, 0, stream>>>(
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
__global__ void mul_tracegen_compact(
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
        Rv64MultiplicationRecord full;
        full.adapter = rvr_decode_alu3_mult(rec_c, entry);
#pragma unroll
        for (size_t i = 0; i < RV64_REGISTER_NUM_LIMBS; i++) {
            full.core.b[i] = rvr_u8_limb(rec_c.b, i);
            full.core.c[i] = rvr_u8_limb(rec_c.c, i);
        }

        Rv64MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr);
        Rv64MultiplicationCore core(range_tuple_checker, bitwise_lookup);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MultiplicationCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MultiplicationCols<uint8_t>));
    }
}

extern "C" int _mul_tracegen_compact(
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
#ifdef OPENVM_RVR_CUDA_G2_ONLY
    return int(cudaErrorNotSupported);
#else
    assert(width == sizeof(Rv64MultiplicationCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    mul_tracegen_compact<<<grid, block, 0, stream>>>(
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
#endif
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _mul_tracegen_g2, Rv64MultiplicationCols, mul_tracegen_compact, RvrAlu3Compact, 512,
    operand_table, pc_base, range_checker, size_t(range_checker_num_bins), bitwise_lookup,
    range_tuple_checker, range_tuple_checker_sizes, timestamp_max_bits
)
