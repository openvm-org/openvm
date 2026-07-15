#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "riscv/rvr_compact.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64AddSubCoreRecord = AddSubCoreRecord<BLOCK_FE_WIDTH>;
using Rv64AddSubCore = AddSubCore<BLOCK_FE_WIDTH, U16_BITS, true>;
template <typename T> using Rv64AddSubCoreCols = AddSubCoreCols<T, BLOCK_FE_WIDTH>;

template <typename T> struct Rv64AddSubCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64AddSubCoreCols<T> core;
};

struct Rv64AddSubRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    Rv64AddSubCoreRecord core;
};

static_assert(sizeof(Rv64AddSubCoreRecord) == 18);
static_assert(sizeof(Rv64AddSubRecord) == 60);
static_assert(offsetof(Rv64AddSubRecord, core) == 40);

__global__ void add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        auto core =
            Rv64AddSubCore(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddSubCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddSubCols<uint8_t>));
    }
}

extern "C" int _add_sub_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AddSubRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    add_sub_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): tracegen from compact alu3 wire records + the per-exe operand
// table. Materializes the same record structs in registers and calls the SAME
// fill methods as add_sub_tracegen — byte-equality by construction modulo the
// decode, which the three-way differential validates on device.
__global__ void add_sub_tracegen_compact(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        RvrAlu3Compact const rec = d_records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(d_operand_table, pc_base, rec.from_pc);

        Rv64AddSubRecord full;
        full.adapter = rvr_decode_alu3_alu_reg_u16(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.b[i] = rvr_u16_limb(rec.b, i);
            full.core.c[i] = rvr_u16_limb(rec.c, i);
        }
        full.core.local_opcode = entry.local_opcode;

        auto adapter = Rv64BaseAluRegU16Adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        auto core =
            Rv64AddSubCore(VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AddSubCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AddSubCols<uint8_t>));
    }
}

extern "C" int _add_sub_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrAlu3Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AddSubCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    add_sub_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_operand_table,
        pc_base,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
