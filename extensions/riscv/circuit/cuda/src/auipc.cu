#include <assert.h>

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/rdwrite.cuh"
#include "riscv/rvr_compact.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv64AuipcCoreCols {
    T is_valid;
    T is_sign_extend;
    // The immediate is split around the byte shift in AUIPC's `imm << 8`.
    T imm_low_8;
    T imm_high_16;
    T pc_high;
    T rd_data[RV64_PTR_U16_LIMBS];
};

struct Rv64AuipcCoreRecord {
    uint32_t from_pc;
    uint32_t imm;
};

__device__ uint64_t run_auipc(uint32_t pc, uint32_t imm) {
    uint32_t offset = imm << RV64_BYTE_BITS;
    int64_t signed_offset = (int64_t)(int32_t)offset;
    return (uint64_t)pc + (uint64_t)signed_offset;
}

struct Rv64AuipcCore {
    VariableRangeChecker range_checker;

    __device__ Rv64AuipcCore(VariableRangeChecker range_checker) : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, Rv64AuipcCoreRecord record) {
        uint32_t imm_low_8 = record.imm & ((1u << RV64_BYTE_BITS) - 1u);
        uint32_t imm_high_16 = (record.imm >> RV64_BYTE_BITS) & uint32_t(UINT16_MAX);
        uint16_t pc_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(pc_limbs, record.from_pc);
        uint64_t auipc = run_auipc(record.from_pc, record.imm);
        uint64_t auipc_hi = auipc >> 32;
        assert(auipc_hi == 0ull || auipc_hi == 0xffffffffull);
        uint32_t auipc_lo = (uint32_t)auipc;
        uint16_t rd_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rd_limbs, auipc_lo);
        uint32_t rd_lo = rd_limbs[0];
        uint32_t rd_hi = rd_limbs[1];
        uint32_t is_sign_ext = (auipc_hi != 0) ? 1u : 0u;
        uint32_t imm_sign = (imm_high_16 >> (U16_BITS - 1)) & 1u;

        range_checker.add_count(pc_limbs[0], U16_BITS);
        range_checker.add_count(pc_limbs[1], PC_BITS - U16_BITS);
        range_checker.add_count(imm_low_8, RV64_BYTE_BITS);
        range_checker.add_count(imm_high_16, U16_BITS);
        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        // Check that imm_sign matches the top bit of imm_high_16.
        range_checker.add_count(2u * imm_high_16 - (imm_sign << U16_BITS), U16_BITS);

        uint32_t rd_u16[RV64_PTR_U16_LIMBS] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_low_8, imm_low_8);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_high_16, imm_high_16);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, pc_high, pc_limbs[1]);
        COL_WRITE_ARRAY(row, Rv64AuipcCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, is_sign_extend, is_sign_ext);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, is_valid, 1);
    }
};

template <typename T> struct Rv64AuipcCols {
    Rv64RdWriteAdapterCols<T> adapter;
    Rv64AuipcCoreCols<T> core;
};

struct Rv64AuipcRecord {
    Rv64RdWriteAdapterRecord adapter;
    Rv64AuipcCoreRecord core;
};

__global__ void auipc_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv64AuipcRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64RdWriteAdapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core =
            Rv64AuipcCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AuipcCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AuipcCols<uint8_t>));
    }
}

extern "C" int _auipc_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64AuipcRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AuipcCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);
    auipc_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): tracegen from compact wire records + the per-exe operand
// table; materializes the same record structs in registers and calls the SAME
// fill methods as auipc_tracegen.
__global__ void auipc_tracegen_compact(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrWr1Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        RvrWr1Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);

        Rv64AuipcRecord full;
        full.adapter = rvr_decode_wr1_adapter(rec, entry);
        full.core.from_pc = rec.from_pc;
        full.core.imm = entry.c;
        auto adapter = Rv64RdWriteAdapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);
        auto core =
            Rv64AuipcCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64AuipcCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64AuipcCols<uint8_t>));
    }
}

extern "C" int _auipc_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrWr1Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AuipcCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    auipc_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_operand_table, pc_base, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
