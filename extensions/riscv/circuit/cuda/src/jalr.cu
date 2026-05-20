#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/constants.cuh"
#include "riscv/adapters/jalr.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv64JalrCoreCols {
    T imm;
    T rs1_data[RV64_PTR_U16_LIMBS];
    T rd_data[RV64_PTR_U16_LIMBS - 1];
    T is_valid;
    T to_pc_least_sig_bit;
    T to_pc_limbs[2];
    T imm_sign;
};

struct Rv64JalrCoreRecord {
    uint16_t imm;
    uint32_t from_pc;
    uint32_t rs1_val;
    uint8_t imm_sign; // 0 or 1
};

struct Rv64JalrCore {
    VariableRangeChecker rc;

    __device__ Rv64JalrCore(VariableRangeChecker rc) : rc(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64JalrCoreRecord record) {
        uint32_t offset =
            record.imm + (record.imm_sign ? (uint32_t(UINT16_MAX) << U16_BITS) : 0);
        uint32_t to_pc = record.rs1_val + offset;
        assert(to_pc < (1u << PC_BITS));

        uint32_t to_pc_limbs[2] = {
            (to_pc & uint32_t(UINT16_MAX)) >> 1, to_pc >> U16_BITS
        };
        // to_pc_limbs[0] is 15 bits because it is doubled to reconstruct
        // the aligned JALR target.
        rc.add_count(to_pc_limbs[0], U16_BITS - 1);
        rc.add_count(to_pc_limbs[1], PC_BITS - U16_BITS);

        uint32_t rd_low_u32 = record.from_pc + DEFAULT_PC_STEP;
        uint32_t rd_low_u16_lo = rd_low_u32 & uint32_t(UINT16_MAX);
        uint32_t rd_low_u16_hi = (rd_low_u32 >> U16_BITS) & uint32_t(UINT16_MAX);

        // rd writes the low 32 bits of from_pc + DEFAULT_PC_STEP. The high
        // limb is narrowed to the remaining PC bits because from_pc is program-bus bounded.
        rc.add_count(rd_low_u16_lo, U16_BITS);
        rc.add_count(rd_low_u16_hi, PC_BITS - U16_BITS);

        uint32_t rs1_u16_lo = record.rs1_val & uint32_t(UINT16_MAX);
        uint32_t rs1_u16_hi = (record.rs1_val >> U16_BITS) & uint32_t(UINT16_MAX);
        // rs1 is read as two u16 cells.
        rc.add_count(rs1_u16_lo, U16_BITS);
        rc.add_count(rs1_u16_hi, U16_BITS);

        COL_WRITE_VALUE(row, Rv64JalrCoreCols, imm_sign, record.imm_sign);
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, to_pc_limbs, to_pc_limbs);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, to_pc_least_sig_bit, (to_pc & 1) == 1 ? 1 : 0);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, is_valid, 1);

        uint32_t rs1_limbs[RV64_PTR_U16_LIMBS] = {rs1_u16_lo, rs1_u16_hi};
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, rs1_data, rs1_limbs);
        uint32_t rd_limbs[RV64_PTR_U16_LIMBS - 1] = {rd_low_u16_hi};
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, rd_data, rd_limbs);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, imm, record.imm);
    }
};

template <typename T> struct Rv64JalrCols {
    Rv64JalrAdapterCols<T> adapter;
    Rv64JalrCoreCols<T> core;
};

struct Rv64JalrRecord {
    Rv64JalrAdapterRecord adapter;
    Rv64JalrCoreRecord core;
};

__global__ void jalr_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv64JalrRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto full = records[idx];

        Rv64JalrAdapter adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        Rv64JalrCore core(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64JalrCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64JalrCols<uint8_t>));
    }
}

extern "C" int _jalr_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64JalrRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalrCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    jalr_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
