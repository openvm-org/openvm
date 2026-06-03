#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/rdwrite.cuh"

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

__device__ uint32_t run_auipc(uint32_t pc, uint32_t imm) { return pc + (imm << RV64_BYTE_BITS); }

struct Rv64AuipcCore {
    VariableRangeChecker range_checker;

    __device__ Rv64AuipcCore(VariableRangeChecker range_checker) : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, Rv64AuipcCoreRecord record) {
        uint32_t imm_low_8 = record.imm & ((1u << RV64_BYTE_BITS) - 1u);
        uint32_t imm_high_16 = (record.imm >> RV64_BYTE_BITS) & uint32_t(UINT16_MAX);
        uint16_t pc_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(pc_limbs, record.from_pc);
        auto auipc = run_auipc(record.from_pc, record.imm);
        uint16_t rd_u16[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rd_u16, auipc);
        uint16_t rd_hi = rd_u16[1];
        uint32_t is_sign_ext = (rd_hi >> (U16_BITS - 1)) & 1;

        range_checker.add_count(pc_limbs[0], U16_BITS);
        range_checker.add_count(pc_limbs[1], PC_BITS - U16_BITS);
        range_checker.add_count(imm_low_8, RV64_BYTE_BITS);
        range_checker.add_count(imm_high_16, U16_BITS);
        range_checker.add_count(rd_u16[0], U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        range_checker.add_count(
            2u * rd_hi - (is_sign_ext << U16_BITS), U16_BITS
        );

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
    auto [grid, block] = kernel_launch_params(height);
    auipc_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
