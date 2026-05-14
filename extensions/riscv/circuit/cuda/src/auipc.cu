#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

// Pattern B u16: rd_data is 2 u16 limbs (low 32 bits of rd); imm stored as 1 byte + 1 u16
// (`imm = imm_low_8 + imm_high_16 * 256`).
template <typename T> struct Rv64AuipcCoreCols {
    T is_valid;
    T is_sign_extend;
    T imm_low_8;   // imm & 0xff
    T imm_high_16; // (imm >> 8) & 0xffff
    T rd_data[2];  // 2 u16 limbs of rd_low_32
};

struct Rv64AuipcCoreRecord {
    uint32_t from_pc;
    uint32_t imm;
};

__device__ uint32_t run_auipc(uint32_t pc, uint32_t imm) { return pc + (imm << RV64_CELL_BITS); }

struct Rv64AuipcCore {
    BitwiseOperationLookup bitwise_lookup;
    VariableRangeChecker range_checker;

    __device__ Rv64AuipcCore(
        BitwiseOperationLookup bitwise_lookup,
        VariableRangeChecker range_checker
    )
        : bitwise_lookup(bitwise_lookup), range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, Rv64AuipcCoreRecord record) {
        auto imm_bytes = reinterpret_cast<uint8_t *>(&record.imm);
        uint32_t imm_low_8 = imm_bytes[0];
        uint32_t imm_high_16 = (uint32_t)imm_bytes[1] | ((uint32_t)imm_bytes[2] << 8);
        auto auipc = run_auipc(record.from_pc, record.imm);
        uint16_t rd_lo = (uint16_t)(auipc & 0xffff);
        uint16_t rd_hi = (uint16_t)(auipc >> 16);
        uint32_t is_sign_ext = (rd_hi >> 15) & 1;

        // Range checks: low byte via bitwise lookup, high 16 bits + rd via range checker.
        bitwise_lookup.add_range(imm_low_8, imm_low_8);
        range_checker.add_count(imm_high_16, 16);
        range_checker.add_count(rd_lo, 16);
        range_checker.add_count(rd_hi, 16);
        // Sign-extension consistency: 2 * rd_hi - is_sign_ext * 2^16 ∈ [0, 2^16).
        range_checker.add_count(2u * rd_hi - (is_sign_ext << 16), 16);

        uint32_t rd_u16[2] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_low_8, imm_low_8);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_high_16, imm_high_16);
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
    uint32_t *bitwise_lookup_ptr,
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

        auto core = Rv64AuipcCore(
            BitwiseOperationLookup(bitwise_lookup_ptr),
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins)
        );
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
    uint32_t *d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64AuipcCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    auipc_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
