#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv32AuipcCoreCols {
    T is_valid;
    // The limbs of the immediate except the least significant limb since it is always 0
    T imm_limbs[RV32_REGISTER_NUM_LIMBS - 1];
    // The limbs of the PC except the most significant and the least significant limbs
    T pc_limbs[RV32_REGISTER_NUM_LIMBS - 2];
    T rd_data[RV32_REGISTER_NUM_LIMBS];
};

struct Rv32AuipcCoreRecord {
    uint32_t from_pc;
    uint32_t imm;
};

__device__ uint32_t run_auipc(uint32_t pc, uint32_t imm) { return pc + (imm << RV32_CELL_BITS); }

struct Rv32AuipcCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ Rv32AuipcCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, Rv32AuipcCoreRecord record) {
        auto pc_limbs = reinterpret_cast<uint8_t *>(&record.from_pc);
        auto imm_limbs = reinterpret_cast<uint8_t *>(&record.imm);
        auto auipc = run_auipc(record.from_pc, record.imm);
        auto rd_data = reinterpret_cast<uint8_t *>(&auipc);

        bitwise_lookup.add_range(imm_limbs[0], imm_limbs[1]);
        bitwise_lookup.add_range(imm_limbs[2], pc_limbs[1]);
        auto msl_shift = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - PC_BITS;
        bitwise_lookup.add_range(pc_limbs[2], pc_limbs[3] << msl_shift);
#pragma unroll
        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i += 2) {
            bitwise_lookup.add_range(rd_data[i], rd_data[i + 1]);
        }

        COL_WRITE_ARRAY(row, Rv32AuipcCoreCols, imm_limbs, imm_limbs);
        COL_WRITE_ARRAY(row, Rv32AuipcCoreCols, pc_limbs, pc_limbs + 1);
        COL_WRITE_ARRAY(row, Rv32AuipcCoreCols, rd_data, rd_data);
        COL_WRITE_VALUE(row, Rv32AuipcCoreCols, is_valid, 1);
    }
};

template <typename T> struct Rv32AuipcCols {
    Rv32RdWriteAdapterCols<T> adapter;
    Rv32AuipcCoreCols<T> core;
};

struct Rv32AuipcRecord {
    Rv32RdWriteAdapterRecord adapter;
    Rv32AuipcCoreRecord core;
};

__global__ void auipc_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv32AuipcRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv32RdWriteAdapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core = Rv32AuipcCore(BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32AuipcCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv32AuipcCols<uint8_t>));
    }
}

extern "C" int _auipc_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32AuipcRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv32AuipcCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    auipc_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
