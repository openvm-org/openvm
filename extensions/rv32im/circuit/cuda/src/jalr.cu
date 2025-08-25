#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/jalr.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv32JalrCoreCols {
    T imm;                                  // 1 byte
    T rs1_data[RV32_REGISTER_NUM_LIMBS];    // 4 bytes
    T rd_data[RV32_REGISTER_NUM_LIMBS - 1]; // 3 bytes
    T is_valid;                             // 1 byte
    T to_pc_least_sig_bit;                  // 1 byte
    T to_pc_limbs[2];                       // 2 bytes
    T imm_sign;                             // 1 byte
};

struct Rv32JalrCoreRecord {
    uint16_t imm;
    uint32_t from_pc;
    uint32_t rs1_val;
    uint8_t imm_sign; // 0 or 1
};

__device__ void run_jalr(
    uint32_t pc,
    uint32_t rs1,
    uint16_t imm,
    bool imm_sign,
    uint32_t &out_pc,
    uint8_t rd_bytes[RV32_REGISTER_NUM_LIMBS]
) {
    uint32_t offset = imm + (imm_sign ? 0xffff0000 : 0);
    uint32_t to_pc = rs1 + offset;

    assert(to_pc < (1u << PC_BITS));
    out_pc = to_pc;
    uint32_t rd_val = pc + DEFAULT_PC_STEP;
    uint8_t *p = reinterpret_cast<uint8_t *>(&rd_val);
#pragma unroll
    for (int i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        rd_bytes[i] = p[i];
    }
}

struct Rv32JalrCore {
    VariableRangeChecker rc;
    BitwiseOperationLookup bw;

    __device__ Rv32JalrCore(VariableRangeChecker rc, BitwiseOperationLookup bw) : rc(rc), bw(bw) {}

    __device__ void fill_trace_row(RowSlice row, Rv32JalrCoreRecord record) {
        uint32_t to_pc;
        uint8_t rd_bytes[RV32_REGISTER_NUM_LIMBS];
        run_jalr(record.from_pc, record.rs1_val, record.imm, record.imm_sign, to_pc, rd_bytes);

        uint32_t to_pc_limbs[2] = {(to_pc & ((1u << 16) - 1)) >> 1, to_pc >> 16};

        rc.add_count(to_pc_limbs[0], 15);
        rc.add_count(to_pc_limbs[1], PC_BITS - 16);
        bw.add_range(rd_bytes[0], rd_bytes[1]);
        rc.add_count(rd_bytes[2], RV32_CELL_BITS);
        rc.add_count(rd_bytes[3], PC_BITS - RV32_CELL_BITS * 3);

        COL_WRITE_VALUE(row, Rv32JalrCoreCols, imm_sign, record.imm_sign);
        COL_WRITE_ARRAY(row, Rv32JalrCoreCols, to_pc_limbs, to_pc_limbs);
        COL_WRITE_VALUE(row, Rv32JalrCoreCols, to_pc_least_sig_bit, (to_pc & 1) == 1 ? 1 : 0);
        COL_WRITE_VALUE(row, Rv32JalrCoreCols, is_valid, 1);

        uint8_t rs1_bytes[RV32_REGISTER_NUM_LIMBS];
        memcpy(rs1_bytes, &record.rs1_val, sizeof(rs1_bytes));
        COL_WRITE_ARRAY(row, Rv32JalrCoreCols, rs1_data, rs1_bytes);
        COL_WRITE_ARRAY(row, Rv32JalrCoreCols, rd_data, rd_bytes + 1);
        COL_WRITE_VALUE(row, Rv32JalrCoreCols, imm, record.imm);
    }
};

template <typename T> struct Rv32JalrCols {
    Rv32JalrAdapterCols<T> adapter;
    Rv32JalrCoreCols<T> core;
};

struct Rv32JalrRecord {
    Rv32JalrAdapterRecord adapter;
    Rv32JalrCoreRecord core;
};

__global__ void jalr_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv32JalrRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto full = records[idx];

        // adapter pass
        Rv32JalrAdapter adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, full.adapter);

        // core pass
        Rv32JalrCore core(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32JalrCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv32JalrCols<uint8_t>));
    }
}

extern "C" int _jalr_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32JalrRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    assert(height >= d_records.len());
    assert(width == sizeof(Rv32JalrCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    jalr_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}
