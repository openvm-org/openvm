
#include "adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "constants.h"         // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

static constexpr uint8_t BEQ = 0;
// static constexpr uint8_t BNE = 1;

template <typename T> struct BranchEqualCoreCols {
    T a[RV32_REGISTER_NUM_LIMBS];
    T b[RV32_REGISTER_NUM_LIMBS];

    T cmp_result;
    T imm;

    T opcode_beq_flag;
    T opcode_bne_flag;

    T diff_inv_marker[RV32_REGISTER_NUM_LIMBS];
};

struct BranchEqualCoreRecord {
    uint8_t a[RV32_REGISTER_NUM_LIMBS];
    uint8_t b[RV32_REGISTER_NUM_LIMBS];
    uint32_t imm;
    uint8_t local_opcode;
};

template <typename T> struct BranchEqualCols {
    Rv32BranchAdapterCols<T> adapter;
    BranchEqualCoreCols<T> core;
};

struct BranchEqualRecord {
    Rv32BranchAdapterRecord adapter;
    BranchEqualCoreRecord core;
};

struct BranchEqualCoreTracer {
    __device__ BranchEqualCoreTracer() {}

    __device__ void fill_trace_row(RowSlice row, BranchEqualCoreRecord rec) {
        bool cmp_result;
        size_t diff_idx = RV32_REGISTER_NUM_LIMBS;
        Fp diff_inv_val = Fp::zero();

        for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; ++i) {
            if (rec.a[i] != rec.b[i]) {
                diff_idx = i;
                break;
            }
        }

        bool is_beq = (rec.local_opcode == BEQ);
        if (diff_idx == RV32_REGISTER_NUM_LIMBS) {
            cmp_result = is_beq;
            diff_idx = 0;
        } else {
            cmp_result = !is_beq;
            Fp diff = Fp(rec.a[diff_idx]) - Fp(rec.b[diff_idx]);
            diff_inv_val = inv(diff);
        }

        COL_FILL_ZERO(row, BranchEqualCoreCols, diff_inv_marker);
        COL_WRITE_VALUE(row, BranchEqualCoreCols, diff_inv_marker[diff_idx], diff_inv_val);

        COL_WRITE_VALUE(row, BranchEqualCoreCols, opcode_bne_flag, !is_beq);
        COL_WRITE_VALUE(row, BranchEqualCoreCols, opcode_beq_flag, is_beq);

        COL_WRITE_VALUE(row, BranchEqualCoreCols, imm, rec.imm);
        COL_WRITE_VALUE(row, BranchEqualCoreCols, cmp_result, cmp_result);

        COL_WRITE_ARRAY(row, BranchEqualCoreCols, b, rec.b);
        COL_WRITE_ARRAY(row, BranchEqualCoreCols, a, rec.a);
    }
};

__global__ void beq_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t *bw_ptr,
    uint32_t bw_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < num_records) {
        auto full = reinterpret_cast<BranchEqualRecord *>(records)[idx];

        Rv32BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins));
        adapter.fill_trace_row(row, full.adapter);

        BranchEqualCoreTracer core;
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchEqualCols, core)), full.core);
    } else {
        constexpr size_t W = sizeof(BranchEqualCols<uint8_t>);
#pragma unroll
        for (size_t c = 0; c < W; ++c) {
            row.write(c, 0);
        }
    }
}

extern "C" int _beq_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t *d_bw,
    uint32_t bw_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(BranchEqualRecord) >= record_len);
    assert(width == sizeof(BranchEqualCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);
    beq_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        record_len / sizeof(BranchEqualRecord),
        d_rc,
        rc_bins,
        d_bw,
        bw_bits
    );
    return cudaGetLastError();
}
