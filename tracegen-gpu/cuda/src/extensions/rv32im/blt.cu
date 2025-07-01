
#include "adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "constants.h"         // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

static constexpr uint8_t BLT = 0;
static constexpr uint8_t BLTU = 1;
static constexpr uint8_t BGE = 2;
static constexpr uint8_t BGEU = 3;

template <typename T> struct BranchLessThanCoreCols {
    T a[RV32_REGISTER_NUM_LIMBS];
    T b[RV32_REGISTER_NUM_LIMBS];

    T cmp_result;
    T imm;

    T opcode_blt_flag;
    T opcode_bltu_flag;
    T opcode_bge_flag;
    T opcode_bgeu_flag;

    T a_msb_f;
    T b_msb_f;

    T cmp_lt;

    T diff_marker[RV32_REGISTER_NUM_LIMBS];
    T diff_val;
};

struct BranchLessThanCoreRecord {
    uint8_t a[RV32_REGISTER_NUM_LIMBS];
    uint8_t b[RV32_REGISTER_NUM_LIMBS];
    uint32_t imm;
    uint8_t local_opcode;
};

template <typename T> struct BranchLessThanCols {
    Rv32BranchAdapterCols<T> adapter;
    BranchLessThanCoreCols<T> core;
};

struct BranchLessThanRecord {
    Rv32BranchAdapterRecord adapter;
    BranchLessThanCoreRecord core;
};

struct BranchLessThanCoreTracer {
    BitwiseOperationLookup bw;

    __device__ BranchLessThanCoreTracer(uint32_t *bw_ptr, uint32_t bw_bits) : bw(bw_ptr, bw_bits) {}

    __device__ void fill_trace_row(RowSlice row, BranchLessThanCoreRecord rec) {

        int diff_idx = RV32_REGISTER_NUM_LIMBS;

        bool signed_op = (rec.local_opcode == BLT) || (rec.local_opcode == BGE);
        bool ge_op = (rec.local_opcode == BGE) || (rec.local_opcode == BGEU);
        bool cmp_result = ge_op;
        bool a_sign =
            ((rec.a[RV32_REGISTER_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1)) & 1u) && signed_op;
        bool b_sign =
            ((rec.b[RV32_REGISTER_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1)) & 1u) && signed_op;

        for (int i = (int)RV32_REGISTER_NUM_LIMBS - 1; i >= 0; --i) {
            if (rec.a[i] != rec.b[i]) {
                cmp_result = (rec.a[i] < rec.b[i]) ^ a_sign ^ b_sign ^ ge_op;
                diff_idx = i;
                break;
            }
        }
        if (diff_idx == RV32_REGISTER_NUM_LIMBS) {
            cmp_result = ge_op;
        }
        bool cmp_lt = cmp_result ^ ge_op;
        uint8_t msb_a = rec.a[RV32_REGISTER_NUM_LIMBS - 1];
        uint8_t msb_b = rec.b[RV32_REGISTER_NUM_LIMBS - 1];
        uint32_t a_msb_range = signed_op ? (uint32_t(msb_a) - (1u << (RV32_CELL_BITS - 1)))
                                         : (uint32_t(msb_a) + (signed_op << (RV32_CELL_BITS - 1)));
        uint32_t b_msb_range = signed_op ? (uint32_t(msb_b) - (1u << (RV32_CELL_BITS - 1)))
                                         : (uint32_t(msb_b) + (signed_op << (RV32_CELL_BITS - 1)));

        Fp a_msb_f, b_msb_f;
        if (a_sign) {
            a_msb_f = Fp::zero() - Fp((1u << RV32_CELL_BITS) - msb_a);
        } else {
            a_msb_f = Fp(msb_a);
        }
        if (b_sign) {
            b_msb_f = Fp::zero() - Fp((1u << RV32_CELL_BITS) - msb_b);
        } else {
            b_msb_f = Fp(msb_b);
        }

        bw.add_range(a_msb_range, b_msb_range);

        Fp diff_val;
        if (diff_idx == RV32_REGISTER_NUM_LIMBS) {
            diff_val = Fp::zero();
        } else if (diff_idx == RV32_REGISTER_NUM_LIMBS - 1) {
            diff_val = cmp_lt ? (b_msb_f - a_msb_f) : (a_msb_f - b_msb_f);
        } else {
            diff_val = cmp_lt ? Fp(rec.b[diff_idx] - rec.a[diff_idx])
                              : Fp(rec.a[diff_idx] - rec.b[diff_idx]);
        }

        if (diff_idx < RV32_REGISTER_NUM_LIMBS) {
            bw.add_range(diff_val.asUInt32() - 1, 0);
        }

        COL_FILL_ZERO(row, BranchLessThanCoreCols, diff_marker);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, diff_marker[diff_idx], Fp::one());

        COL_WRITE_VALUE(row, BranchLessThanCoreCols, diff_val, diff_val);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, cmp_lt, cmp_lt);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, b_msb_f, b_msb_f);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, a_msb_f, a_msb_f);

        COL_WRITE_VALUE(row, BranchLessThanCoreCols, opcode_bgeu_flag, rec.local_opcode == BGEU);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, opcode_bge_flag, rec.local_opcode == BGE);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, opcode_bltu_flag, rec.local_opcode == BLTU);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, opcode_blt_flag, rec.local_opcode == BLT);

        COL_WRITE_VALUE(row, BranchLessThanCoreCols, imm, rec.imm);
        COL_WRITE_VALUE(row, BranchLessThanCoreCols, cmp_result, cmp_result);

        COL_WRITE_ARRAY(row, BranchLessThanCoreCols, b, rec.b);
        COL_WRITE_ARRAY(row, BranchLessThanCoreCols, a, rec.a);
    }
};

__global__ void blt_tracegen(
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
        auto full = reinterpret_cast<BranchLessThanRecord *>(records)[idx];

        Rv32BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins));
        adapter.fill_trace_row(row, full.adapter);

        BranchLessThanCoreTracer core(bw_ptr, bw_bits);
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThanCols, core)), full.core);
    } else {
        constexpr size_t W = sizeof(BranchLessThanCols<uint8_t>);
#pragma unroll
        for (size_t c = 0; c < W; ++c) {
            row.write(c, 0);
        }
    }
}

extern "C" int _blt_tracegen(
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
    assert(height * sizeof(BranchLessThanRecord) >= record_len);
    assert(width == sizeof(BranchLessThanCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);
    blt_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        record_len / sizeof(BranchLessThanRecord),
        d_rc,
        rc_bins,
        d_bw,
        bw_bits
    );
    return cudaGetLastError();
}
