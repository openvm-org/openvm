#include "adapters/alu.cuh"
#include "constants.h"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;
using namespace program;

template <typename T> struct LessThanCoreCols {
    T b[RV32_REGISTER_NUM_LIMBS];
    T c[RV32_REGISTER_NUM_LIMBS];
    T cmp_result;
    
    T opcode_slt_flag;
    T opcode_sltu_flag;
    
    T b_msb_f;
    T c_msb_f;
    
    // 1 at the most significant index i such that b[i] != c[i], otherwise 0
    T diff_marker[RV32_REGISTER_NUM_LIMBS];
    T diff_val;
};

struct LessThanCoreRecord {
    uint8_t b[RV32_REGISTER_NUM_LIMBS];
    uint8_t c[RV32_REGISTER_NUM_LIMBS];
    uint8_t local_opcode;
};

constexpr uint8_t SLT = 0;

struct LessThanResult {
    bool cmp_result;
    size_t diff_idx;
    bool x_sign;
    bool y_sign;
};

// Returns (cmp_result, diff_idx, x_sign, y_sign)
__device__ LessThanResult run_less_than(
    bool is_slt,
    const uint8_t x[RV32_REGISTER_NUM_LIMBS],
    const uint8_t y[RV32_REGISTER_NUM_LIMBS]
) {
    bool x_sign = (x[RV32_REGISTER_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1) && is_slt;
    bool y_sign = (y[RV32_REGISTER_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1) && is_slt;
    
    for (int i = RV32_REGISTER_NUM_LIMBS - 1; i >= 0; i--) {
        if (x[i] != y[i]) {
            return {bool((x[i] < y[i]) ^ x_sign ^ y_sign), (size_t)i, x_sign, y_sign};
        }
    }
    return {false, RV32_REGISTER_NUM_LIMBS, x_sign, y_sign};
}

struct LessThanCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ LessThanCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, LessThanCoreRecord record) {
        bool is_slt = record.local_opcode == SLT;
        LessThanResult result = run_less_than(is_slt, record.b, record.c);
        bool cmp_result = result.cmp_result;
        size_t diff_idx = result.diff_idx;
        bool b_sign = result.x_sign;
        bool c_sign = result.y_sign;

        // Represent the MSB as a field element:
        //   signed  : -((1 << 8) - byte)  ->  P - ((1 << 8) - byte)
        //   unsigned: byte
        uint8_t b_raw_msb = record.b[RV32_REGISTER_NUM_LIMBS - 1];
        uint8_t c_raw_msb = record.c[RV32_REGISTER_NUM_LIMBS - 1];

        uint32_t b_msb_f = b_sign
            ? (Fp::P - ((1u << RV32_CELL_BITS) - b_raw_msb))
            : uint32_t(b_raw_msb);

        uint32_t c_msb_f = c_sign
            ? (Fp::P - ((1u << RV32_CELL_BITS) - c_raw_msb))
            : uint32_t(c_raw_msb);

        // Values used for range-checking (always 8-bit unsigned)
        uint8_t b_msb_range = b_sign
            ? uint8_t(record.b[RV32_REGISTER_NUM_LIMBS - 1] - (1u << (RV32_CELL_BITS - 1)))
            : uint8_t(record.b[RV32_REGISTER_NUM_LIMBS - 1] + ((is_slt ? 1u : 0u) << (RV32_CELL_BITS - 1)));

        uint8_t c_msb_range = c_sign
            ? uint8_t(record.c[RV32_REGISTER_NUM_LIMBS - 1] - (1u << (RV32_CELL_BITS - 1)))
            : uint8_t(record.c[RV32_REGISTER_NUM_LIMBS - 1] + ((is_slt ? 1u : 0u) << (RV32_CELL_BITS - 1)));

        uint32_t diff_val = 0;
        if (diff_idx == RV32_REGISTER_NUM_LIMBS) {
            diff_val = 0;
        } else if (diff_idx == (RV32_REGISTER_NUM_LIMBS - 1)) {
            // MSB comparison uses the field-element encoded values
            diff_val = cmp_result
                ? ((c_msb_f >= b_msb_f) ? (c_msb_f - b_msb_f) : (c_msb_f + Fp::P - b_msb_f))
                : ((b_msb_f >= c_msb_f) ? (b_msb_f - c_msb_f) : (b_msb_f + Fp::P - c_msb_f));
        } else if (cmp_result) {
            diff_val = uint32_t(record.c[diff_idx] - record.b[diff_idx]);
        } else {
            diff_val = uint32_t(record.b[diff_idx] - record.c[diff_idx]);
        }

        bitwise_lookup.add_range(b_msb_range, c_msb_range);
        
        uint8_t diff_marker[RV32_REGISTER_NUM_LIMBS] = {0};
        if (diff_idx != RV32_REGISTER_NUM_LIMBS) {
            bitwise_lookup.add_range(diff_val - 1, 0);
            diff_marker[diff_idx] = 1;
        }

        COL_WRITE_ARRAY(row, LessThanCoreCols, b, record.b);
        COL_WRITE_ARRAY(row, LessThanCoreCols, c, record.c);
        COL_WRITE_VALUE(row, LessThanCoreCols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, LessThanCoreCols, opcode_slt_flag, is_slt);
        COL_WRITE_VALUE(row, LessThanCoreCols, opcode_sltu_flag, !is_slt);
        COL_WRITE_VALUE(row, LessThanCoreCols, b_msb_f, b_msb_f);
        COL_WRITE_VALUE(row, LessThanCoreCols, c_msb_f, c_msb_f);
        COL_WRITE_ARRAY(row, LessThanCoreCols, diff_marker, diff_marker);
        COL_WRITE_VALUE(row, LessThanCoreCols, diff_val, diff_val);
    }
};

template <typename T> struct LessThanCols {
    Rv32BaseAluAdapterCols<T> adapter;
    LessThanCoreCols<T> core;
};

struct LessThanRecord {
    Rv32BaseAluAdapterRecord adapter;
    LessThanCoreRecord core;
};

__global__ void rv32_less_than_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        auto record = reinterpret_cast<LessThanRecord *>(records)[idx];

        auto adapter = Rv32BaseAluAdapter(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        adapter.fill_trace_row(row, record.adapter);

        auto core = LessThanCore(BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThanCols, core)), record.core);
    } else {
#pragma unroll
        for (size_t i = 0; i < sizeof(LessThanCols<uint8_t>); i++) {
            row.write(i, 0);
        }
    }
}

extern "C" int _rv32_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits
) {
    // We require the height to be a power of two for the tracegen to work
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(LessThanRecord) >= record_len);
    assert(width == sizeof(LessThanCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    rv32_less_than_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        record_len / sizeof(LessThanRecord),
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits
    );
    return cudaGetLastError();
} 