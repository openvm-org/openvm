#include "adapters/alu.cuh"
#include "constants.h"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;
using namespace program;

template <typename T>
struct ShiftCoreCols {
    T a[RV32_REGISTER_NUM_LIMBS];
    T b[RV32_REGISTER_NUM_LIMBS];
    T c[RV32_REGISTER_NUM_LIMBS];

    T opcode_sll_flag;
    T opcode_srl_flag;
    T opcode_sra_flag;

    T bit_multiplier_left;
    T bit_multiplier_right;
    T b_sign;

    T bit_shift_marker[RV32_CELL_BITS];
    T limb_shift_marker[RV32_REGISTER_NUM_LIMBS];

    T bit_shift_carry[RV32_REGISTER_NUM_LIMBS];
};

struct ShiftCoreRecord {
    uint8_t b[RV32_REGISTER_NUM_LIMBS];
    uint8_t c[RV32_REGISTER_NUM_LIMBS];
    uint8_t local_opcode;
}; 

__device__ void get_shift(const uint8_t y[RV32_REGISTER_NUM_LIMBS], size_t &limb_shift, size_t &bit_shift) {
    size_t max_bits = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS;
    size_t shift = y[0] % max_bits;
    limb_shift = shift / RV32_CELL_BITS;
    bit_shift = shift % RV32_CELL_BITS;
}

__device__ void run_shift_left(
    const uint8_t x[RV32_REGISTER_NUM_LIMBS],
    const uint8_t y[RV32_REGISTER_NUM_LIMBS], 
    uint8_t result[RV32_REGISTER_NUM_LIMBS],
    size_t &limb_shift,
    size_t &bit_shift
) {
    get_shift(y, limb_shift, bit_shift);

#pragma unroll
    for (size_t i = 0; i < limb_shift; i++) {
        result[i] = 0;
    }

#pragma unroll
    for (size_t i = limb_shift; i < RV32_REGISTER_NUM_LIMBS; i++) {
        if (i > limb_shift) {
            uint16_t high = (uint16_t)x[i - limb_shift] << bit_shift;
            uint16_t low = (uint16_t)x[i - limb_shift - 1] >> (RV32_CELL_BITS - bit_shift);
            result[i] = (high | low) % (1u << RV32_CELL_BITS);
        } else {
            uint16_t high = (uint16_t)x[i - limb_shift] << bit_shift;
            result[i] = high % (1u << RV32_CELL_BITS);
        }
    }
}

__device__ void run_shift_right(
    const uint8_t x[RV32_REGISTER_NUM_LIMBS],
    const uint8_t y[RV32_REGISTER_NUM_LIMBS],
    uint8_t result[RV32_REGISTER_NUM_LIMBS],
    size_t &limb_shift,
    size_t &bit_shift,
    bool logical
) {
    uint8_t msb = x[RV32_REGISTER_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1);
    uint8_t fill = logical ? 0u : ((1u << RV32_CELL_BITS) - 1u) * msb;
#pragma unroll
    for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
        result[i] = fill;
    }
    get_shift(y, limb_shift, bit_shift);
    size_t limit = RV32_REGISTER_NUM_LIMBS - limb_shift;
#pragma unroll
    for (size_t i = 0; i < limit; i++) {
        uint16_t part1 = (uint16_t)(x[i + limb_shift] >> bit_shift);
        uint16_t part2_val = (i + limb_shift + 1 < RV32_REGISTER_NUM_LIMBS) ? x[i + limb_shift + 1] : fill;
        uint16_t part2 = (uint16_t)part2_val << (RV32_CELL_BITS - bit_shift);
        result[i] = (part1 | part2) % (1u << RV32_CELL_BITS);
    }
}

struct ShiftCore {
    BitwiseOperationLookup lookup;
    VariableRangeChecker range;

    __device__ ShiftCore(BitwiseOperationLookup lookup_, VariableRangeChecker range_)
        : lookup(lookup_), range(range_) {}

    __device__ void fill_trace_row(RowSlice row, ShiftCoreRecord record) {
        bool is_sll = record.local_opcode == 0;
        bool is_srl = record.local_opcode == 1;
        bool is_sra = record.local_opcode == 2;
        
        uint8_t a[RV32_REGISTER_NUM_LIMBS];
        size_t limb_shift = 0, bit_shift = 0;
        if (is_sll) {
            run_shift_left(record.b, record.c, a, limb_shift, bit_shift);
        } else {
            run_shift_right(record.b, record.c, a, limb_shift, bit_shift, is_srl);
        }
        
        // Bitwise-lookup for each pair of result limbs
#pragma unroll
        for (size_t i = 0; i + 1 < RV32_REGISTER_NUM_LIMBS; i += 2) {
            lookup.add_range(a[i], a[i+1]);
        }
        
        // Range-check the overall shift amount
        size_t combined_bits = RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS;
        size_t num_bits_log = 0;
        while ((1u << num_bits_log) < combined_bits) { ++num_bits_log; }
        range.add_count(
            ((uint32_t)record.c[0] - (uint32_t)bit_shift - (uint32_t)(limb_shift * RV32_CELL_BITS)) >> num_bits_log,
            RV32_CELL_BITS - num_bits_log
        );

        uint8_t carry_arr[RV32_REGISTER_NUM_LIMBS];
        if (bit_shift == 0) {
#pragma unroll
            for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                range.add_count(0u, 0u);
                carry_arr[i] = 0u;
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                uint8_t carry = is_sll
                    ? (record.b[i] >> (RV32_CELL_BITS - bit_shift))
                    : (record.b[i] & ((1u << bit_shift) - 1u));
                range.add_count((uint32_t)carry, bit_shift);
                carry_arr[i] = carry;
            }
        }
        COL_WRITE_ARRAY(row, ShiftCoreCols, bit_shift_carry, carry_arr);

        uint8_t limb_marker[RV32_REGISTER_NUM_LIMBS] = {0};
        limb_marker[limb_shift] = 1u;
        COL_WRITE_ARRAY(row, ShiftCoreCols, limb_shift_marker, limb_marker);
        uint8_t bit_marker[RV32_CELL_BITS] = {0};
        bit_marker[bit_shift] = 1u;
        COL_WRITE_ARRAY(row, ShiftCoreCols, bit_shift_marker, bit_marker);


        COL_WRITE_VALUE(row, ShiftCoreCols, b_sign, 
            is_sra ? (record.b[RV32_REGISTER_NUM_LIMBS-1] >> (RV32_CELL_BITS-1)) : 0u);
        COL_WRITE_VALUE(row, ShiftCoreCols, bit_multiplier_left, 
            is_sll ? (1u << bit_shift) : 0u);
        COL_WRITE_VALUE(row, ShiftCoreCols, bit_multiplier_right, 
            is_sll ? 0u : (1u << bit_shift));

        COL_WRITE_VALUE(row, ShiftCoreCols, opcode_sll_flag, is_sll ? 1u : 0u);
        COL_WRITE_VALUE(row, ShiftCoreCols, opcode_srl_flag, is_srl ? 1u : 0u);
        COL_WRITE_VALUE(row, ShiftCoreCols, opcode_sra_flag, is_sra ? 1u : 0u);

        COL_WRITE_ARRAY(row, ShiftCoreCols, b, record.b);
        COL_WRITE_ARRAY(row, ShiftCoreCols, c, record.c);
        COL_WRITE_ARRAY(row, ShiftCoreCols, a, a);
    }
};

template <typename T>
struct ShiftCols {
    Rv32BaseAluAdapterCols<T> adapter;
    ShiftCoreCols<T> core;
};

struct ShiftRecord {
    Rv32BaseAluAdapterRecord adapter;
    ShiftCoreRecord core;
};

__global__ void rv32_shift_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    uint8_t *records,
    size_t num_records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t lookup_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<ShiftRecord *>(records)[idx];
        auto adapter = Rv32BaseAluAdapter(VariableRangeChecker(range_ptr, range_bins));
        adapter.fill_trace_row(
            row,
            rec.adapter
        );
        auto core = ShiftCore(
            BitwiseOperationLookup(lookup_ptr, lookup_bits),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(
            row.slice_from(COL_INDEX(ShiftCols, core)),
            rec.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv32_shift_tracegen(
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
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(ShiftRecord) >= record_len);
    assert(width == sizeof(ShiftCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    rv32_shift_tracegen<<<grid, block>>>(
        d_trace,
        height,
        width,
        d_records,
        record_len / sizeof(ShiftRecord),
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits
    );
    return cudaGetLastError();
} 