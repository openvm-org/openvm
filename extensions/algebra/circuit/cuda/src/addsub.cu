#include "fp.h"
#include "launcher.cuh"
#include "mod-builder/bigint_ops.cuh"
#include "mod-builder/overflow_ops.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32-adapters/vec_heap.cuh"
#include <cstddef>
#include <cstdint>

using namespace mod_builder;
using namespace riscv;

template <size_t NUM_LIMBS> struct ModularAddSubCoreRecord {
    uint8_t opcode;
    uint8_t input_limbs[2 * NUM_LIMBS];
};

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS> struct ModularAddSubRecord {
    Rv32VecHeapAdapterRecord<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter;
    ModularAddSubCoreRecord<NUM_LIMBS> core;
};

template <size_t NUM_LIMBS>
__device__ inline void write_biguint(RowSlice row, size_t &col, const BigUintGpu &value) {
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        uint32_t v = limb < value.num_limbs ? value.limbs[limb] : 0;
        row[col++] = Fp(v);
    }
}

template <size_t NUM_LIMBS> __device__ inline BigUintGpu read_biguint(RowSlice row, size_t col) {
    uint32_t limbs[NUM_LIMBS];
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        limbs[limb] = row[col + limb].asUInt32();
    }
    BigUintGpu value(limbs, NUM_LIMBS, 8);
    value.normalize();
    return value;
}

__device__ inline void write_signed_bigint_limb(
    RowSlice row,
    size_t &col,
    const BigIntGpu &value,
    size_t limb
) {
    uint32_t q_limb = limb < value.mag.num_limbs ? value.mag.limbs[limb] : 0;
    row[col++] = value.is_negative ? (Fp::zero() - Fp(q_limb)) : Fp(q_limb);
}

__device__ inline int64_t biguint_limb_or_zero(const BigUintGpu &value, uint32_t limb) {
    return limb < value.num_limbs ? (int64_t)value.limbs[limb] : 0;
}

__device__ inline void read_signed_bigint_limbs(
    const BigIntGpu &value,
    int32_t out_limbs[MAX_LIMBS]
) {
    for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
        int32_t signed_limb = limb < value.mag.num_limbs ? (int32_t)value.mag.limbs[limb] : 0;
        out_limbs[limb] = value.is_negative ? -signed_limb : signed_limb;
    }
}

__device__ inline int64_t subtract_signed_limb_product_limb(
    const int32_t *lhs_limbs,
    uint32_t lhs_limb_count,
    const uint8_t *rhs_limbs,
    uint32_t rhs_limb_count,
    uint32_t limb
) {
    int64_t coeff = 0;
    for (uint32_t lhs_limb = 0; lhs_limb < lhs_limb_count && lhs_limb <= limb; lhs_limb++) {
        uint32_t rhs_limb = limb - lhs_limb;
        if (rhs_limb < rhs_limb_count) {
            coeff -= (int64_t)lhs_limbs[lhs_limb] * (int64_t)rhs_limbs[rhs_limb];
        }
    }
    return coeff;
}

__device__ inline void write_carry_limb(
    RowSlice core_row,
    size_t &col,
    VariableRangeChecker &range_checker,
    int64_t carry,
    uint32_t carry_min_abs,
    uint32_t carry_bits
) {
    int32_t carry_i32 = (int32_t)carry;
    core_row[col++] =
        carry_i32 >= 0 ? Fp((uint32_t)carry_i32) : (Fp::zero() - Fp((uint32_t)(-carry_i32)));
    range_checker.add_count((uint32_t)(carry_i32 + (int32_t)carry_min_abs), carry_bits);
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void modular_addsub_adapter_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    RowSlice row(trace + idx, height);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    if (idx >= num_records) {
        row.fill_zero(0, ADAPTER_WIDTH);
        return;
    }

    const auto *record =
        reinterpret_cast<const ModularAddSubRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    Rv32VecHeapAdapter<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter(
        pointer_max_bits, range_checker, bitwise_lookup, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record->adapter);
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void modular_addsub_compute_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *barrett_mu,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);

    if (idx >= num_records) {
        core_row.fill_zero(0, core_width);
        return;
    }

    const auto *record =
        reinterpret_cast<const ModularAddSubRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    const bool is_add = record->core.opcode == add_opcode;
    const bool is_sub = record->core.opcode == sub_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu x2(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    x2.normalize();
    x1 = x1.rem(prime);
    x2 = x2.rem(prime);

    BigUintGpu x6 = x1;
    if (is_add) {
        x6.add_in_place(x2);
        x6 = x6.mod_reduce(prime, barrett_mu);
    } else if (is_sub) {
        x6 = x1.mod_sub(x2, prime);
    }

    size_t col = 0;
    core_row[col++] = Fp::one();
    for (size_t i = 0; i < 2 * NUM_LIMBS; i++) {
        core_row[col++] = Fp(input_limbs[i]);
    }

    write_biguint<NUM_LIMBS>(core_row, col, x6);
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        range_checker.add_count(core_row[1 + 2 * NUM_LIMBS + i].asUInt32(), 8);
    }
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void modular_addsub_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    uint32_t q_limbs,
    uint32_t carry_limbs,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);

    if (idx >= num_records) {
        core_row.fill_zero(0, core_width);
        return;
    }

    const auto *record =
        reinterpret_cast<const ModularAddSubRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    const bool is_add = record->core.opcode == add_opcode;
    const bool is_sub = record->core.opcode == sub_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    OverflowInt prime_overflow(prime, prime.num_limbs);

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu x2(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    x2.normalize();
    const size_t vars_col = 1 + 2 * NUM_LIMBS;
    BigUintGpu x6 = read_biguint<NUM_LIMBS>(core_row, vars_col);

    size_t q_col = vars_col + NUM_LIMBS;
    size_t carry_col = q_col + q_limbs;

    BigIntGpu constraint_big(x1);
    BigIntGpu tmp_big(x6);
    OverflowInt constraint_ov(x1, NUM_LIMBS);
    OverflowInt tmp_ov(x6, NUM_LIMBS);
    if (is_add) {
        BigIntGpu x2_big(x2);
        constraint_big += x2_big;
        constraint_big -= tmp_big;
        OverflowInt x2_ov(x2, NUM_LIMBS);
        constraint_ov += x2_ov;
        constraint_ov -= tmp_ov;
    } else if (is_sub) {
        BigIntGpu x2_big(x2);
        constraint_big -= x2_big;
        constraint_big -= tmp_big;
        OverflowInt x2_ov(x2, NUM_LIMBS);
        constraint_ov -= x2_ov;
        constraint_ov -= tmp_ov;
    } else {
        constraint_big -= tmp_big;
        constraint_ov -= tmp_ov;
    }

    BigIntGpu quotient = constraint_big.div_biguint(prime);
    for (uint32_t limb = 0; limb < q_limbs; limb++) {
        write_signed_bigint_limb(core_row, q_col, quotient, limb);
    }
    for (uint32_t limb = 0; limb < q_limbs; limb++) {
        int32_t q_signed = limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
        if (quotient.is_negative) {
            q_signed = -q_signed;
        }
        range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
    }

    OverflowInt result = constraint_ov;
    tmp_ov = OverflowInt(quotient, q_limbs) * prime_overflow;
    result -= tmp_ov;

    uint32_t carry_bits = result.max_overflow_bits - 8;
    uint32_t carry_min_abs = 1u << carry_bits;
    carry_bits++;
    int32_t quotient_signed_limbs[MAX_LIMBS];
    read_signed_bigint_limbs(quotient, quotient_signed_limbs);
    int64_t carry = 0;
    for (uint32_t limb = 0; limb < carry_limbs; limb++) {
        int64_t coeff = biguint_limb_or_zero(x1, limb) - biguint_limb_or_zero(x6, limb);
        if (is_add) {
            coeff += biguint_limb_or_zero(x2, limb);
        } else if (is_sub) {
            coeff -= biguint_limb_or_zero(x2, limb);
        }
        coeff += subtract_signed_limb_product_limb(
            quotient_signed_limbs, q_limbs, prime_limbs, prime.num_limbs, limb
        );
        carry = (carry + coeff) >> 8;
        write_carry_limb(core_row, carry_col, range_checker, carry, carry_min_abs, carry_bits);
    }

    core_row[carry_col++] = is_add ? Fp::one() : Fp::zero();
    core_row[carry_col++] = is_sub ? Fp::one() : Fp::zero();
    while (carry_col < core_width) {
        core_row[carry_col++] = Fp::zero();
    }
}

extern "C" int launch_modular_addsub_tracegen(
    Fp *d_trace,
    size_t trace_height,
    const uint8_t *d_records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *d_prime,
    uint32_t prime_limb_count,
    const uint8_t *d_barrett_mu,
    uint32_t q_limbs,
    uint32_t carry_limbs,
    uint32_t add_opcode,
    uint32_t sub_opcode,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    auto [main_grid, main_block] = kernel_launch_params(trace_height, 256);
    auto [constraint_grid, constraint_block] = kernel_launch_params(trace_height, 128);

    if (prime_limb_count <= 32) {
        constexpr size_t BLOCKS = 8;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 32;
        const size_t core_width = 1 + 2 * NUM_LIMBS + NUM_LIMBS + q_limbs + carry_limbs + 2;

        modular_addsub_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_range_checker,
                range_checker_num_bins,
                d_bitwise_lookup,
                bitwise_num_bits,
                pointer_max_bits,
                timestamp_max_bits
            );
        int ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_addsub_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_barrett_mu,
                add_opcode,
                sub_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_addsub_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q_limbs,
                carry_limbs,
                add_opcode,
                sub_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    } else {
        constexpr size_t BLOCKS = 12;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 48;
        const size_t core_width = 1 + 2 * NUM_LIMBS + NUM_LIMBS + q_limbs + carry_limbs + 2;

        modular_addsub_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_range_checker,
                range_checker_num_bins,
                d_bitwise_lookup,
                bitwise_num_bits,
                pointer_max_bits,
                timestamp_max_bits
            );
        int ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_addsub_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_barrett_mu,
                add_opcode,
                sub_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_addsub_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q_limbs,
                carry_limbs,
                add_opcode,
                sub_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    }

    return CHECK_KERNEL();
}
