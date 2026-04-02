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

template <size_t NUM_LIMBS> struct ModularMulDivCoreRecord {
    uint8_t opcode;
    uint8_t input_limbs[2 * NUM_LIMBS];
};

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS> struct ModularMulDivRecord {
    Rv32VecHeapAdapterRecord<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter;
    ModularMulDivCoreRecord<NUM_LIMBS> core;
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

__device__ inline int64_t biguint_product_limb(
    const BigUintGpu &lhs,
    const BigUintGpu &rhs,
    uint32_t limb
) {
    int64_t coeff = 0;
    for (uint32_t i = 0; i < lhs.num_limbs && i <= limb; i++) {
        uint32_t j = limb - i;
        if (j < rhs.num_limbs) {
            coeff += (int64_t)lhs.limbs[i] * (int64_t)rhs.limbs[j];
        }
    }
    return coeff;
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

__device__ inline int32_t read_signed_trace_limb(RowSlice row, size_t col) {
    uint32_t raw = row[col].asUInt32();
    if (raw <= 255) {
        return (int32_t)raw;
    }
    return -(int32_t)(Fp::P - raw);
}

__device__ inline void read_signed_bigint_limbs_from_trace(
    RowSlice row,
    size_t col,
    uint32_t limb_count,
    int32_t out_limbs[MAX_LIMBS]
) {
    for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
        out_limbs[limb] = limb < limb_count ? read_signed_trace_limb(row, col + limb) : 0;
    }
}

__device__ inline OverflowInt make_signed_overflow(
    const int32_t *signed_limbs,
    uint32_t count,
    uint32_t bits
) {
    OverflowInt value(bits);
    value.num_limbs = count;
    value.limb_bits = bits;
    for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
        value.limbs[limb] = limb < count ? signed_limbs[limb] : 0;
    }
    value.limb_max_abs = 1u << bits;
    value.max_overflow_bits = bits + 1;
    return value;
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
__global__ void modular_muldiv_adapter_tracegen_kernel(
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
        reinterpret_cast<const ModularMulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    Rv32VecHeapAdapter<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter(
        pointer_max_bits, range_checker, bitwise_lookup, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record->adapter);
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void modular_muldiv_compute_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *barrett_mu,
    uint32_t mul_opcode,
    uint32_t div_opcode,
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
        reinterpret_cast<const ModularMulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    const bool is_mul = record->core.opcode == mul_opcode;
    const bool is_div = record->core.opcode == div_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x.normalize();
    y.normalize();
    x = x.rem(prime);
    y = y.rem(prime);

    BigUintGpu z = x;
    if (is_mul) {
        z = x.mul(y).mod_reduce(prime, barrett_mu);
    } else if (is_div) {
        z = x.mod_div(y, prime, barrett_mu);
    }

    size_t col = 0;
    core_row[col++] = Fp::one();
    for (size_t i = 0; i < 2 * NUM_LIMBS; i++) {
        core_row[col++] = Fp(input_limbs[i]);
    }

    write_biguint<NUM_LIMBS>(core_row, col, z);
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        range_checker.add_count(core_row[1 + 2 * NUM_LIMBS + i].asUInt32(), 8);
    }
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void modular_muldiv_quotient_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    size_t q_col,
    uint32_t q_limbs,
    uint32_t mul_opcode,
    uint32_t div_opcode,
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
        return;
    }

    const auto *record =
        reinterpret_cast<const ModularMulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    const bool is_mul = record->core.opcode == mul_opcode;
    const bool is_div = record->core.opcode == div_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x.normalize();
    y.normalize();
    const size_t vars_col = 1 + 2 * NUM_LIMBS;
    BigUintGpu z = read_biguint<NUM_LIMBS>(core_row, vars_col);

    BigIntGpu lvar_big(is_mul ? x : z);
    BigIntGpu tmp_big(y);
    lvar_big *= tmp_big;
    tmp_big = BigIntGpu(is_mul ? z : x);
    lvar_big -= tmp_big;

    BigIntGpu quotient = lvar_big.div_biguint(prime);
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
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void modular_muldiv_carry_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    size_t q_col,
    size_t carry_col,
    uint32_t q_limbs,
    uint32_t carry_limbs,
    uint32_t mul_opcode,
    uint32_t div_opcode,
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
        return;
    }

    const auto *record =
        reinterpret_cast<const ModularMulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
            records + idx * record_stride
        );
    const bool is_mul = record->core.opcode == mul_opcode;
    const bool is_div = record->core.opcode == div_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    OverflowInt prime_overflow(prime, prime.num_limbs);

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x.normalize();
    y.normalize();
    const size_t vars_col = 1 + 2 * NUM_LIMBS;
    BigUintGpu z = read_biguint<NUM_LIMBS>(core_row, vars_col);

    OverflowInt lvar_ov(is_mul ? x : z, NUM_LIMBS);
    OverflowInt tmp_ov(y, NUM_LIMBS);
    lvar_ov *= tmp_ov;
    tmp_ov = OverflowInt(is_mul ? z : x, NUM_LIMBS);
    lvar_ov -= tmp_ov;

    int32_t quotient_signed_limbs[MAX_LIMBS];
    read_signed_bigint_limbs_from_trace(core_row, q_col, q_limbs, quotient_signed_limbs);

    OverflowInt result = lvar_ov;
    tmp_ov = make_signed_overflow(quotient_signed_limbs, q_limbs, 8) * prime_overflow;
    result -= tmp_ov;

    uint32_t carry_bits = result.max_overflow_bits - 8;
    uint32_t carry_min_abs = 1u << carry_bits;
    carry_bits++;
    int64_t carry = 0;
    for (uint32_t limb = 0; limb < carry_limbs; limb++) {
        int64_t coeff = is_mul ? biguint_product_limb(x, y, limb) : biguint_product_limb(z, y, limb);
        coeff -= is_mul ? biguint_limb_or_zero(z, limb) : biguint_limb_or_zero(x, limb);
        coeff += subtract_signed_limb_product_limb(
            quotient_signed_limbs, q_limbs, prime_limbs, prime.num_limbs, limb
        );
        carry = (carry + coeff) >> 8;
        write_carry_limb(core_row, carry_col, range_checker, carry, carry_min_abs, carry_bits);
    }

    core_row[carry_col++] = is_mul ? Fp::one() : Fp::zero();
    core_row[carry_col++] = is_div ? Fp::one() : Fp::zero();
}

extern "C" int launch_modular_muldiv_tracegen(
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
    uint32_t mul_opcode,
    uint32_t div_opcode,
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
        const size_t q_col = 1 + 3 * NUM_LIMBS;
        const size_t carry_col = q_col + q_limbs;

        modular_muldiv_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
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

        modular_muldiv_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
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
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_muldiv_quotient_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q_col,
                q_limbs,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_muldiv_carry_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q_col,
                carry_col,
                q_limbs,
                carry_limbs,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    } else {
        constexpr size_t BLOCKS = 12;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 48;
        const size_t core_width = 1 + 2 * NUM_LIMBS + NUM_LIMBS + q_limbs + carry_limbs + 2;
        const size_t q_col = 1 + 3 * NUM_LIMBS;
        const size_t carry_col = q_col + q_limbs;

        modular_muldiv_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
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

        modular_muldiv_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
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
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_muldiv_quotient_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q_col,
                q_limbs,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        modular_muldiv_carry_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q_col,
                carry_col,
                q_limbs,
                carry_limbs,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    }

    return CHECK_KERNEL();
}
