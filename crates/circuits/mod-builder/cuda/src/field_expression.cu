#include "fp.h"
#include "launcher.cuh"
#include "mod-builder/bigint_ops.cuh"
#include "mod-builder/expr_codec.cuh"
#include "mod-builder/meta.cuh"
#include "mod-builder/overflow_ops.cuh"
#include "mod-builder/records.cuh"
#include "mod-builder/rv32_vec_heap_router.cuh"
#include "primitives/trace_access.h"
#include <cstdint>

using namespace mod_builder;

#define INPUT_U32_COUNT(meta) ((meta)->num_inputs * (meta)->num_limbs)
#define VAR_U32_COUNT(meta) ((meta)->expr_meta.num_vars * (meta)->num_limbs)
#define FLAG_U32_COUNT(meta) (((meta)->num_u32_flags + 3) / 4)
#define THREAD_U32_COUNT(meta) (INPUT_U32_COUNT(meta) + VAR_U32_COUNT(meta) + FLAG_U32_COUNT(meta))

__device__ inline uint32_t get_total_carry_count(const FieldExprMeta *meta) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
        total += meta->carry_limb_counts[i];
    }
    return total;
}

__device__ void generate_subrow_gpu(
    const FieldExprMeta *meta,
    const uint32_t *inputs,
    const bool *flags,
    uint32_t *vars,
    uint32_t *all_carries,
    VariableRangeChecker &range_checker,
    bool is_valid,
    RowSlice core_row
) {
    uint32_t num_limbs = meta->num_limbs;
    uint32_t limb_bits = meta->limb_bits;

    BigUintGpu prime(
        meta->expr_meta.prime_limbs, meta->expr_meta.prime_limb_count, meta->limb_bits
    );
    prime.normalize();
    OverflowInt prime_overflow(prime, prime.num_limbs);

    for (uint32_t var = 0; var < meta->expr_meta.num_vars; var++) {
        uint32_t root = meta->compute_root_indices[var];
        uint32_t *result = &vars[var * num_limbs];

        compute(
            result,
            meta->compute_expr_ops,
            root,
            &meta->expr_meta,
            inputs,
            vars,
            flags,
            num_limbs,
            limb_bits,
            prime
        );
    }

    uint32_t col = 0;

    core_row[col++] = is_valid;

    for (uint32_t i = 0; i < meta->num_inputs; i++) {
        for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
            core_row[col++] = Fp(inputs[i * meta->num_limbs + limb]);
        }
    }

    for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
        for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
            core_row[col++] = Fp(vars[i * meta->num_limbs + limb]);
        }
    }
    memset(all_carries, 0, get_total_carry_count(meta) * sizeof(uint32_t));
    uint32_t c_offset = 0;

    if (meta->expr_meta.num_vars > 0) {
        for (uint32_t var_idx = 0; var_idx < meta->expr_meta.num_vars; var_idx++) {
            uint32_t constraint_root = meta->constraint_root_indices[var_idx];

            BigIntGpu constraint_result = evaluate_bigint(
                meta->constraint_expr_ops,
                constraint_root,
                &meta->expr_meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits
            );

            BigIntGpu quotient = constraint_result.div_biguint(prime);

            uint32_t q_count = meta->q_limb_counts[var_idx];

            if (is_valid) {
                for (uint32_t i = 0; i < q_count; i++) {
                    int32_t q_signed = (int32_t)quotient.mag.limbs[i];
                    if (quotient.is_negative) {
                        q_signed = -q_signed;
                    }
                    range_checker.add_count(q_signed + (1 << limb_bits), limb_bits + 1);
                }
            }

            OverflowInt expr = evaluate_overflow_int(
                meta->constraint_expr_ops,
                constraint_root,
                &meta->expr_meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits
            );

            // result = expr - q * p
            OverflowInt result = expr - (OverflowInt(quotient, q_count) * prime_overflow);

            uint32_t c_count = meta->carry_limb_counts[var_idx];

            OverflowInt carries = result.carry_limbs(c_count);
            for (uint32_t i = 0; i < c_count; i++) {
                all_carries[c_offset + i] = carries.limbs[i];
            }

            for (uint32_t limb = 0; limb < q_count; limb++) {
                uint32_t q_limb = quotient.mag.limbs[limb];

                if (!quotient.is_negative) {
                    core_row[col] = Fp(q_limb);
                } else {
                    core_row[col] = Fp::zero() - Fp(q_limb);
                }

                col++;
            }

            uint32_t max_overflow_bits = result.max_overflow_bits;
            uint32_t carry_bits = max_overflow_bits - limb_bits;
            uint32_t carry_min_abs = 1 << carry_bits;
            carry_bits++;

            if (is_valid) {
                for (uint32_t i = 0; i < c_count; i++) {
                    range_checker.add_count(all_carries[c_offset + i] + carry_min_abs, carry_bits);
                }
            }

            c_offset += c_count;
        }

        c_offset = 0;
        for (uint32_t var_idx = 0; var_idx < meta->expr_meta.num_vars; var_idx++) {
            uint32_t c_count = meta->carry_limb_counts[var_idx];

            for (uint32_t limb = 0; limb < c_count; limb++) {
                int32_t signed_carry = (int32_t)all_carries[c_offset + limb];

                if (signed_carry >= 0) {
                    core_row[col] = Fp((uint32_t)signed_carry);
                } else {
                    core_row[col] = Fp::zero() - Fp((uint32_t)(-signed_carry));
                }
                col++;
            }
            c_offset += c_count;
        }
    } else {
        for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
            uint32_t q_count = meta->q_limb_counts[i];
            for (uint32_t limb = 0; limb < q_count; limb++) {
                core_row[col] = Fp::zero();
                col++;
            }
        }
        for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
            uint32_t c_count = meta->carry_limb_counts[i];
            for (uint32_t limb = 0; limb < c_count; limb++) {
                core_row[col] = Fp::zero();
                col++;
            }
        }
    }

    for (uint32_t i = 0; i < meta->num_u32_flags; i++) {
        core_row[col++] = Fp(flags[i]);
    }

    while (col < meta->core_width) {
        core_row[col++] = Fp::zero();
    }
}

struct FieldExprCore {
    // If true, we shouldn't do any range checks
    bool is_valid;
    const FieldExprMeta *meta;
    VariableRangeChecker range_checker;
    uint8_t *workspace;

    __device__ explicit FieldExprCore(
        const FieldExprMeta *m,
        VariableRangeChecker rc,
        bool is_valid,
        uint8_t *ws
    )
        : meta(m), range_checker(rc), is_valid(is_valid), workspace(ws) {}

    __device__ void fill_trace_row(RowSlice core_row, const FieldExprCoreRecord *core_rec) {
        const uint8_t *rec_bytes = core_rec->input_limbs;
        uint8_t opcode = core_rec->opcode;

        uint32_t in_size = INPUT_U32_COUNT(meta);
        uint32_t var_size = VAR_U32_COUNT(meta);
        uint32_t carry_size = get_total_carry_count(meta);

        uint32_t *inputs = (uint32_t *)workspace;
        uint32_t *vars = (uint32_t *)(workspace + in_size * sizeof(uint32_t));
        uint32_t *all_carries = (uint32_t *)(workspace + (in_size + var_size) * sizeof(uint32_t));
        bool *flags = (bool *)(workspace + (in_size + var_size + carry_size) * sizeof(uint32_t));

        size_t total_size = (in_size + var_size + carry_size) * sizeof(uint32_t) +
                            meta->num_u32_flags * sizeof(bool);

        memset(workspace, 0, total_size);

        uint32_t bytes_per_limb = (meta->limb_bits + 7) / 8;

        for (uint32_t i = 0; i < meta->num_inputs; i++) {
            for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
                size_t base = (size_t(i) * meta->num_limbs + limb) * bytes_per_limb;
                uint32_t v = 0;
                for (uint32_t b = 0; b < bytes_per_limb; b++) {
                    v |= (uint32_t)rec_bytes[base + b] << (8 * b);
                }
                inputs[i * meta->num_limbs + limb] = v;
            }
        }

        // flags for needs setup. These will all be false if opcode == SETUP
        // or opcode == 0xFF (the dummy opcode for dummy fill trace row)
        for (uint32_t j = 0; j < meta->num_local_opcodes; j++) {
            if (opcode == meta->local_opcode_idx[j] && j < meta->num_u32_flags) {
                flags[meta->opcode_flag_idx[j]] = true;
            }
        }

        generate_subrow_gpu(
            meta, inputs, flags, vars, all_carries, range_checker, is_valid, core_row
        );

        if (is_valid) {
            for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
                for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
                    uint32_t var_val = vars[i * meta->num_limbs + limb];
                    range_checker.add_count(var_val, meta->limb_bits);
                }
            }
        }
    }
};

__global__ void field_expression_tracegen(
    const uint8_t *records,
    Fp *trace,
    const FieldExprMeta *meta,
    size_t num_records,
    size_t record_stride,
    size_t width,
    size_t height,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    size_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint8_t *workspace,
    uint32_t workspace_per_thread
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height)
        return;

    RowSlice row(trace + idx, height);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);

    uint8_t *thread_workspace = workspace + idx * workspace_per_thread;

    // Ensure workspace is aligned to 4 bytes for uint32_t access
    assert(((uintptr_t)thread_workspace & 3) == 0);

    if (idx < num_records) {
        const uint8_t *rec_bytes = records + idx * record_stride;

        size_t adapter_size = 0;
        route_rv32_vec_heap_adapter(
            row,
            rec_bytes,
            meta,
            pointer_max_bits,
            range_checker,
            bitwise_lookup,
            timestamp_max_bits,
            adapter_size
        );

        const uint8_t *core_bytes = rec_bytes + adapter_size;
        const FieldExprCoreRecord *core_rec =
            reinterpret_cast<const FieldExprCoreRecord *>(core_bytes);

        FieldExprCore core(meta, range_checker, true, thread_workspace);
        core.fill_trace_row(row.slice_from(meta->adapter_width), core_rec);
    } else {
        // We can't just fill with 0s, instead calling w/ invalid opcode
        row.fill_zero(0, meta->adapter_width);

        FieldExprCore dummy_core(meta, range_checker, false, thread_workspace);

        uint8_t *dummy_record = thread_workspace;
        memset(dummy_record, 0, workspace_per_thread);
        dummy_record[0] = 0xFF; // Invalid opcode

        const FieldExprCoreRecord *dummy_core_record =
            reinterpret_cast<const FieldExprCoreRecord *>(dummy_record);

        dummy_core.fill_trace_row(row.slice_from(meta->adapter_width), dummy_core_record);
    }
}

extern "C" int _field_expression_tracegen(
    const uint8_t *d_records,
    Fp *d_trace,
    const FieldExprMeta *d_meta,
    size_t num_records,
    size_t record_stride,
    size_t width,
    size_t height,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint8_t *d_workspace,
    uint32_t workspace_per_thread
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height, 256);
    field_expression_tracegen<<<grid, block>>>(
        (uint8_t *)d_records,
        d_trace,
        d_meta,
        num_records,
        record_stride,
        width,
        height,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        pointer_max_bits,
        timestamp_max_bits,
        d_workspace,
        workspace_per_thread
    );

    return cudaGetLastError();
}