#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "trace_access.h"
#include "mod_builder/bigint_ops.cuh"
#include "mod_builder/limb_ops.cuh"
#include "mod_builder/expr_codec.cuh"
#include "mod_builder/meta.cuh"
#include "mod_builder/records.cuh"
#include "mod_builder/rv32_vec_heap_router.cuh"
#include "../extensions/rv32im_adapters/vec_heap.cuh"

using namespace mod_builder;

#define INPUT_U32_COUNT(meta)    ((meta)->num_inputs * (meta)->num_limbs)
#define VAR_U32_COUNT(meta)      ((meta)->expr_meta.num_vars * (meta)->num_limbs)
#define FLAG_U32_COUNT(meta)     (((meta)->num_u32_flags + 3) / 4)
#define THREAD_U32_COUNT(meta)   (INPUT_U32_COUNT(meta) + VAR_U32_COUNT(meta) + FLAG_U32_COUNT(meta))

__device__ void generate_subrow_gpu(
    const FieldExprMeta* meta,
    const uint32_t* inputs,
    const bool* flags,
    const uint32_t* vars,
    RowSlice core_row
) {
    uint32_t col = 0;
    
    // is_valid flag
    core_row[col++] = Fp::one();
    
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
    
    uint32_t num_limbs = meta->num_limbs;
    uint32_t limb_bits = meta->limb_bits;
    int64_t overflow_remainder[2 * MAX_LIMBS];
    
    uint32_t max_carry_count = 0;
    uint32_t total_carry_count = 0;
    uint32_t max_q_count = 0;
    for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
        if (meta->carry_limb_counts[i] > max_carry_count) {
            max_carry_count = meta->carry_limb_counts[i];
        }
        total_carry_count += meta->carry_limb_counts[i];
        if (meta->q_limb_counts[i] > max_q_count) {
            max_q_count = meta->q_limb_counts[i];
        }
    }
    
    const uint32_t constraint_size = 2 * MAX_LIMBS;
    const uint32_t eval_temp_size = 4 * MAX_LIMBS;
    
    const uint32_t total_buffer_size = 
        (MAX_LIMBS + 1) +
        (2 * MAX_LIMBS * meta->expr_meta.num_vars) +
        constraint_size +
        constraint_size +
        constraint_size +
        (2 * MAX_LIMBS) +
        eval_temp_size;
    
    uint32_t* temp_buffer = (uint32_t*)malloc(total_buffer_size * sizeof(uint32_t));
    if (temp_buffer == nullptr) return;
    memset(temp_buffer, 0, total_buffer_size * sizeof(uint32_t));
    
    uint32_t* quotient_buf = temp_buffer;
    uint32_t* all_carries = quotient_buf + (MAX_LIMBS + 1);
    uint32_t* constraint_body = all_carries + (2 * MAX_LIMBS * meta->expr_meta.num_vars);
    uint32_t* qp_raw = constraint_body + constraint_size;
    uint32_t* remainder_raw = qp_raw + constraint_size;
    uint32_t* carry_buf = remainder_raw + constraint_size;
    uint32_t* eval_temp = carry_buf + (2 * MAX_LIMBS);
    
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
            
            uint32_t q_count = meta->q_limb_counts[var_idx];
            
            memset(quotient_buf, 0, q_count * sizeof(uint32_t));

            BigUintGpu prime(meta->expr_meta.prime_limbs, meta->expr_meta.prime_limb_count, limb_bits);
            
            BigIntGpu quotient;
            bigint_div_biguint(&quotient, &constraint_result, &prime);
            
            int64_t q_signed[MAX_LIMBS];
            bigint_to_signed_limbs(&quotient, q_signed);
            
            for (uint32_t i = 0; i < q_count; i++) {
                quotient_buf[i] = (i < quotient.mag.num_limbs) ? quotient.mag.limbs[i] : 0;
            }
            
            evaluate_overflow(
                constraint_body,
                meta->constraint_expr_ops,
                constraint_root,
                &meta->expr_meta,
                inputs,
                vars,
                flags,
                num_limbs,
                limb_bits,
                eval_temp
            );
            
            for (uint32_t i = 0; i < 2 * MAX_LIMBS; i++) {
                overflow_remainder[i] = (i < 2 * num_limbs) ? (int32_t)constraint_body[i] : 0;
            }
            
            // Subtract q * p in overflow representation
            for (uint32_t i = 0; i < quotient.mag.num_limbs; i++) {
                for (uint32_t j = 0; j < num_limbs; j++) {
                    if (i + j < 2 * num_limbs) {
                        int64_t prod = q_signed[i] * (int64_t)meta->expr_meta.prime_limbs[j];
                        overflow_remainder[i + j] -= prod;
                    }
                }
            }
            
            uint32_t c_count = meta->carry_limb_counts[var_idx];
            
            memset(carry_buf, 0, max_carry_count * sizeof(uint32_t));
            
            carry_limbs_overflow(
                overflow_remainder,
                carry_buf,
                c_count,
                limb_bits
            );

            for (uint32_t limb = 0; limb < q_count; limb++) {
                uint32_t q_limb = quotient_buf[limb];
                
                if (!quotient.is_negative) {
                    core_row[col] = Fp(q_limb);
                } else {
                    core_row[col] = Fp::zero() - Fp(q_limb);
                }
                
                col++;
            }

            memcpy(&all_carries[c_offset], carry_buf, c_count * sizeof(uint32_t));
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
    
    free(temp_buffer);
}

__device__ void run_field_expression_gpu(
    const FieldExprMeta* meta,
    const uint32_t* inputs,
    const bool* flags,
    uint32_t* output_vars
) {
    uint32_t num_limbs = meta->num_limbs;
    uint32_t limb_bits = meta->limb_bits;
    
    uint32_t temp_storage_size = 4 * num_limbs + meta->max_ast_depth * num_limbs * sizeof(uint32_t);
    
    uint32_t* temp_storage = (uint32_t*)malloc(temp_storage_size * sizeof(uint32_t));
    if (temp_storage == nullptr) return;
    memset(temp_storage, 0, temp_storage_size * sizeof(uint32_t));

    for (uint32_t var = 0; var < meta->expr_meta.num_vars; var++) {
        for (uint32_t limb = 0; limb < num_limbs; limb++) {
            output_vars[var * num_limbs + limb] = 0;
        }
    }
    
    for (uint32_t var = 0; var < meta->expr_meta.num_vars; var++) {
        uint32_t root = meta->compute_root_indices[var];
        uint32_t* result = &output_vars[var * num_limbs];

        compute(
            result,
            meta->compute_expr_ops,
            root,
            &meta->expr_meta,
            inputs,
            output_vars,
            flags,
            num_limbs,
            limb_bits,
            temp_storage
        );
    }
    
    BigUintGpu prime(meta->expr_meta.prime_limbs, meta->expr_meta.prime_limb_count, meta->limb_bits);
    
    for (uint32_t var_idx = 0; var_idx < meta->expr_meta.num_vars; var_idx++) {
        uint32_t* var_ptr = &output_vars[var_idx * meta->num_limbs];
        
        BigUintGpu var_value(var_ptr, meta->num_limbs, meta->limb_bits);
        
        while (biguint_compare(&var_value, &prime) >= 0) {
            biguint_sub(&var_value, &var_value, &prime);
        }
        
        for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
            var_ptr[limb] = (limb < var_value.num_limbs) ? var_value.limbs[limb] : 0;
        }
    }
    
    free(temp_storage);
}

struct FieldExprCore {
    const FieldExprMeta* meta;
    VariableRangeChecker range_checker;
    BitwiseOperationLookup bitwise_lookup;
    
    __device__ explicit FieldExprCore(
        const FieldExprMeta* m, 
        VariableRangeChecker rc, 
        BitwiseOperationLookup bw
    ) : meta(m), range_checker(rc), bitwise_lookup(bw) {}
    
    __device__ void fill_trace_row(RowSlice core_row, const FieldExprCoreRecord* core_rec) {
        const uint8_t* rec_bytes = core_rec->input_limbs;
        uint8_t opcode = core_rec->opcode;
        
        uint32_t in_size = INPUT_U32_COUNT(meta);
        uint32_t var_size = VAR_U32_COUNT(meta);
        
        uint32_t* inputs = (uint32_t*)malloc(in_size * sizeof(uint32_t));
        uint32_t* output_vars = (uint32_t*)malloc(var_size * sizeof(uint32_t));
        bool* flags = (bool*)malloc(meta->num_u32_flags * sizeof(bool));
        
        if (inputs == nullptr || output_vars == nullptr || flags == nullptr) {
            if (inputs) free(inputs);
            if (output_vars) free(output_vars);
            if (flags) free(flags);
            return;
        }
        
        memset(inputs, 0, in_size * sizeof(uint32_t));
        memset(output_vars, 0, var_size * sizeof(uint32_t));
        memset(flags, 0, meta->num_u32_flags * sizeof(bool));

        uint32_t bytes_per_limb = (meta->limb_bits + 7) / 8;
        
        for (uint32_t i = 0; i < meta->num_inputs; i++) {
            for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
                size_t base = (size_t(i) * meta->num_limbs + limb) * bytes_per_limb;
                uint32_t v = 0;
                for (uint32_t b = 0; b < bytes_per_limb; b++) {
                    v |= (uint32_t)rec_bytes[base + b] << (8 * b);
                }
                inputs[i * meta->num_limbs + limb] = v;
                range_checker.add_count(v, meta->limb_bits);
            }
        }

        for (uint32_t j = 0; j < meta->num_local_opcodes; j++) {
            if (opcode == meta->local_opcode_idx[j] && j < meta->num_u32_flags) {
                flags[meta->opcode_flag_idx[j]] = true;
            }
        }

        run_field_expression_gpu(
            meta,
            inputs,
            flags,
            output_vars
        );

        for (uint32_t i = 0; i < meta->expr_meta.num_vars; i++) {
            for (uint32_t limb = 0; limb < meta->num_limbs; limb++) {
                uint32_t var_val = output_vars[i * meta->num_limbs + limb];
                range_checker.add_count(var_val, meta->limb_bits);
            }
        }

        generate_subrow_gpu(
            meta,
            inputs,
            flags,
            output_vars,
            core_row
        );
        
        free(inputs);
        free(output_vars);
        free(flags);
    }
};

__global__ void field_expression_tracegen(
    const uint8_t* records,
    Fp* trace,
    const FieldExprMeta* meta,
    size_t num_records,
    size_t record_stride,
    size_t width,
    size_t height,
    uint32_t* range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t* bitwise_lookup_ptr,
    uint32_t bitwise_num_bits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) return;

    RowSlice row(trace + idx, height);
    size_t pointer_max_bits = 32;
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);
    
    if (idx < num_records) {
        const uint8_t* rec_bytes = records + idx * record_stride;
        
        size_t adapter_size = 0;
        route_rv32_vec_heap_adapter(
            row, rec_bytes, meta, pointer_max_bits, 
            range_checker, bitwise_lookup, adapter_size
        );
        
        const uint8_t* core_bytes = rec_bytes + adapter_size;
        const FieldExprCoreRecord* core_rec = reinterpret_cast<const FieldExprCoreRecord*>(core_bytes);
        
        FieldExprCore core(meta, range_checker, bitwise_lookup);
        core.fill_trace_row(
            row.slice_from(meta->adapter_width),
            core_rec
        );
    } else {
        // We can't just fill with 0s, instead calling w/ invalid opcode
        row.fill_zero(0, meta->adapter_width);
        
        FieldExprCore dummy_core(meta, range_checker, bitwise_lookup);
        
        uint32_t dummy_core_size = 1 + meta->num_inputs * meta->num_limbs * 4;
        uint8_t* dummy_core_data = (uint8_t*)malloc(dummy_core_size);
        
        if (dummy_core_data != nullptr) {
            memset(dummy_core_data, 0, dummy_core_size);
            dummy_core_data[0] = 0xFF;
            const FieldExprCoreRecord* dummy_core_record = 
                reinterpret_cast<const FieldExprCoreRecord*>(dummy_core_data);
            
            dummy_core.fill_trace_row(row.slice_from(meta->adapter_width), dummy_core_record);
            
            row.write(meta->adapter_width, Fp::zero());
            
            free(dummy_core_data);
        }
    }
}

extern "C" int _field_expression_tracegen(
    const uint8_t* d_records,
    Fp* d_trace,
    const FieldExprMeta* d_meta,
    size_t num_records,
    size_t record_stride,
    size_t width,
    size_t height,
    uint32_t* d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t* d_bitwise_lookup,
    uint32_t bitwise_num_bits
) {
    assert((height & (height - 1)) == 0);
    auto [grid, block] = kernel_launch_params(height);
    field_expression_tracegen<<<grid, block>>>(
        (uint8_t*)d_records,
        d_trace,
        d_meta,
        num_records,
        record_stride,
        width,
        height,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits
    );
    
    return cudaGetLastError();
}