// Generic GPU tracegen for mod-builder FieldExpr chips.
// Interprets the trace-generation IR encoded by tracegen_ir (see that module
// for the semantics contract). One thread per row, grid-stride.
//
// Validated bit-exact against the CPU tracegen (FieldExpressionFiller) by the
// cuda-gated chip tests in the algebra and ecc circuit extensions (modular,
// Fp2, and Weierstrass chips), which compare full GPU and CPU traces
// element-wise. Adapter columns use the shared Rv64VecHeapAdapter device fill.
#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/vec_heap.cuh"
#include "system/memory/params.cuh"
#include "tracegen_abi.cuh"

#include <cstdint>

static constexpr uint32_t MB_BB_P = 0x78000001u; // canonical arithmetic on carries/q
// Required configuration boundary for shared field headers.
#define F_P 0x78000001u

struct FieldExprProg {
    int num_limbs, limb_bits, k, num_input, num_vars, num_flags, needs_setup, width;
    int num_slots, n_eval_ops, n_witness_ops, n_cons, scratch_len, p8_len;
    int n_local_ops, n_op_flags;
    uint32_t mprime;
    const uint32_t *d_eval_ops, *d_witness_ops, *d_cons, *d_p, *d_r2, *d_pm2, *d_pinv, *d_p8;
    const uint32_t *d_mont, *d_climbs, *d_optab;
};
#define Prog FieldExprProg

__device__ __forceinline__ FieldExprProg load_prog(const uint32_t *d_blob) {
    FieldExprProg s;
    s.num_limbs = d_blob[H_NUM_LIMBS]; s.limb_bits = d_blob[H_LIMB_BITS]; s.k = d_blob[H_K];
    s.num_input = d_blob[H_NUM_INPUT]; s.num_vars = d_blob[H_NUM_VARS];
    s.num_flags = d_blob[H_NUM_FLAGS]; s.needs_setup = d_blob[H_NEEDS_SETUP];
    s.width = d_blob[H_WIDTH]; s.num_slots = d_blob[H_NUM_SLOTS];
    s.n_eval_ops = d_blob[H_N_EVAL_OPS]; s.n_witness_ops = d_blob[H_N_WITNESS_OPS];
    s.n_cons = d_blob[H_N_CONS];
    s.scratch_len = d_blob[H_SCRATCH_LEN]; s.p8_len = d_blob[H_P8_LEN];
    s.n_local_ops = d_blob[H_N_LOCAL_OPS]; s.n_op_flags = d_blob[H_N_OP_FLAGS];
    s.mprime = d_blob[H_MPRIME];
    s.d_eval_ops = d_blob + d_blob[H_OFF_EVAL_OPS];
    s.d_witness_ops = d_blob + d_blob[H_OFF_WITNESS_OPS];
    s.d_cons = d_blob + d_blob[H_OFF_CONS]; s.d_p = d_blob + d_blob[H_OFF_P];
    s.d_r2 = d_blob + d_blob[H_OFF_R2]; s.d_pm2 = d_blob + d_blob[H_OFF_PM2];
    s.d_pinv = d_blob + d_blob[H_OFF_PINV]; s.d_p8 = d_blob + d_blob[H_OFF_P8];
    s.d_mont = d_blob + d_blob[H_OFF_MONT]; s.d_climbs = d_blob + d_blob[H_OFF_CLIMBS];
    s.d_optab = d_blob + d_blob[H_OFF_OPTAB];
    return s;
}

// ---- K-limb Montgomery arithmetic (k <= TRACEGEN_MAX_K, runtime loops) ----
__device__ void mont_mul(const Prog &s, const uint32_t *a, const uint32_t *b, uint32_t *r) {
    uint32_t t[TRACEGEN_MAX_K + 2];
    const int k = s.k;
    for (int i = 0; i < k + 2; i++) t[i] = 0;
    for (int i = 0; i < k; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < k; j++) {
            uint64_t cur = (uint64_t)t[j] + (uint64_t)a[i] * b[j] + carry;
            t[j] = (uint32_t)cur;
            carry = cur >> 32;
        }
        uint64_t cur = (uint64_t)t[k] + carry;
        t[k] = (uint32_t)cur;
        t[k + 1] = (uint32_t)(cur >> 32);
        uint32_t m = t[0] * s.mprime;
        carry = ((uint64_t)t[0] + (uint64_t)m * s.d_p[0]) >> 32;
        for (int j = 1; j < k; j++) {
            uint64_t cur2 = (uint64_t)t[j] + (uint64_t)m * s.d_p[j] + carry;
            t[j - 1] = (uint32_t)cur2;
            carry = cur2 >> 32;
        }
        uint64_t cur3 = (uint64_t)t[k] + carry;
        t[k - 1] = (uint32_t)cur3;
        t[k] = t[k + 1] + (uint32_t)(cur3 >> 32);
        t[k + 1] = 0;
    }
    uint32_t sub[TRACEGEN_MAX_K];
    uint32_t borrow = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)t[j] - s.d_p[j] - borrow;
        sub[j] = (uint32_t)cur;
        borrow = (cur >> 32) ? 1 : 0;
    }
    bool ge = (t[k] != 0) || !borrow;
    for (int j = 0; j < k; j++) r[j] = ge ? sub[j] : t[j];
}

__device__ void add_mod(const Prog &s, const uint32_t *a, const uint32_t *b, uint32_t *r) {
    const int k = s.k;
    uint32_t t[TRACEGEN_MAX_K];
    uint64_t carry = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)a[j] + b[j] + carry;
        t[j] = (uint32_t)cur;
        carry = cur >> 32;
    }
    uint32_t sub[TRACEGEN_MAX_K];
    uint32_t borrow = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)t[j] - s.d_p[j] - borrow;
        sub[j] = (uint32_t)cur;
        borrow = (cur >> 32) ? 1 : 0;
    }
    bool ge = carry || !borrow;
    for (int j = 0; j < k; j++) r[j] = ge ? sub[j] : t[j];
}

__device__ void sub_mod(const Prog &s, const uint32_t *a, const uint32_t *b, uint32_t *r) {
    const int k = s.k;
    uint32_t borrow = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)a[j] - b[j] - borrow;
        r[j] = (uint32_t)cur;
        borrow = (cur >> 32) ? 1 : 0;
    }
    if (borrow) {
        uint64_t carry = 0;
        for (int j = 0; j < k; j++) {
            uint64_t cur = (uint64_t)r[j] + s.d_p[j] + carry;
            r[j] = (uint32_t)cur;
            carry = cur >> 32;
        }
    }
}

// a^(p-2) via square-and-multiply; inv(0) = 0 by convention.
__device__ void mont_inv(const Prog &s, const uint32_t *a, uint32_t *r) {
    const int k = s.k;
    uint32_t acc[TRACEGEN_MAX_K];
    bool started = false;
    for (int bit = 32 * k - 1; bit >= 0; bit--) {
        if (started) mont_mul(s, acc, acc, acc);
        if ((s.d_pm2[bit / 32] >> (bit % 32)) & 1) {
            if (!started) {
                for (int j = 0; j < k; j++) acc[j] = a[j];
                started = true;
            } else {
                mont_mul(s, acc, a, acc);
            }
        }
    }
    for (int j = 0; j < k; j++) r[j] = started ? acc[j] : 0;
}



__device__ __forceinline__ uint32_t f_of_i64(int64_t v) {
    int64_t m = v % (int64_t)MB_BB_P;
    if (m < 0) m += MB_BB_P;
    return (uint32_t)m;
}

// Fill the core sub-row. CUDA equivalence tests compare this output against CPU tracegen.
// `core_row` must point at the first core column. When `is_dummy`, inputs are zero,
// flags false, range checks skipped, is_valid = 0 (mirrors fill_dummy_trace_row).
__device__ void field_expr_fill_core_row(
    const FieldExprProg &s,
    RowSlice core_row,
    const uint8_t *d_rec, // opcode byte + input limbs; may be null when is_dummy
    VariableRangeChecker rc,
    uint32_t *d_aux,
    bool is_dummy,
    uint32_t *d_err) {
    const int k = s.k, nl = s.num_limbs, lb = s.limb_bits;
    uint32_t *d_slots = d_aux;                             // num_slots * k
    uint32_t *d_var_canon = d_slots + s.num_slots * k;     // num_vars * k
    int32_t *d_scratch = (int32_t *)(d_var_canon + s.num_vars * k); // scratch_len
    uint32_t *d_nacc = (uint32_t *)(d_scratch + s.scratch_len);     // 2k
    uint32_t *d_q512 = d_nacc + 2 * k;                            // 2k

    const uint32_t opcode = is_dummy ? 0xffffffffu : d_rec[0];
    const uint8_t *d_in_limbs = d_rec + 1;

    // flags
    bool flags[TRACEGEN_MAX_FLAGS];
    for (int f = 0; f < s.num_flags; f++) flags[f] = false;
    if (s.needs_setup && !is_dummy) {
        for (int posn = 0; posn < s.n_local_ops; posn++) {
            if (s.d_optab[posn] == opcode && posn < s.n_op_flags) {
                flags[s.d_optab[s.n_local_ops + posn]] = true;
                break;
            }
        }
    }

    // ---- evaluation phase ----
    uint32_t one[TRACEGEN_MAX_K];
    for (int j = 0; j < k; j++) one[j] = j == 0 ? 1 : 0;
    for (int i = 0; i < s.num_slots * k; i++) d_slots[i] = 0;
    for (int io = 0; io < s.n_eval_ops; io++) {
        const uint32_t *d_op = s.d_eval_ops + 5 * io;
        const uint32_t opc = d_op[0], flag = d_op[1], dst = d_op[2], a = d_op[3], b = d_op[4];
        uint32_t *d_dst = d_slots + dst * k;
        const uint32_t *d_a = d_slots + a * k;
        const uint32_t *d_b = d_slots + b * k;
        switch (opc) {
            case EVAL_OP_LOAD_INPUT: {
                uint32_t canon[TRACEGEN_MAX_K];
                for (int j = 0; j < k; j++) canon[j] = 0;
                if (!is_dummy) {
                    const uint8_t *d_src = d_in_limbs + a * nl;
                    for (int i = 0; i < nl; i++)
                        canon[i * lb / 32] |= (uint32_t)d_src[i] << ((i * lb) % 32);
                }
                mont_mul(s, canon, s.d_r2, d_dst);
                break;
            }
            case EVAL_OP_CONST:
                for (int j = 0; j < k; j++) d_dst[j] = s.d_mont[a * k + j];
                break;
            case EVAL_OP_ADD: add_mod(s, d_a, d_b, d_dst); break;
            case EVAL_OP_SUB: sub_mod(s, d_a, d_b, d_dst); break;
            case EVAL_OP_MUL: mont_mul(s, d_a, d_b, d_dst); break;
            case EVAL_OP_DIV: {
                uint32_t inv[TRACEGEN_MAX_K];
                mont_inv(s, d_b, inv);
                mont_mul(s, d_a, inv, d_dst);
                break;
            }
            case EVAL_OP_INTADD: add_mod(s, d_a, s.d_mont + b * k, d_dst); break;
            case EVAL_OP_INTMUL: mont_mul(s, d_a, s.d_mont + b * k, d_dst); break;
            case EVAL_OP_SELECT: {
                const uint32_t *d_src = flags[flag] ? d_a : d_b;
                for (int j = 0; j < k; j++) d_dst[j] = d_src[j];
                break;
            }
            case EVAL_OP_SAVE_VAR:
                mont_mul(s, d_b, one, d_var_canon + a * k);
                for (int j = 0; j < k; j++) d_dst[j] = d_b[j];
                break;
        }
    }

    // ---- trace columns: is_valid, inputs, vars ----
    size_t col = 0;
#define MB_PUT(v) do { core_row[col] = Fp((uint32_t)(v)); col++; } while (0)
    MB_PUT(is_dummy ? 0u : 1u);
    for (int i = 0; i < s.num_input * nl; i++)
        MB_PUT(is_dummy ? 0u : (uint32_t)d_in_limbs[i]);
    for (int v = 0; v < s.num_vars; v++)
        for (int i = 0; i < nl; i++) {
            uint32_t limb = (d_var_canon[v * k + i / 4] >> ((i % 4) * 8)) & 0xff;
            MB_PUT(limb);
            if (!is_dummy) rc.add_count(limb, lb);
        }

    // ---- witness phase ----
    size_t carry_col = col;
    for (int ci = 0; ci < s.n_cons; ci++) carry_col += (s.d_cons + 8 * ci)[4];
    for (int ci = 0; ci < s.n_cons; ci++) {
        const uint32_t *d_constraint = s.d_cons + 8 * ci;
        const uint32_t tape_start = d_constraint[0], tape_len = d_constraint[1];
        const uint32_t res_off = d_constraint[2], res_len = d_constraint[3];
        const uint32_t q_limbs_n = d_constraint[4], carry_limbs_n = d_constraint[5];
        const uint32_t carry_min_abs = d_constraint[6], carry_bits = d_constraint[7];

        for (uint32_t io = 0; io < tape_len; io++) {
            const uint32_t *d_op = s.d_witness_ops + 9 * (tape_start + io);
            const uint32_t opc = d_op[0], flag = d_op[1];
            const uint32_t d = d_op[2], dl = d_op[3], ao = d_op[4], al = d_op[5];
            const uint32_t bo = d_op[6], bl = d_op[7];
            const int32_t imm = (int32_t)d_op[8];
            switch (opc) {
                case WITNESS_OP_INPUT:
                    for (uint32_t i = 0; i < dl; i++)
                        d_scratch[d + i] =
                            is_dummy ? 0 : (int32_t)d_in_limbs[ao * nl + i];
                    break;
                case WITNESS_OP_VAR:
                    for (uint32_t i = 0; i < dl; i++)
                        d_scratch[d + i] =
                            (int32_t)((d_var_canon[ao * k + i / 4] >> ((i % 4) * 8)) & 0xff);
                    break;
                case WITNESS_OP_CONST:
                    for (uint32_t i = 0; i < dl; i++)
                        d_scratch[d + i] = (int32_t)s.d_climbs[ao + i];
                    break;
                case WITNESS_OP_ADD:
                case WITNESS_OP_SUB:
                    for (uint32_t i = 0; i < dl; i++) {
                        int32_t a = i < al ? d_scratch[ao + i] : 0;
                        int32_t b = i < bl ? d_scratch[bo + i] : 0;
                        d_scratch[d + i] = opc == WITNESS_OP_ADD ? a + b : a - b;
                    }
                    break;
                case WITNESS_OP_MUL:
                    for (int32_t i = (int32_t)dl - 1; i >= 0; i--) {
                        int64_t acc = 0;
                        int32_t lo = i - (int32_t)bl + 1 < 0 ? 0 : i - (int32_t)bl + 1;
                        int32_t hi = i < (int32_t)al - 1 ? i : (int32_t)al - 1;
                        for (int32_t j = lo; j <= hi; j++)
                            acc += (int64_t)d_scratch[ao + j] * d_scratch[bo + (i - j)];
                        d_scratch[d + i] = (int32_t)acc;
                    }
                    break;
                case WITNESS_OP_INTADD:
                    for (uint32_t i = 0; i < dl; i++) d_scratch[d + i] = d_scratch[ao + i];
                    d_scratch[d] += imm;
                    break;
                case WITNESS_OP_INTMUL:
                    for (uint32_t i = 0; i < dl; i++)
                        d_scratch[d + i] = d_scratch[ao + i] * imm;
                    break;
                case WITNESS_OP_SELECT: {
                    const uint32_t src = flags[flag] ? ao : bo;
                    const uint32_t sl = flags[flag] ? al : bl;
                    for (uint32_t i = 0; i < dl; i++)
                        d_scratch[d + i] = i < sl ? d_scratch[src + i] : 0;
                    break;
                }
            }
        }

        // N mod 2^(64K) from result limbs
        for (int i = 0; i < 2 * k; i++) d_nacc[i] = 0;
        for (uint32_t i = 0; i < res_len; i++) {
            int64_t v = d_scratch[res_off + i];
            if (v == 0) continue;
            uint64_t mag = v < 0 ? (uint64_t)(-v) : (uint64_t)v;
            int word = i / 4, shift = (i % 4) * 8;
            uint64_t lo64 = mag << shift;
            uint64_t hi64 = shift ? (mag >> (64 - shift)) : 0;
            uint32_t parts[3] = {(uint32_t)lo64, (uint32_t)(lo64 >> 32), (uint32_t)hi64};
            if (v > 0) {
                uint64_t carry = 0;
                for (int w = 0; w < 2 * k - word; w++) {
                    uint64_t add = (w < 3 ? parts[w] : 0) + carry;
                    if (w >= 3 && add == 0) break;
                    uint64_t cur = (uint64_t)d_nacc[word + w] + add;
                    d_nacc[word + w] = (uint32_t)cur;
                    carry = cur >> 32;
                }
            } else {
                int64_t borrow = 0;
                for (int w = 0; w < 2 * k - word; w++) {
                    int64_t sub = (int64_t)(w < 3 ? parts[w] : 0) + borrow;
                    if (w >= 3 && sub == 0) break;
                    int64_t cur = (int64_t)d_nacc[word + w] - sub;
                    d_nacc[word + w] = (uint32_t)cur;
                    borrow = cur < 0 ? 1 : 0;
                }
            }
        }
        // q = N * pinv mod 2^(64K), exact division
        for (int i = 0; i < 2 * k; i++) d_q512[i] = 0;
        for (int i = 0; i < 2 * k; i++) {
            if (d_nacc[i] == 0) continue;
            uint64_t carry = 0;
            for (int j = 0; j < 2 * k - i; j++) {
                uint64_t prod = (uint64_t)d_nacc[i] * s.d_pinv[j];
                uint64_t cur = (uint64_t)d_q512[i + j] + (prod & 0xffffffffu) + carry;
                d_q512[i + j] = (uint32_t)cur;
                carry = (cur >> 32) + (prod >> 32);
            }
        }
        bool neg = (d_q512[2 * k - 1] >> 31) != 0;
        if (neg) {
            uint64_t carry = 1;
            for (int w = 0; w < 2 * k; w++) {
                uint64_t cur = (uint64_t)(~d_q512[w]) + carry;
                d_q512[w] = (uint32_t)cur;
                carry = cur >> 32;
            }
        }
        int32_t ql[TRACEGEN_MAX_Q_LIMBS];
        for (uint32_t i = 0; i < q_limbs_n; i++) {
            int32_t byte = (int32_t)((d_q512[i / 4] >> ((i % 4) * 8)) & 0xff);
            ql[i] = neg ? -byte : byte;
            MB_PUT(f_of_i64(ql[i]));
            if (!is_dummy) rc.add_count((uint32_t)(ql[i] + (1 << lb)), lb + 1);
        }
        int64_t carry_acc = 0;
        for (uint32_t i = 0; i < carry_limbs_n; i++) {
            int64_t e = i < res_len ? (int64_t)d_scratch[res_off + i] : 0;
            int32_t lo = (int32_t)i - s.p8_len + 1 < 0 ? 0 : (int32_t)i - s.p8_len + 1;
            int32_t hi = i < q_limbs_n - 1 ? (int32_t)i : (int32_t)q_limbs_n - 1;
            for (int32_t j = lo; j <= hi; j++)
                e -= (int64_t)ql[j] * (int32_t)s.d_p8[i - j];
            carry_acc = (e + carry_acc) >> lb;
            core_row[carry_col] = Fp(f_of_i64(carry_acc));
            carry_col++;
            if (!is_dummy) rc.add_count((uint32_t)(carry_acc + carry_min_abs), carry_bits);
        }
    }
    for (int f = 0; f < s.num_flags; f++)
        core_row[carry_col + f] = Fp((!is_dummy && flags[f]) ? 1u : 0u);
    if (carry_col + s.num_flags != (size_t)s.width) atomicAdd(d_err, 1u);
#undef MB_PUT
}

// ---------------- templated kernel: adapter + core ----------------
template <size_t NUM_READS, size_t BLOCKS>
__global__ void field_expr_tracegen_kernel(
    Fp *__restrict__ d_trace, size_t height, size_t rows_used,
    const uint32_t *__restrict__ d_blob,
    const uint8_t *__restrict__ d_records, size_t rec_stride, size_t rec_core_offset,
    uint32_t *__restrict__ d_range_checker, size_t rc_bins,
    uint32_t *__restrict__ d_aux, size_t aux_words,
    uint32_t pointer_max_bits, uint32_t timestamp_max_bits,
    int should_finalize, uint32_t *__restrict__ d_err) {
    __shared__ FieldExprProg shared_prog;
    if (threadIdx.x == 0) shared_prog = load_prog(d_blob);
    __syncthreads();
    const FieldExprProg &s = shared_prog;
    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv64VecHeapAdapterCols<uint8_t, NUM_READS, BLOCKS, BLOCKS>);
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t nthreads = gridDim.x * (size_t)blockDim.x;
    VariableRangeChecker rc(d_range_checker, rc_bins);

    for (size_t row = tid; row < height; row += nthreads) {
        uint32_t *d_thread_aux = d_aux + tid * aux_words;
        RowSlice row_slice(d_trace + row, height);
        if (row < rows_used) {
            const uint8_t *d_rec = d_records + row * rec_stride;
            Rv64VecHeapAdapter<NUM_READS, BLOCKS, BLOCKS> adapter(
                pointer_max_bits, rc, timestamp_max_bits);
            adapter.fill_trace_row(
                row_slice,
                *(const Rv64VecHeapAdapterRecord<NUM_READS, BLOCKS, BLOCKS> *)d_rec);
            field_expr_fill_core_row(s, row_slice.slice_from(ADAPTER_WIDTH),
                                     d_rec + rec_core_offset, rc, d_thread_aux, false, d_err);
        } else {
            row_slice.fill_zero(0, ADAPTER_WIDTH);
            if (should_finalize) {
                field_expr_fill_core_row(s, row_slice.slice_from(ADAPTER_WIDTH), nullptr, rc,
                                         d_thread_aux, true, d_err);
            } else {
                row_slice.fill_zero(ADAPTER_WIDTH, s.width);
            }
        }
    }
}

#define MB_LAUNCHER(R, B)                                                                     \
    extern "C" int _field_expr_tracegen_r##R##_b##B(                                          \
        Fp *d_trace, size_t height, size_t rows_used, const uint32_t *d_blob,                 \
        const uint8_t *d_records, size_t rec_stride, size_t rec_core_offset,                  \
        uint32_t *d_range_checker, size_t rc_bins, uint32_t *d_aux, size_t aux_words,         \
        uint32_t pointer_max_bits, uint32_t timestamp_max_bits, int should_finalize,          \
        uint32_t *d_err, cudaStream_t stream) {                                               \
        const int threads = 256;                                                              \
        int device, sm_count;                                                                 \
        cudaError_t err = cudaGetDevice(&device);                                             \
        if (err != cudaSuccess) return err;                                                   \
        err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);       \
        if (err != cudaSuccess) return err;                                                   \
        const size_t want = (height + threads - 1) / threads;                                 \
        const size_t max_blocks = 512 / sm_count * sm_count;                                  \
        const size_t rounded = (want + sm_count - 1) / sm_count * sm_count;                    \
        const int blocks = (int)(rounded < max_blocks ? rounded : max_blocks);                 \
        field_expr_tracegen_kernel<R, B><<<blocks, threads, 0, stream>>>(                     \
            d_trace, height, rows_used, d_blob, d_records, rec_stride, rec_core_offset,       \
            d_range_checker, rc_bins, d_aux, aux_words, pointer_max_bits,                     \
            timestamp_max_bits, should_finalize, d_err);                                      \
        return CHECK_KERNEL();                                                                \
    }

MB_LAUNCHER(2, 4)   // modular 32-limb (addsub, muldiv)
MB_LAUNCHER(2, 6)   // modular 48-limb
MB_LAUNCHER(2, 8)   // fp2 32-limb, EcAddNe 32-limb
MB_LAUNCHER(2, 12)  // fp2 48-limb, EcAddNe 48-limb
MB_LAUNCHER(1, 8)   // EcDouble 32-limb
MB_LAUNCHER(1, 12)  // EcDouble 48-limb
