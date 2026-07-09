// Generic GPU tracegen interpreter for mod-builder FieldExpr chips.
// Executes the "device program" blob produced by device_program.rs (see that file
// for the semantics contract). One thread per row, grid-stride.
//
// Validated bit-exact against FieldExpressionFiller (CPU) on EcAddNe, MulDiv
// (flags + Select + Div-under-Select) and IntAdd/IntMul expressions; see
// device_program.rs tests and the standalone harness this was extracted from.
//
// TODO(integration):
//  - adapter columns: call the Rv64VecHeapAdapter device fill for columns
//    [0, adapter_width) per row (same pattern as bigint.cu); this kernel fills
//    core columns at [adapter_width, adapter_width + width).
//  - records: rec_core_offset selects the core record inside the
//    (adapter_record, core_record) dense layout.
#include "launcher.cuh"
#include "primitives/trace_access.h"

#include <cstdint>

#define MAX_K 12       // up to 384-bit fields (BLS12-381)
#define F_P 0x78000001u // BabyBear

// Value-phase opcodes
enum { VOP_LOAD_INPUT = 0, VOP_CONST, VOP_ADD, VOP_SUB, VOP_MUL, VOP_DIV,
       VOP_INTADD, VOP_INTMUL, VOP_SELECT, VOP_SAVE_VAR };
// Limb-phase opcodes
enum { LOP_INPUT = 0, LOP_VAR, LOP_CONST, LOP_ADD, LOP_SUB, LOP_MUL,
       LOP_INTADD, LOP_INTMUL, LOP_SELECT };

// Header word indices (see device_program.rs to_blob)
enum { H_NUM_LIMBS = 0, H_LIMB_BITS, H_K, H_NUM_INPUT, H_NUM_VARS, H_NUM_FLAGS,
       H_NEEDS_SETUP, H_WIDTH, H_NUM_SLOTS, H_N_VOPS, H_N_LOPS, H_N_CONS,
       H_SCRATCH_LEN, H_P8_LEN, H_N_LOCAL_OPS, H_N_OP_FLAGS,
       H_OFF_VOPS, H_OFF_LOPS, H_OFF_CONS, H_OFF_P, H_OFF_R2, H_OFF_PM2,
       H_OFF_PINV, H_OFF_P8, H_OFF_MONT, H_OFF_CLIMBS, H_OFF_OPTAB, H_MPRIME };

struct Prog {
    // scalars
    int num_limbs, limb_bits, k, num_input, num_vars, num_flags, needs_setup, width;
    int num_slots, n_vops, n_lops, n_cons, scratch_len, p8_len, n_local_ops, n_op_flags;
    uint32_t mprime;
    // sections
    const uint32_t *vops;   // 5 words each
    const uint32_t *lops;   // 9 words each
    const uint32_t *cons;   // 8 words each
    const uint32_t *p, *r2, *pm2, *pinv;
    const uint32_t *p8;     // i32 stored as u32
    const uint32_t *mont;   // K-limb montgomery payloads
    const uint32_t *climbs; // i32 const limbs
    const uint32_t *optab;  // local_opcode_idx ++ opcode_flag_idx
};

__device__ __forceinline__ Prog load_prog(const uint32_t *blob) {
    Prog s;
    s.num_limbs = blob[H_NUM_LIMBS]; s.limb_bits = blob[H_LIMB_BITS]; s.k = blob[H_K];
    s.num_input = blob[H_NUM_INPUT]; s.num_vars = blob[H_NUM_VARS];
    s.num_flags = blob[H_NUM_FLAGS]; s.needs_setup = blob[H_NEEDS_SETUP];
    s.width = blob[H_WIDTH]; s.num_slots = blob[H_NUM_SLOTS];
    s.n_vops = blob[H_N_VOPS]; s.n_lops = blob[H_N_LOPS]; s.n_cons = blob[H_N_CONS];
    s.scratch_len = blob[H_SCRATCH_LEN]; s.p8_len = blob[H_P8_LEN];
    s.n_local_ops = blob[H_N_LOCAL_OPS]; s.n_op_flags = blob[H_N_OP_FLAGS];
    s.mprime = blob[H_MPRIME];
    s.vops = blob + blob[H_OFF_VOPS]; s.lops = blob + blob[H_OFF_LOPS];
    s.cons = blob + blob[H_OFF_CONS]; s.p = blob + blob[H_OFF_P];
    s.r2 = blob + blob[H_OFF_R2]; s.pm2 = blob + blob[H_OFF_PM2];
    s.pinv = blob + blob[H_OFF_PINV]; s.p8 = blob + blob[H_OFF_P8];
    s.mont = blob + blob[H_OFF_MONT]; s.climbs = blob + blob[H_OFF_CLIMBS];
    s.optab = blob + blob[H_OFF_OPTAB];
    return s;
}

// ---- K-limb Montgomery arithmetic (k <= MAX_K, runtime loops) ----
__device__ void mont_mul(const Prog &s, const uint32_t *a, const uint32_t *b, uint32_t *r) {
    uint32_t t[MAX_K + 2];
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
        carry = ((uint64_t)t[0] + (uint64_t)m * s.p[0]) >> 32;
        for (int j = 1; j < k; j++) {
            uint64_t cur2 = (uint64_t)t[j] + (uint64_t)m * s.p[j] + carry;
            t[j - 1] = (uint32_t)cur2;
            carry = cur2 >> 32;
        }
        uint64_t cur3 = (uint64_t)t[k] + carry;
        t[k - 1] = (uint32_t)cur3;
        t[k] = t[k + 1] + (uint32_t)(cur3 >> 32);
        t[k + 1] = 0;
    }
    uint32_t sub[MAX_K];
    uint32_t borrow = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)t[j] - s.p[j] - borrow;
        sub[j] = (uint32_t)cur;
        borrow = (cur >> 32) ? 1 : 0;
    }
    bool ge = (t[k] != 0) || !borrow;
    for (int j = 0; j < k; j++) r[j] = ge ? sub[j] : t[j];
}

__device__ void add_mod(const Prog &s, const uint32_t *a, const uint32_t *b, uint32_t *r) {
    const int k = s.k;
    uint32_t t[MAX_K];
    uint64_t carry = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)a[j] + b[j] + carry;
        t[j] = (uint32_t)cur;
        carry = cur >> 32;
    }
    uint32_t sub[MAX_K];
    uint32_t borrow = 0;
    for (int j = 0; j < k; j++) {
        uint64_t cur = (uint64_t)t[j] - s.p[j] - borrow;
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
            uint64_t cur = (uint64_t)r[j] + s.p[j] + carry;
            r[j] = (uint32_t)cur;
            carry = cur >> 32;
        }
    }
}

// a^(p-2) via square-and-multiply; inv(0) = 0 by convention.
__device__ void mont_inv(const Prog &s, const uint32_t *a, uint32_t *r) {
    const int k = s.k;
    uint32_t acc[MAX_K];
    bool started = false;
    for (int bit = 32 * k - 1; bit >= 0; bit--) {
        if (started) mont_mul(s, acc, acc, acc);
        if ((s.pm2[bit / 32] >> (bit % 32)) & 1) {
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
    int64_t m = v % (int64_t)F_P;
    if (m < 0) m += F_P;
    return (uint32_t)m;
}

// Per-thread auxiliary layout (u32 words):
//   [0, num_slots*k)                       value slots
//   [slots_end, slots_end + num_vars*k)    canonical var limbs (packed u32)
//   [.., + scratch_len)                    limb scratch (i32)
//   [.., + 2k]  n accumulator; [.., + 2k]  q512
__device__ __forceinline__ size_t aux_words_needed(const Prog &s) {
    return (size_t)(s.num_slots + s.num_vars) * s.k + s.scratch_len + 4 * s.k;
}

__global__ void field_expr_tracegen(
    const uint32_t *__restrict__ blob,
    const uint8_t *__restrict__ records, size_t rec_stride, size_t rec_core_offset,
    size_t rows_used, size_t height,
    Fp *__restrict__ trace_core, // column-major core sub-trace, stride = height
    uint32_t *__restrict__ rc_count,
    uint32_t *__restrict__ aux, size_t aux_stride, // per-thread scratch
    int should_finalize,
    uint32_t *__restrict__ err) {
    const Prog s = load_prog(blob);
    const int k = s.k, nl = s.num_limbs, lb = s.limb_bits;
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t nthreads = gridDim.x * (size_t)blockDim.x;

    uint32_t *my_aux = aux + tid * aux_stride;
    uint32_t *slots = my_aux;                        // num_slots * k
    uint32_t *var_canon = slots + s.num_slots * k;   // num_vars * k
    int32_t *scratch = (int32_t *)(var_canon + s.num_vars * k); // scratch_len
    uint32_t *nacc = (uint32_t *)(scratch + s.scratch_len);     // 2k
    uint32_t *q512 = nacc + 2 * k;                              // 2k

    for (size_t row = tid; row < height; row += nthreads) {
        const bool is_dummy = row >= rows_used;
        RowSlice row_slice(trace_core + row, height);
        if (is_dummy && !should_finalize) {
            row_slice.fill_zero(0, s.width);
            continue;
        }
        const uint8_t *rec = records + row * rec_stride + rec_core_offset;
        const uint32_t opcode = is_dummy ? 0xffffffffu : rec[0];
        const uint8_t *in_limbs = rec + 1;

        // flags
        bool flags[32];
        for (int f = 0; f < s.num_flags; f++) flags[f] = false;
        if (s.needs_setup && !is_dummy) {
            for (int posn = 0; posn < s.n_local_ops; posn++) {
                if (s.optab[posn] == opcode && posn < s.n_op_flags) {
                    flags[s.optab[s.n_local_ops + posn]] = true;
                    break;
                }
            }
        }

        // ---- value phase ----
        uint32_t one[MAX_K];
        for (int j = 0; j < k; j++) one[j] = j == 0 ? 1 : 0;
        for (int i = 0; i < s.num_slots * k; i++) slots[i] = 0;
        for (int io = 0; io < s.n_vops; io++) {
            const uint32_t *op = s.vops + 5 * io;
            const uint32_t opc = op[0], flag = op[1], dst = op[2], a = op[3], b = op[4];
            uint32_t *d = slots + dst * k;
            const uint32_t *pa = slots + a * k;
            const uint32_t *pb = slots + b * k;
            switch (opc) {
                case VOP_LOAD_INPUT: {
                    uint32_t canon[MAX_K];
                    for (int j = 0; j < k; j++) canon[j] = 0;
                    if (!is_dummy) {
                        const uint8_t *src = in_limbs + a * nl;
                        for (int i = 0; i < nl; i++)
                            canon[i * lb / 32] |= (uint32_t)src[i] << ((i * lb) % 32);
                    }
                    mont_mul(s, canon, s.r2, d);
                    break;
                }
                case VOP_CONST:
                    for (int j = 0; j < k; j++) d[j] = s.mont[a * k + j];
                    break;
                case VOP_ADD: add_mod(s, pa, pb, d); break;
                case VOP_SUB: sub_mod(s, pa, pb, d); break;
                case VOP_MUL: mont_mul(s, pa, pb, d); break;
                case VOP_DIV: {
                    uint32_t inv[MAX_K];
                    mont_inv(s, pb, inv);
                    mont_mul(s, pa, inv, d);
                    break;
                }
                case VOP_INTADD: add_mod(s, pa, s.mont + b * k, d); break;
                case VOP_INTMUL: mont_mul(s, pa, s.mont + b * k, d); break;
                case VOP_SELECT: {
                    const uint32_t *src = flags[flag] ? pa : pb;
                    for (int j = 0; j < k; j++) d[j] = src[j];
                    break;
                }
                case VOP_SAVE_VAR:
                    // a = var index, b = source slot; dst = var slot
                    mont_mul(s, pb, one, var_canon + a * k);
                    for (int j = 0; j < k; j++) d[j] = pb[j];
                    break;
            }
        }

        // ---- trace columns: is_valid, inputs, vars ----
        size_t col = 0;
        auto put = [&](uint32_t v) { row_slice[col] = Fp(v); col++; };
        put(is_dummy ? 0u : 1u);
        for (int i = 0; i < s.num_input * nl; i++)
            put(is_dummy ? 0u : (uint32_t)in_limbs[i]);
        for (int v = 0; v < s.num_vars; v++)
            for (int i = 0; i < nl; i++) {
                uint32_t limb = (var_canon[v * k + i / 4] >> ((i % 4) * 8)) & 0xff;
                put(limb);
                if (!is_dummy) {
                    uint32_t idx = (1u << lb) + limb - 1;
                    atomicAdd(&rc_count[idx], 1u);
                }
            }

        // ---- constraint phase ----
        size_t carry_col = col;
        // q columns come first (all constraints), then carry columns.
        for (int ci = 0; ci < s.n_cons; ci++) carry_col += (s.cons + 8 * ci)[4];
        for (int ci = 0; ci < s.n_cons; ci++) {
            const uint32_t *c = s.cons + 8 * ci;
            const uint32_t tape_start = c[0], tape_len = c[1], res_off = c[2], res_len = c[3];
            const uint32_t q_limbs_n = c[4], carry_limbs_n = c[5];
            const uint32_t carry_min_abs = c[6], carry_bits = c[7];

            for (uint32_t io = 0; io < tape_len; io++) {
                const uint32_t *op = s.lops + 9 * (tape_start + io);
                const uint32_t opc = op[0], flag = op[1];
                const uint32_t d = op[2], dl = op[3], ao = op[4], al = op[5], bo = op[6], bl = op[7];
                const int32_t imm = (int32_t)op[8];
                switch (opc) {
                    case LOP_INPUT:
                        for (uint32_t i = 0; i < dl; i++)
                            scratch[d + i] = is_dummy ? 0 : (int32_t)in_limbs[ao * nl + i];
                        break;
                    case LOP_VAR:
                        for (uint32_t i = 0; i < dl; i++)
                            scratch[d + i] =
                                (int32_t)((var_canon[ao * k + i / 4] >> ((i % 4) * 8)) & 0xff);
                        break;
                    case LOP_CONST:
                        for (uint32_t i = 0; i < dl; i++)
                            scratch[d + i] = (int32_t)s.climbs[ao + i];
                        break;
                    case LOP_ADD:
                    case LOP_SUB:
                        for (uint32_t i = 0; i < dl; i++) {
                            int32_t a = i < al ? scratch[ao + i] : 0;
                            int32_t b = i < bl ? scratch[bo + i] : 0;
                            scratch[d + i] = opc == LOP_ADD ? a + b : a - b;
                        }
                        break;
                    case LOP_MUL:
                        for (int32_t i = (int32_t)dl - 1; i >= 0; i--) {
                            int64_t acc = 0;
                            int32_t lo = i - (int32_t)bl + 1 < 0 ? 0 : i - (int32_t)bl + 1;
                            int32_t hi = i < (int32_t)al - 1 ? i : (int32_t)al - 1;
                            for (int32_t j = lo; j <= hi; j++)
                                acc += (int64_t)scratch[ao + j] * scratch[bo + (i - j)];
                            // NOTE: bounds guaranteed by builder's max_carry_bits (< 2^31)
                            scratch[d + i] = (int32_t)acc;
                        }
                        break;
                    case LOP_INTADD:
                        for (uint32_t i = 0; i < dl; i++) scratch[d + i] = scratch[ao + i];
                        scratch[d] += imm;
                        break;
                    case LOP_INTMUL:
                        for (uint32_t i = 0; i < dl; i++)
                            scratch[d + i] = scratch[ao + i] * imm;
                        break;
                    case LOP_SELECT: {
                        const uint32_t src = flags[flag] ? ao : bo;
                        const uint32_t sl = flags[flag] ? al : bl;
                        for (uint32_t i = 0; i < dl; i++)
                            scratch[d + i] = i < sl ? scratch[src + i] : 0;
                        break;
                    }
                }
            }

            // N mod 2^(64K) from result limbs
            for (int i = 0; i < 2 * k; i++) nacc[i] = 0;
            for (uint32_t i = 0; i < res_len; i++) {
                int64_t v = scratch[res_off + i];
                if (v == 0) continue;
                uint64_t mag = v < 0 ? (uint64_t)(-v) : (uint64_t)v;
                int word = i / 4, shift = (i % 4) * 8;
                // mag < 2^63; shifted spans at most 3 words
                uint64_t lo64 = mag << shift; // low 64 of the shifted value
                uint64_t hi64 = shift ? (mag >> (64 - shift)) : 0;
                uint32_t parts[3] = {(uint32_t)lo64, (uint32_t)(lo64 >> 32), (uint32_t)hi64};
                if (v > 0) {
                    uint64_t carry = 0;
                    for (int w = 0; w < 2 * k - word; w++) {
                        uint64_t add = (w < 3 ? parts[w] : 0) + carry;
                        if (w >= 3 && add == 0) break;
                        uint64_t cur = (uint64_t)nacc[word + w] + add;
                        nacc[word + w] = (uint32_t)cur;
                        carry = cur >> 32;
                    }
                } else {
                    int64_t borrow = 0;
                    for (int w = 0; w < 2 * k - word; w++) {
                        int64_t sub = (int64_t)(w < 3 ? parts[w] : 0) + borrow;
                        if (w >= 3 && sub == 0) break;
                        int64_t cur = (int64_t)nacc[word + w] - sub;
                        nacc[word + w] = (uint32_t)cur;
                        borrow = cur < 0 ? 1 : 0;
                    }
                }
            }
            // q = N * pinv mod 2^(64K)
            for (int i = 0; i < 2 * k; i++) q512[i] = 0;
            for (int i = 0; i < 2 * k; i++) {
                if (nacc[i] == 0) continue;
                uint64_t carry = 0;
                for (int j = 0; j < 2 * k - i; j++) {
                    uint64_t prod = (uint64_t)nacc[i] * s.pinv[j];
                    uint64_t cur = (uint64_t)q512[i + j] + (prod & 0xffffffffu) + carry;
                    q512[i + j] = (uint32_t)cur;
                    carry = (cur >> 32) + (prod >> 32);
                }
            }
            // signed q limbs
            bool neg = (q512[2 * k - 1] >> 31) != 0;
            if (neg) {
                uint64_t carry = 1;
                for (int w = 0; w < 2 * k; w++) {
                    uint64_t cur = (uint64_t)(~q512[w]) + carry;
                    q512[w] = (uint32_t)cur;
                    carry = cur >> 32;
                }
            }
            int32_t ql[80];
            for (uint32_t i = 0; i < q_limbs_n; i++) {
                int32_t byte = (int32_t)((q512[i / 4] >> ((i % 4) * 8)) & 0xff);
                ql[i] = neg ? -byte : byte;
                put(f_of_i64(ql[i]));
                if (!is_dummy)
                    atomicAdd(&rc_count[(1u << (lb + 1)) + (uint32_t)(ql[i] + (1 << lb)) - 1], 1u);
            }
            // carries of expr - q*p, streamed
            int64_t carry_acc = 0;
            for (uint32_t i = 0; i < carry_limbs_n; i++) {
                int64_t e = i < res_len ? (int64_t)scratch[res_off + i] : 0;
                int32_t lo = (int32_t)i - s.p8_len + 1 < 0 ? 0 : (int32_t)i - s.p8_len + 1;
                int32_t hi = i < q_limbs_n - 1 ? (int32_t)i : (int32_t)q_limbs_n - 1;
                for (int32_t j = lo; j <= hi; j++)
                    e -= (int64_t)ql[j] * (int32_t)s.p8[i - j];
                carry_acc = (e + carry_acc) >> lb;
                row_slice[carry_col] = Fp(f_of_i64(carry_acc));
                carry_col++;
                if (!is_dummy)
                    atomicAdd(&rc_count[(1u << carry_bits) +
                                        (uint32_t)(carry_acc + carry_min_abs) - 1], 1u);
            }
        }

        // flags columns (after carries)
        for (int f = 0; f < s.num_flags; f++)
            row_slice[carry_col + f] = Fp((!is_dummy && flags[f]) ? 1u : 0u);

        // sanity: total width
        if (carry_col + s.num_flags != (size_t)s.width) atomicAdd(err, 1u);
    }
}


extern "C" int _field_expr_tracegen(
    Fp *d_trace_core,
    size_t height,
    size_t rows_used,
    const uint32_t *d_blob,
    const uint8_t *d_records,
    size_t rec_stride,
    size_t rec_core_offset,
    uint32_t *d_range_checker,
    uint32_t *d_aux,
    size_t aux_words,
    int should_finalize,
    uint32_t *d_err,
    cudaStream_t stream) {
    const int threads = 256;
    // Grid-stride with a capped grid bounds the aux scratch VRAM (see Rust side).
    const size_t want = (height + threads - 1) / threads;
    const int blocks = (int)(want < 512 ? want : 512);
    field_expr_tracegen<<<blocks, threads, 0, stream>>>(
        d_blob, d_records, rec_stride, rec_core_offset, rows_used, height, d_trace_core,
        d_range_checker, d_aux, aux_words, should_finalize, d_err);
    return CHECK_KERNEL();
}
