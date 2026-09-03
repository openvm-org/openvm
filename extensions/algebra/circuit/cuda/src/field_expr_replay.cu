// Generic GPU tracegen for mod-builder FieldExpr chips.
// Interprets the "device program" blob produced by device_program.rs (see that
// file for the semantics contract). One thread per row, grid-stride.
//
// The core-column interpreter is validated bit-exact against
// FieldExpressionFiller::fill_trace_row (rows and range-checker histograms) on
// EcAddNe, MulDiv (flags/Select/Div-under-Select/setup rows) and IntMul/IntAdd
// expressions. Adapter columns use the shared VecHeapAdapter device fill.
#include "algebra/vec_heap_replay.cuh"
#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

#include <cstddef>
#include <cstdint>

static constexpr uint32_t FIELD_EXPR_BAD_BLOB = 0x4d030002;
static constexpr uint32_t FIELD_EXPR_BAD_OPCODE_OR_SETUP = 0x4d030003;
static constexpr uint32_t FIELD_EXPR_ACTIVE_ZERO_DIVISOR = 0x4d030004;
static constexpr uint32_t FIELD_EXPR_OUTPUT_MISMATCH = 0x4d030005;
static constexpr uint32_t FIELD_EXPR_BAD_PROGRAM_OP = 0x4d030006;
static constexpr uint32_t FIELD_EXPR_BAD_TRACE_SHAPE = 0x4d030007;
static constexpr uint32_t BABYBEAR_MODULUS = 0x78000001;
static constexpr uint32_t NO_FLAG = UINT32_MAX;
static constexpr size_t FIELD_EXPR_HEADER_WORDS = 34;
static constexpr uint32_t MAX_U32_LIMBS = 12;

// Evaluation opcodes compute canonical field values.
enum { VOP_LOAD_INPUT = 0, VOP_CONST, VOP_ADD, VOP_SUB, VOP_MUL, VOP_DIV,
       VOP_INTADD, VOP_INTMUL, VOP_SELECT, VOP_SAVE_VAR, VOP_LOAD_OUTPUT };
// Witness opcodes evaluate limb expressions used by the constraints.
enum { LOP_INPUT = 0, LOP_VAR, LOP_CONST, LOP_ADD, LOP_SUB, LOP_MUL,
       LOP_INTADD, LOP_INTMUL, LOP_SELECT };

// Header word indices. This is an internal host/device ABI, not a file format.
enum {
    H_NUM_LIMBS = 0,
    H_LIMB_BITS,
    H_K,
    H_NUM_INPUT,
    H_NUM_VARS,
    H_NUM_FLAGS,
    H_NEEDS_SETUP,
    H_SHOULD_FINALIZE,
    H_WIDTH,
    H_NUM_SLOTS,
    H_N_VOPS,
    H_N_LOPS,
    H_N_CONS,
    H_SCRATCH_LEN,
    H_P8_LEN,
    H_N_OPCODE_METADATA,
    H_N_SETUP_VALUES,
    H_N_OUTPUTS,
    H_MAX_Q_LIMBS,
    H_AUX_WORDS,
    H_OFF_VOPS,
    H_OFF_LOPS,
    H_OFF_CONS,
    H_OFF_P,
    H_OFF_R2,
    H_OFF_PM2,
    H_OFF_PINV,
    H_OFF_P8,
    H_OFF_MONT,
    H_OFF_CLIMBS,
    H_OFF_OPTAB,
    H_OFF_SETUP_VALUES,
    H_OFF_OUTPUTS,
    H_MPRIME,
};

struct FieldExprProg {
    uint32_t num_limbs;
    uint32_t limb_bits;
    uint32_t k;
    uint32_t num_input;
    uint32_t num_vars;
    uint32_t num_flags;
    uint32_t needs_setup;
    uint32_t should_finalize;
    uint32_t width;
    uint32_t num_slots;
    uint32_t n_vops;
    uint32_t n_lops;
    uint32_t n_cons;
    uint32_t scratch_len;
    uint32_t p8_len;
    uint32_t n_opcode_metadata;
    uint32_t n_setup_values;
    uint32_t n_outputs;
    uint32_t max_q_limbs;
    uint32_t aux_words;
    uint32_t mprime;
    const uint32_t *vops;
    const uint32_t *lops;
    const uint32_t *cons;
    const uint32_t *p;
    const uint32_t *r2;
    const uint32_t *pm2;
    const uint32_t *pinv;
    const uint32_t *p8;
    const uint32_t *mont;
    const uint32_t *climbs;
    const uint32_t *optab;
    const uint32_t *setup_values;
    const uint32_t *outputs;
    const uint32_t *dummy_outputs;
};

__device__ __forceinline__ void load_prog(
    const uint32_t *blob, FieldExprProg &s
) {
    s.num_limbs = blob[H_NUM_LIMBS];
    s.limb_bits = blob[H_LIMB_BITS];
    s.k = blob[H_K];
    s.num_input = blob[H_NUM_INPUT];
    s.num_vars = blob[H_NUM_VARS];
    s.num_flags = blob[H_NUM_FLAGS];
    s.needs_setup = blob[H_NEEDS_SETUP];
    s.should_finalize = blob[H_SHOULD_FINALIZE];
    s.width = blob[H_WIDTH];
    s.num_slots = blob[H_NUM_SLOTS];
    s.n_vops = blob[H_N_VOPS];
    s.n_lops = blob[H_N_LOPS];
    s.n_cons = blob[H_N_CONS];
    s.scratch_len = blob[H_SCRATCH_LEN];
    s.p8_len = blob[H_P8_LEN];
    s.n_opcode_metadata = blob[H_N_OPCODE_METADATA];
    s.n_setup_values = blob[H_N_SETUP_VALUES];
    s.n_outputs = blob[H_N_OUTPUTS];
    s.max_q_limbs = blob[H_MAX_Q_LIMBS];
    s.aux_words = blob[H_AUX_WORDS];
    s.mprime = blob[H_MPRIME];
    s.vops = blob + blob[H_OFF_VOPS];
    s.lops = blob + blob[H_OFF_LOPS];
    s.cons = blob + blob[H_OFF_CONS];
    s.p = blob + blob[H_OFF_P];
    s.r2 = blob + blob[H_OFF_R2];
    s.pm2 = blob + blob[H_OFF_PM2];
    s.pinv = blob + blob[H_OFF_PINV];
    s.p8 = blob + blob[H_OFF_P8];
    s.mont = blob + blob[H_OFF_MONT];
    s.climbs = blob + blob[H_OFF_CLIMBS];
    s.optab = blob + blob[H_OFF_OPTAB];
    s.setup_values = blob + blob[H_OFF_SETUP_VALUES];
    s.outputs = blob + blob[H_OFF_OUTPUTS];
    s.dummy_outputs = s.outputs + s.n_outputs;
}

static __device__ bool checked_segment(
    uint32_t offset, uint32_t count, uint32_t stride, size_t blob_words
) {
    uint64_t end = static_cast<uint64_t>(offset) +
                   static_cast<uint64_t>(count) * static_cast<uint64_t>(stride);
    return offset >= FIELD_EXPR_HEADER_WORDS && end <= blob_words;
}

static __device__ bool range_within(uint32_t offset, uint32_t len, uint32_t limit) {
    return static_cast<uint64_t>(offset) + len <= limit;
}

static __device__ bool validate_and_load_prog(
    const uint32_t *blob, size_t blob_words, FieldExprProg &s
) {
    if (blob == nullptr || blob_words < FIELD_EXPR_HEADER_WORDS) return false;
    load_prog(blob, s);
    if (s.limb_bits != 8 || s.k == 0 || s.k > MAX_U32_LIMBS || s.num_limbs == 0 ||
        s.num_limbs > 4 * s.k || s.num_flags > 32 || s.needs_setup > 1 ||
        s.should_finalize > 1 ||
        static_cast<uint64_t>(s.num_slots) <
            static_cast<uint64_t>(s.num_input) + s.num_vars ||
        s.aux_words == 0 || s.p8_len != s.num_limbs ||
        static_cast<uint64_t>(s.max_q_limbs) * 8 + 1 >=
            static_cast<uint64_t>(64) * s.k) {
        return false;
    }
    const uint32_t *offsets = blob + H_OFF_VOPS;
    for (size_t i = 0; i + 1 < H_MPRIME - H_OFF_VOPS; i++) {
        if (offsets[i] > offsets[i + 1]) return false;
    }
    if (blob[H_OFF_VOPS] != FIELD_EXPR_HEADER_WORDS ||
        !checked_segment(blob[H_OFF_VOPS], s.n_vops, 7, blob_words) ||
        static_cast<uint64_t>(blob[H_OFF_LOPS]) !=
            static_cast<uint64_t>(blob[H_OFF_VOPS]) + 7 * s.n_vops ||
        !checked_segment(blob[H_OFF_LOPS], s.n_lops, 9, blob_words) ||
        static_cast<uint64_t>(blob[H_OFF_CONS]) !=
            static_cast<uint64_t>(blob[H_OFF_LOPS]) + 9 * s.n_lops ||
        !checked_segment(blob[H_OFF_CONS], s.n_cons, 8, blob_words) ||
        static_cast<uint64_t>(blob[H_OFF_P]) !=
            static_cast<uint64_t>(blob[H_OFF_CONS]) + 8 * s.n_cons ||
        static_cast<uint64_t>(blob[H_OFF_R2]) !=
            static_cast<uint64_t>(blob[H_OFF_P]) + s.k ||
        static_cast<uint64_t>(blob[H_OFF_PM2]) !=
            static_cast<uint64_t>(blob[H_OFF_R2]) + s.k ||
        static_cast<uint64_t>(blob[H_OFF_PINV]) !=
            static_cast<uint64_t>(blob[H_OFF_PM2]) + s.k ||
        static_cast<uint64_t>(blob[H_OFF_P8]) !=
            static_cast<uint64_t>(blob[H_OFF_PINV]) + 2 * s.k ||
        static_cast<uint64_t>(blob[H_OFF_MONT]) !=
            static_cast<uint64_t>(blob[H_OFF_P8]) + s.p8_len ||
        static_cast<uint64_t>(blob[H_OFF_SETUP_VALUES]) !=
            static_cast<uint64_t>(blob[H_OFF_OPTAB]) +
                2 * s.n_opcode_metadata ||
        static_cast<uint64_t>(blob[H_OFF_OUTPUTS]) !=
            static_cast<uint64_t>(blob[H_OFF_SETUP_VALUES]) +
                static_cast<uint64_t>(s.n_setup_values) * s.num_limbs ||
        static_cast<uint64_t>(blob[H_OFF_OUTPUTS]) + s.n_outputs +
                static_cast<uint64_t>(s.should_finalize) * s.n_outputs * s.k !=
            blob_words) {
        return false;
    }
    if ((s.p[0] & 1) == 0) return false;
    uint64_t value_workspace =
        static_cast<uint64_t>(s.num_slots) * s.k + 3 * s.k + 2;
    uint64_t constraint_workspace =
        static_cast<uint64_t>(s.scratch_len) + 4 * s.k + s.max_q_limbs;
    uint64_t expected_aux = static_cast<uint64_t>(s.num_vars) * s.k +
                            (value_workspace > constraint_workspace
                                 ? value_workspace
                                 : constraint_workspace);
    if (expected_aux != s.aux_words) return false;

    size_t mont_words = blob[H_OFF_CLIMBS] - blob[H_OFF_MONT];
    size_t constant_words = blob[H_OFF_OPTAB] - blob[H_OFF_CLIMBS];
    if (mont_words % s.k != 0) return false;
    size_t mont_values = mont_words / s.k;
    uint32_t valid_flag_mask =
        s.num_flags == 32 ? UINT32_MAX : (uint32_t(1) << s.num_flags) - 1;
    for (uint32_t index = 0; index < s.n_vops; index++) {
        const uint32_t *op = s.vops + 7 * index;
        uint32_t code = op[0], flag = op[1], guard_true = op[2], guard_false = op[3];
        uint32_t dst = op[4], a = op[5], b = op[6];
        if (code > VOP_LOAD_OUTPUT || dst >= s.num_slots ||
            ((guard_true | guard_false) & ~valid_flag_mask) != 0) {
            return false;
        }
        bool slots_a = code == VOP_ADD || code == VOP_SUB || code == VOP_MUL ||
                       code == VOP_DIV || code == VOP_SELECT;
        bool slots_b = slots_a || code == VOP_SAVE_VAR;
        if ((slots_a && a >= s.num_slots) || (slots_b && b >= s.num_slots) ||
            (code == VOP_LOAD_INPUT && a >= s.num_input) ||
            (code == VOP_CONST && a >= mont_values) ||
            ((code == VOP_INTADD || code == VOP_INTMUL) &&
             (a >= s.num_slots || b >= mont_values)) ||
            (code == VOP_SELECT && flag >= s.num_flags) ||
            (code == VOP_SAVE_VAR && a >= s.num_vars) ||
            (code == VOP_LOAD_OUTPUT && a >= s.n_outputs)) {
            return false;
        }
    }
    for (uint32_t index = 0; index < s.n_lops; index++) {
        const uint32_t *op = s.lops + 9 * index;
        uint32_t code = op[0], flag = op[1], dst = op[2], dst_len = op[3];
        uint32_t a = op[4], a_len = op[5], b = op[6], b_len = op[7];
        if (code > LOP_SELECT || !range_within(dst, dst_len, s.scratch_len)) return false;
        bool a_scratch = code == LOP_ADD || code == LOP_SUB || code == LOP_MUL ||
                         code == LOP_SELECT;
        bool b_scratch =
            code == LOP_ADD || code == LOP_SUB || code == LOP_MUL ||
            code == LOP_SELECT;
        if ((a_scratch && !range_within(a, a_len, s.scratch_len)) ||
            (b_scratch && !range_within(b, b_len, s.scratch_len)) ||
            ((code == LOP_INTADD || code == LOP_INTMUL) &&
             (dst_len == 0 || dst_len != a_len ||
              !range_within(a, dst_len, s.scratch_len))) ||
            (code == LOP_INPUT &&
             (a >= s.num_input || dst_len > s.num_limbs)) ||
            (code == LOP_VAR && (a >= s.num_vars || dst_len > s.num_limbs)) ||
            (code == LOP_CONST &&
             static_cast<uint64_t>(a) + dst_len > constant_words) ||
            (code == LOP_SELECT && flag >= s.num_flags)) {
            return false;
        }
    }
    uint64_t expected_width =
        1 + static_cast<uint64_t>(s.num_input) * s.num_limbs +
        static_cast<uint64_t>(s.num_vars) * s.num_limbs + s.num_flags;
    for (uint32_t index = 0; index < s.n_cons; index++) {
        const uint32_t *constraint = s.cons + 8 * index;
        uint32_t tape_start = constraint[0], tape_len = constraint[1];
        uint32_t result = constraint[2], result_len = constraint[3];
        uint32_t q_limbs = constraint[4], carry_limbs = constraint[5];
        uint32_t carry_bits = constraint[7];
        if (static_cast<uint64_t>(tape_start) + tape_len > s.n_lops ||
            !range_within(result, result_len, s.scratch_len) ||
            q_limbs > s.max_q_limbs ||
            carry_bits > 30) {
            return false;
        }
        expected_width += static_cast<uint64_t>(q_limbs) + carry_limbs;
    }
    if (expected_width != s.width) return false;

    for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
        uint32_t expected = (s.p[byte / 4] >> (8 * (byte % 4))) & 0xff;
        if (s.p8[byte] != expected) return false;
    }
    if (s.n_opcode_metadata == 0 ||
        (s.needs_setup && s.n_setup_values >= s.num_input) ||
        (!s.needs_setup &&
         (s.num_flags != 0 || s.n_opcode_metadata != 1 || s.n_setup_values != 0))) {
        return false;
    }
    uint32_t no_flag_count = 0;
    uint32_t seen_flags = 0;
    for (uint32_t index = 0; index < s.n_opcode_metadata; index++) {
        uint32_t opcode = s.optab[2 * index];
        uint32_t flag = s.optab[2 * index + 1];
        for (uint32_t previous = 0; previous < index; previous++) {
            if (s.optab[2 * previous] == opcode) return false;
        }
        if (flag == NO_FLAG) {
            no_flag_count++;
        } else {
            if (flag >= s.num_flags || (seen_flags & (uint32_t(1) << flag)) != 0) {
                return false;
            }
            seen_flags |= uint32_t(1) << flag;
        }
    }
    if (no_flag_count != 1) return false;
    for (uint32_t index = 0; index < s.n_outputs; index++) {
        if (s.outputs[index] >= s.num_vars) return false;
    }
    return true;
}

static __device__ __forceinline__ uint32_t sub_u32_limbs(
    const uint32_t *lhs,
    const uint32_t *rhs,
    uint32_t *output,
    uint32_t limbs
) {
    uint32_t borrow = 0;
    for (uint32_t index = 0; index < limbs; index++) {
        uint64_t subtrahend = static_cast<uint64_t>(rhs[index]) + borrow;
        uint32_t lhs_word = lhs[index];
        output[index] = lhs_word - static_cast<uint32_t>(subtrahend);
        borrow = static_cast<uint64_t>(lhs_word) < subtrahend;
    }
    return borrow;
}

// The serializer reserves 3*k+2 value-workspace words: 2*k+2 for Montgomery
// reduction and k for a canonical input, inverse, or the integer one.
template <uint32_t K>
static __device__ void mont_mul(
    const FieldExprProg &s,
    const uint32_t *a,
    const uint32_t *b,
    uint32_t *r,
    uint32_t *workspace
) {
    uint32_t *t = workspace;
    uint32_t *sub = workspace + K + 2;
    constexpr uint32_t k = K;
    for (uint32_t i = 0; i < k + 2; i++) t[i] = 0;
    for (uint32_t i = 0; i < k; i++) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < k; j++) {
            uint64_t cur = static_cast<uint64_t>(t[j]) +
                           static_cast<uint64_t>(a[i]) * b[j] + carry;
            t[j] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        uint64_t cur = static_cast<uint64_t>(t[k]) + carry;
        t[k] = static_cast<uint32_t>(cur);
        t[k + 1] = static_cast<uint32_t>(cur >> 32);
        uint32_t m = t[0] * s.mprime;
        carry =
            (static_cast<uint64_t>(t[0]) + static_cast<uint64_t>(m) * s.p[0]) >> 32;
        for (uint32_t j = 1; j < k; j++) {
            uint64_t cur2 = static_cast<uint64_t>(t[j]) +
                            static_cast<uint64_t>(m) * s.p[j] + carry;
            t[j - 1] = static_cast<uint32_t>(cur2);
            carry = cur2 >> 32;
        }
        uint64_t cur3 = static_cast<uint64_t>(t[k]) + carry;
        t[k - 1] = static_cast<uint32_t>(cur3);
        t[k] = t[k + 1] + static_cast<uint32_t>(cur3 >> 32);
        t[k + 1] = 0;
    }
    uint32_t borrow = sub_u32_limbs(t, s.p, sub, k);
    bool ge = (t[k] != 0) || !borrow;
    for (uint32_t j = 0; j < k; j++) r[j] = ge ? sub[j] : t[j];
}

template <uint32_t K>
static __device__ void add_mod(
    const FieldExprProg &s,
    const uint32_t *a,
    const uint32_t *b,
    uint32_t *r,
    uint32_t *workspace
) {
    constexpr uint32_t k = K;
    uint32_t *sum = workspace;
    uint32_t *sub = workspace + k;
    uint64_t carry = 0;
    for (uint32_t j = 0; j < k; j++) {
        uint64_t cur = static_cast<uint64_t>(a[j]) + b[j] + carry;
        sum[j] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
    uint32_t borrow = sub_u32_limbs(sum, s.p, sub, k);
    bool ge = carry || !borrow;
    for (uint32_t j = 0; j < k; j++) r[j] = ge ? sub[j] : sum[j];
}

template <uint32_t K>
static __device__ void sub_mod(
    const FieldExprProg &s, const uint32_t *a, const uint32_t *b, uint32_t *r
) {
    constexpr uint32_t k = K;
    uint32_t borrow = sub_u32_limbs(a, b, r, k);
    if (borrow) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < k; j++) {
            uint64_t cur = static_cast<uint64_t>(r[j]) + s.p[j] + carry;
            r[j] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
    }
}

static __device__ bool limbs_are_zero(const uint32_t *value, uint32_t len) {
    uint32_t aggregate = 0;
    for (uint32_t index = 0; index < len; index++) aggregate |= value[index];
    return aggregate == 0;
}

template <uint32_t K>
static __device__ void mont_inv(
    const FieldExprProg &s,
    const uint32_t *a,
    uint32_t *r,
    uint32_t *mont_workspace
) {
    constexpr uint32_t k = K;
    bool started = false;
    for (int bit = static_cast<int>(32 * k) - 1; bit >= 0; bit--) {
        if (started) mont_mul<K>(s, r, r, r, mont_workspace);
        if ((s.pm2[bit / 32] >> (bit % 32)) & 1) {
            if (!started) {
                for (uint32_t j = 0; j < k; j++) r[j] = a[j];
                started = true;
            } else {
                mont_mul<K>(s, r, a, r, mont_workspace);
            }
        }
    }
    if (!started) {
        for (uint32_t j = 0; j < k; j++) r[j] = 0;
    }
}

static __device__ __forceinline__ void put_core_value(
    RowSlice row, size_t &column, uint32_t value
) {
    row[column++] = Fp(value);
}

__device__ __forceinline__ uint32_t f_of_i64(int64_t v) {
    int64_t m = v % static_cast<int64_t>(BABYBEAR_MODULUS);
    if (m < 0) m += BABYBEAR_MODULUS;
    return static_cast<uint32_t>(m);
}

// Fill the core sub-row; CUDA tests compare it against FieldExpressionFiller.
// `core_row` must point at the first core column. When `is_dummy`, inputs are zero,
// flags false, range checks skipped, is_valid = 0 (mirrors fill_dummy_trace_row).
template <uint32_t K>
static __device__ bool field_expr_fill_core_row(
    const FieldExprProg &s,
    RowSlice core_row,
    const uint8_t *in_limbs,
    const uint8_t *logged_output,
    uint32_t opcode,
    VariableRangeChecker rc,
    uint32_t *my_aux,
    bool is_dummy,
    uint32_t *err) {
    constexpr uint32_t k = K;
    const uint32_t nl = s.num_limbs, lb = s.limb_bits;
    uint32_t *var_canon = my_aux; // num_vars * k, retained for the witness phase
    uint32_t *workspace = var_canon + s.num_vars * k;
    uint32_t *slots = workspace; // num_slots * k
    uint32_t *mont_workspace = slots + s.num_slots * k; // 2k+2
    uint32_t *value_extra = mont_workspace + 2 * k + 2; // k
    int32_t *scratch = reinterpret_cast<int32_t *>(workspace); // scratch_len
    uint32_t *nacc = reinterpret_cast<uint32_t *>(scratch + s.scratch_len); // 2k
    uint32_t *q512 = nacc + 2 * k;                              // 2k

    uint32_t flags = 0;
    bool opcode_found = is_dummy;
    bool is_setup = false;
    if (!is_dummy) {
        for (uint32_t index = 0; index < s.n_opcode_metadata; index++) {
            const uint32_t *metadata = s.optab + 2 * index;
            if (metadata[0] == opcode) {
                opcode_found = true;
                is_setup = s.needs_setup && metadata[1] == NO_FLAG;
                if (metadata[1] != NO_FLAG) flags |= uint32_t(1) << metadata[1];
                break;
            }
        }
        if (!opcode_found) {
            preflight_set_error(err, FIELD_EXPR_BAD_OPCODE_OR_SETUP);
            return false;
        }
        if (is_setup) {
            for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
                uint8_t expected =
                    static_cast<uint8_t>(s.p[byte / 4] >> (8 * (byte % 4)));
                if (in_limbs[byte] != expected) {
                    preflight_set_error(err, FIELD_EXPR_BAD_OPCODE_OR_SETUP);
                    return false;
                }
            }
            for (uint32_t value = 0; value < s.n_setup_values; value++) {
                for (uint32_t byte = 0; byte < s.num_limbs; byte++) {
                    size_t input_offset =
                        static_cast<size_t>(value + 1) * s.num_limbs + byte;
                    if (in_limbs[input_offset] !=
                        static_cast<uint8_t>(
                            s.setup_values[value * s.num_limbs + byte]
                        )) {
                        preflight_set_error(err, FIELD_EXPR_BAD_OPCODE_OR_SETUP);
                        return false;
                    }
                }
            }
        }
    }

    // Evaluation computes canonical field values and stores the variables written to the trace.
    for (uint32_t i = 0; i < s.num_slots * k; i++) slots[i] = 0;
    for (uint32_t io = 0; io < s.n_vops; io++) {
        const uint32_t *op = s.vops + 7 * io;
        const uint32_t opc = op[0], flag = op[1], guard_true = op[2];
        const uint32_t guard_false = op[3], dst = op[4], a = op[5], b = op[6];
        if ((flags & guard_true) != guard_true || (flags & guard_false) != 0) continue;
        uint32_t *d = slots + dst * k;
        switch (opc) {
            case VOP_LOAD_INPUT: {
                for (uint32_t j = 0; j < k; j++) value_extra[j] = 0;
                if (!is_dummy) {
                    const uint8_t *src = in_limbs + a * nl;
                    for (uint32_t i = 0; i < nl; i++) {
                        value_extra[i * lb / 32] |=
                            static_cast<uint32_t>(src[i]) << ((i * lb) % 32);
                    }
                }
                mont_mul<K>(s, value_extra, s.r2, d, mont_workspace);
                break;
            }
            case VOP_CONST: {
                for (uint32_t j = 0; j < k; j++) d[j] = s.mont[a * k + j];
                break;
            }
            case VOP_ADD:
                add_mod<K>(s, slots + a * k, slots + b * k, d, mont_workspace);
                break;
            case VOP_SUB:
                sub_mod<K>(s, slots + a * k, slots + b * k, d);
                break;
            case VOP_MUL:
                mont_mul<K>(s, slots + a * k, slots + b * k, d, mont_workspace);
                break;
            case VOP_DIV: {
                const uint32_t *pa = slots + a * k;
                const uint32_t *pb = slots + b * k;
                if (limbs_are_zero(pb, k)) {
                    preflight_set_error(err, FIELD_EXPR_ACTIVE_ZERO_DIVISOR);
                    return false;
                }
                mont_inv<K>(s, pb, value_extra, mont_workspace);
                mont_mul<K>(s, pa, value_extra, d, mont_workspace);
                break;
            }
            case VOP_INTADD:
                add_mod<K>(s, slots + a * k, s.mont + b * k, d, mont_workspace);
                break;
            case VOP_INTMUL:
                mont_mul<K>(s, slots + a * k, s.mont + b * k, d, mont_workspace);
                break;
            case VOP_SELECT: {
                const uint32_t *src = (flags & (uint32_t(1) << flag)) != 0
                                          ? slots + a * k
                                          : slots + b * k;
                for (uint32_t j = 0; j < k; j++) d[j] = src[j];
                break;
            }
            case VOP_SAVE_VAR: {
                const uint32_t *pb = slots + b * k;
                for (uint32_t j = 0; j < k; j++) value_extra[j] = j == 0 ? 1 : 0;
                mont_mul<K>(
                    s, pb, value_extra, var_canon + a * k, mont_workspace
                );
                for (uint32_t j = 0; j < k; j++) d[j] = pb[j];
                break;
            }
            case VOP_LOAD_OUTPUT: {
                if (is_dummy) {
                    for (uint32_t j = 0; j < k; j++) {
                        value_extra[j] = s.dummy_outputs[a * k + j];
                    }
                } else {
                    for (uint32_t j = 0; j < k; j++) value_extra[j] = 0;
                    const uint8_t *src = logged_output + a * nl;
                    for (uint32_t i = 0; i < nl; i++) {
                        value_extra[i * lb / 32] |=
                            static_cast<uint32_t>(src[i]) << ((i * lb) % 32);
                    }
                    if (sub_u32_limbs(value_extra, s.p, d, k) == 0) {
                        preflight_set_error(err, FIELD_EXPR_OUTPUT_MISMATCH);
                        return false;
                    }
                }
                mont_mul<K>(s, value_extra, s.r2, d, mont_workspace);
                break;
            }
            default:
                preflight_set_error(err, FIELD_EXPR_BAD_PROGRAM_OP);
                return false;
        }
    }

    if (!is_dummy) {
        size_t output_byte = 0;
        for (uint32_t output = 0; output < s.n_outputs; output++) {
            const uint32_t *value = var_canon + s.outputs[output] * k;
            for (uint32_t byte = 0; byte < s.num_limbs; byte++, output_byte++) {
                uint8_t expected =
                    static_cast<uint8_t>(value[byte / 4] >> (8 * (byte % 4)));
                if (logged_output[output_byte] != expected) {
                    preflight_set_error(err, FIELD_EXPR_OUTPUT_MISMATCH);
                    return false;
                }
            }
        }
    }

    // ---- trace columns: is_valid, inputs, vars ----
    size_t col = 0;
    put_core_value(core_row, col, is_dummy ? 0u : 1u);
    for (int i = 0; i < s.num_input * nl; i++)
        put_core_value(core_row, col, is_dummy ? 0u : static_cast<uint32_t>(in_limbs[i]));
    for (int v = 0; v < s.num_vars; v++)
        for (int i = 0; i < nl; i++) {
            uint32_t limb = (var_canon[v * k + i / 4] >> ((i % 4) * 8)) & 0xff;
            put_core_value(core_row, col, limb);
            if (!is_dummy) rc.add_count(limb, lb);
        }

    // Witness generation emits the quotients and carries that prove each limb constraint.
    size_t carry_col = col;
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
                    bool selected = (flags & (uint32_t(1) << flag)) != 0;
                    const uint32_t src = selected ? ao : bo;
                    const uint32_t sl = selected ? al : bl;
                    for (uint32_t i = 0; i < dl; i++)
                        scratch[d + i] = i < sl ? scratch[src + i] : 0;
                    break;
                }
                default:
                    preflight_set_error(err, FIELD_EXPR_BAD_PROGRAM_OP);
                    return false;
            }
        }

        // N mod 2^(64K) from result limbs
        for (int i = 0; i < 2 * k; i++) nacc[i] = 0;
        for (uint32_t i = 0; i < res_len; i++) {
            int64_t v = scratch[res_off + i];
            if (v == 0) continue;
            uint64_t mag = v < 0 ? (uint64_t)(-v) : (uint64_t)v;
            int word = i / 4, shift = (i % 4) * 8;
            uint64_t lo64 = mag << shift;
            uint64_t hi64 = shift ? (mag >> (64 - shift)) : 0;
            if (v > 0) {
                uint64_t carry = 0;
                for (int w = 0; w < 2 * k - word; w++) {
                    uint32_t part = w == 0   ? static_cast<uint32_t>(lo64)
                                    : w == 1 ? static_cast<uint32_t>(lo64 >> 32)
                                    : w == 2 ? static_cast<uint32_t>(hi64)
                                             : 0;
                    uint64_t add = static_cast<uint64_t>(part) + carry;
                    if (w >= 3 && add == 0) break;
                    uint64_t cur = (uint64_t)nacc[word + w] + add;
                    nacc[word + w] = (uint32_t)cur;
                    carry = cur >> 32;
                }
            } else {
                int64_t borrow = 0;
                for (int w = 0; w < 2 * k - word; w++) {
                    uint32_t part = w == 0   ? static_cast<uint32_t>(lo64)
                                    : w == 1 ? static_cast<uint32_t>(lo64 >> 32)
                                    : w == 2 ? static_cast<uint32_t>(hi64)
                                             : 0;
                    int64_t sub = static_cast<int64_t>(part) + borrow;
                    if (w >= 3 && sub == 0) break;
                    int64_t cur = (int64_t)nacc[word + w] - sub;
                    nacc[word + w] = (uint32_t)cur;
                    borrow = cur < 0 ? 1 : 0;
                }
            }
        }
        // q = N * pinv mod 2^(64K), exact division
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
        bool neg = (q512[2 * k - 1] >> 31) != 0;
        if (neg) {
            uint64_t carry = 1;
            for (int w = 0; w < 2 * k; w++) {
                uint64_t cur = (uint64_t)(~q512[w]) + carry;
                q512[w] = (uint32_t)cur;
                carry = cur >> 32;
            }
        }
        int32_t *ql = reinterpret_cast<int32_t *>(q512 + 2 * k);
        for (uint32_t i = 0; i < q_limbs_n; i++) {
            int32_t byte = (int32_t)((q512[i / 4] >> ((i % 4) * 8)) & 0xff);
            ql[i] = neg ? -byte : byte;
            put_core_value(core_row, col, f_of_i64(ql[i]));
            if (!is_dummy) rc.add_count((uint32_t)(ql[i] + (1 << lb)), lb + 1);
        }
        int64_t carry_acc = 0;
        for (uint32_t i = 0; i < carry_limbs_n; i++) {
            int64_t e = i < res_len ? (int64_t)scratch[res_off + i] : 0;
            int32_t lo_candidate =
                static_cast<int32_t>(i) - static_cast<int32_t>(s.p8_len) + 1;
            int32_t lo = lo_candidate < 0 ? 0 : lo_candidate;
            int32_t hi = static_cast<int32_t>(i) <
                                 static_cast<int32_t>(q_limbs_n) - 1
                             ? static_cast<int32_t>(i)
                             : static_cast<int32_t>(q_limbs_n) - 1;
            for (int32_t j = lo; j <= hi; j++)
                e -= (int64_t)ql[j] * (int32_t)s.p8[i - j];
            carry_acc = (e + carry_acc) >> lb;
            core_row[carry_col] = Fp(f_of_i64(carry_acc));
            carry_col++;
            if (!is_dummy) rc.add_count((uint32_t)(carry_acc + carry_min_abs), carry_bits);
        }
        if (!is_dummy && carry_acc != 0) {
            preflight_set_error(err, FIELD_EXPR_OUTPUT_MISMATCH);
            return false;
        }
    }
    for (uint32_t f = 0; f < s.num_flags; f++) {
        core_row[carry_col + f] =
            Fp((!is_dummy && (flags & (uint32_t(1) << f)) != 0) ? 1u : 0u);
    }
    if (carry_col + s.num_flags != static_cast<size_t>(s.width)) {
        preflight_set_error(err, FIELD_EXPR_BAD_TRACE_SHAPE);
        return false;
    }
    return true;
}

template <size_t NUM_READS, size_t BLOCKS>
static __global__ void validate_field_expr_replay(
    size_t height,
    size_t width,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    size_t aux_words,
    uint32_t *error
) {
    constexpr uint32_t K = BLOCKS <= 6 ? 2 * BLOCKS : BLOCKS;
    FieldExprProg s;
    if (!validate_and_load_prog(blob, blob_words, s)) {
        preflight_set_error(error, FIELD_EXPR_BAD_BLOB);
        return;
    }
    constexpr size_t ADAPTER_WIDTH =
        sizeof(VecHeapAdapterCols<uint8_t, NUM_READS, BLOCKS, BLOCKS>);
    constexpr size_t INPUT_BYTES = NUM_READS * BLOCKS * MEMORY_BLOCK_BYTES;
    constexpr size_t OUTPUT_BYTES = BLOCKS * MEMORY_BLOCK_BYTES;
    if (s.k != K || width != ADAPTER_WIDTH + s.width ||
        static_cast<uint64_t>(s.num_input) * s.num_limbs != INPUT_BYTES ||
        static_cast<uint64_t>(s.n_outputs) * s.num_limbs != OUTPUT_BYTES ||
        projection_len > height || aux_words != s.aux_words) {
        preflight_set_error(error, FIELD_EXPR_BAD_TRACE_SHAPE);
    }
}

template <size_t NUM_READS, size_t BLOCKS>
static __global__ void field_expr_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    const VecHeapTraceInput<NUM_READS, BLOCKS> *projection,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    uint32_t *range_delta,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error
) {
    constexpr uint32_t K = BLOCKS <= 6 ? 2 * BLOCKS : BLOCKS;
    constexpr size_t ADAPTER_WIDTH =
        sizeof(VecHeapAdapterCols<uint8_t, NUM_READS, BLOCKS, BLOCKS>);
    __shared__ FieldExprProg shared_program;
    if (threadIdx.x == 0) load_prog(blob, shared_program);
    __syncthreads();
    if (*error != 0) return;
    const FieldExprProg &s = shared_program;
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t nthreads = gridDim.x * (size_t)blockDim.x;
    if (nthreads == 0 ||
        static_cast<uint64_t>(nthreads) * aux_words > scratch_words) {
        preflight_set_error(error, FIELD_EXPR_BAD_TRACE_SHAPE);
        return;
    }
    VariableRangeChecker range_checker(range_delta, range_bins);
    uint32_t *thread_scratch = scratch + tid * aux_words;

    for (size_t row_index = tid; row_index < height; row_index += nthreads) {
        RowSlice row(trace + row_index, height);
        row.fill_zero(0, width);
        if (row_index < projection_len) {
            auto const &input = projection[row_index];
            const uint8_t *input_limbs =
                reinterpret_cast<const uint8_t *>(&input.heap_reads[0][0][0]);
            const uint8_t *logged_output =
                reinterpret_cast<const uint8_t *>(&input.writes[0][0]);
            if (!field_expr_fill_core_row<K>(
                    s,
                    row.slice_from(ADAPTER_WIDTH),
                    input_limbs,
                    logged_output,
                    input.local_opcode,
                    range_checker,
                    thread_scratch,
                    false,
                    error
                )) {
                return;
            }
            fill_vec_heap_adapter_from_projection(
                row, input, range_checker, pointer_max_bits, timestamp_max_bits
            );
        } else {
            if (s.should_finalize &&
                !field_expr_fill_core_row<K>(
                    s,
                    row.slice_from(ADAPTER_WIDTH),
                    nullptr,
                    nullptr,
                    UINT32_MAX,
                    range_checker,
                    thread_scratch,
                    true,
                    error
                )) {
                return;
            }
        }
    }
}

template <size_t NUM_READS, size_t BLOCKS>
static int field_expr_kernel_config(
    size_t *max_grid_blocks,
    size_t *block_threads,
    size_t *local_bytes_per_thread
) {
    if (max_grid_blocks == nullptr || block_threads == nullptr ||
        local_bytes_per_thread == nullptr) {
        return cudaErrorInvalidValue;
    }
    static constexpr int THREADS = 128;
    // These properties depend only on the selected device and kernel variant. The host caches
    // them with the chip and derives height/scratch-limited launch dimensions per trace.
    int blocks_per_multiprocessor;
    cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_multiprocessor,
        field_expr_replay_tracegen<NUM_READS, BLOCKS>,
        THREADS,
        0
    );
    if (result != cudaSuccess) return result;
    int device;
    result = cudaGetDevice(&device);
    if (result != cudaSuccess) return result;
    int multiprocessors;
    result =
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (result != cudaSuccess) return result;
    cudaFuncAttributes attributes;
    result = cudaFuncGetAttributes(
        &attributes, field_expr_replay_tracegen<NUM_READS, BLOCKS>
    );
    if (result != cudaSuccess) return result;
    size_t resident_blocks =
        static_cast<size_t>(blocks_per_multiprocessor) * multiprocessors;
    if (resident_blocks == 0) return cudaErrorInvalidValue;
    *max_grid_blocks = resident_blocks;
    *block_threads = THREADS;
    *local_bytes_per_thread = attributes.localSizeBytes;
    return cudaSuccess;
}

extern "C" int _field_expr_replay_kernel_config(
    size_t num_reads,
    size_t blocks,
    size_t *max_grid_blocks,
    size_t *block_threads,
    size_t *local_bytes_per_thread
) {
    if (num_reads == 2 && blocks == 4)
        return field_expr_kernel_config<2, 4>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 2 && blocks == 6)
        return field_expr_kernel_config<2, 6>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 2 && blocks == 8)
        return field_expr_kernel_config<2, 8>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 2 && blocks == 12)
        return field_expr_kernel_config<2, 12>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 1 && blocks == 8)
        return field_expr_kernel_config<1, 8>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    if (num_reads == 1 && blocks == 12)
        return field_expr_kernel_config<1, 12>(
            max_grid_blocks, block_threads, local_bytes_per_thread
        );
    return cudaErrorInvalidValue;
}

template <size_t NUM_READS, size_t BLOCKS>
static int launch_field_expr_replay(
    Fp *trace,
    size_t height,
    size_t width,
    const void *projection,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    uint32_t *range_delta,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    size_t grid_blocks,
    size_t block_threads,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (trace == nullptr || projection == nullptr || blob == nullptr ||
        range_delta == nullptr || scratch == nullptr || error == nullptr ||
        grid_blocks == 0 || block_threads == 0 || grid_blocks > UINT32_MAX ||
        block_threads > 1024) {
        return cudaErrorInvalidValue;
    }
    validate_field_expr_replay<NUM_READS, BLOCKS><<<1, 1, 0, stream>>>(
        height, width, projection_len, blob, blob_words, aux_words, error
    );
    if (int result = CHECK_KERNEL(); result != 0) return result;
    field_expr_replay_tracegen<NUM_READS, BLOCKS>
        <<<static_cast<uint32_t>(grid_blocks),
           static_cast<uint32_t>(block_threads),
           0,
           stream>>>(
            trace,
            height,
            width,
            static_cast<const VecHeapTraceInput<NUM_READS, BLOCKS> *>(projection),
            projection_len,
            blob,
            blob_words,
            range_delta,
            range_bins,
            scratch,
            scratch_words,
            aux_words,
            pointer_max_bits,
            timestamp_max_bits,
            error
        );
    return CHECK_KERNEL();
}

extern "C" int _field_expr_replay_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    size_t num_reads,
    size_t blocks,
    const void *projection,
    size_t projection_len,
    const uint32_t *blob,
    size_t blob_words,
    uint32_t *range_delta,
    size_t range_bins,
    uint32_t *scratch,
    size_t scratch_words,
    size_t aux_words,
    size_t grid_blocks,
    size_t block_threads,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits,
    uint32_t *error,
    cudaStream_t stream
) {
    if (num_reads == 2 && blocks == 4)
        return launch_field_expr_replay<2, 4>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 2 && blocks == 6)
        return launch_field_expr_replay<2, 6>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 2 && blocks == 8)
        return launch_field_expr_replay<2, 8>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 2 && blocks == 12)
        return launch_field_expr_replay<2, 12>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 1 && blocks == 8)
        return launch_field_expr_replay<1, 8>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    if (num_reads == 1 && blocks == 12)
        return launch_field_expr_replay<1, 12>(
            trace, height, width, projection, projection_len, blob, blob_words,
            range_delta, range_bins, scratch, scratch_words, aux_words, grid_blocks,
            block_threads, pointer_max_bits, timestamp_max_bits, error, stream
        );
    return cudaErrorInvalidValue;
}
