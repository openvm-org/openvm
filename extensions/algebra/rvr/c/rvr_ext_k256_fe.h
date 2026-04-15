/* k256 field (mod p) helpers. Emit code only when RVR_EXT_K256_IMPL is set
 * after including "openvm.h" and "secp256k1.c". */

#ifndef RVR_EXT_K256_FE_H
#define RVR_EXT_K256_FE_H

#include <stdint.h>
#include <string.h>

#ifdef RVR_EXT_K256_IMPL
#ifndef RVR_EXT_K256_FE_IMPL
#define RVR_EXT_K256_FE_IMPL

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "k256 guest limb codecs require a little-endian host"
#endif

static constexpr uint32_t SECP256K1_ELEM_BYTES = 32;
static constexpr uint32_t SECP256K1_ELEM_WORDS = SECP256K1_ELEM_BYTES / WORD_SIZE;

/* libsecp256k1 b32 codecs are big-endian; guest memory is little-endian. */
static inline void bytes_reverse_32(uint8_t out[SECP256K1_ELEM_BYTES],
                                    const uint8_t in[SECP256K1_ELEM_BYTES]) {
    static_assert(SECP256K1_ELEM_BYTES == 4 * sizeof(uint64_t));

    uint64_t words[4];
    memcpy(words, in, sizeof(words));

    uint64_t reversed[4] = {
        __builtin_bswap64(words[3]),
        __builtin_bswap64(words[2]),
        __builtin_bswap64(words[1]),
        __builtin_bswap64(words[0]),
    };
    memcpy(out, reversed, sizeof(reversed));
}

static inline secp256k1_fe fe_read(RvState* restrict state, uint32_t ptr) {
    uint32_t words[SECP256K1_ELEM_WORDS];
    rd_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
    uint8_t le[SECP256K1_ELEM_BYTES];
    memcpy(le, words, SECP256K1_ELEM_BYTES);
    uint8_t be[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(be, le);
    secp256k1_fe r;
    /* Match the old halo2curves-backed path: guest inputs are reduced mod p. */
    secp256k1_fe_set_b32_mod(&r, be);
    return r;
}

static inline void fe_write(RvState* restrict state, uint32_t ptr, const secp256k1_fe* val) {
    secp256k1_fe t = *val;
    secp256k1_fe_normalize_var(&t);
    uint8_t be[SECP256K1_ELEM_BYTES];
    secp256k1_fe_get_b32(be, &t);
    uint8_t le[SECP256K1_ELEM_BYTES];
    bytes_reverse_32(le, be);
    uint32_t words[SECP256K1_ELEM_WORDS];
    memcpy(words, le, SECP256K1_ELEM_BYTES);
    wr_mem_u32_range_traced(state, ptr, words, SECP256K1_ELEM_WORDS);
}

/* `a` is intentionally by-value so callers can chain fe_add(fe_add(...), ...)
 * without mutating an input in place. */
static inline secp256k1_fe fe_add(secp256k1_fe a, const secp256k1_fe* b) {
    secp256k1_fe_add(&a, b);
    secp256k1_fe_normalize_weak(&a);
    return a;
}

static inline secp256k1_fe fe_sub(const secp256k1_fe* a, const secp256k1_fe* b) {
    secp256k1_fe neg_b;
    secp256k1_fe_negate(&neg_b, b, 1);
    secp256k1_fe r = *a;
    secp256k1_fe_add(&r, &neg_b);
    secp256k1_fe_normalize_weak(&r);
    return r;
}

static inline secp256k1_fe fe_mul(const secp256k1_fe* a, const secp256k1_fe* b) {
    secp256k1_fe r;
    secp256k1_fe_mul(&r, a, b);
    return r;
}

static inline secp256k1_fe fe_inv(const secp256k1_fe* a) {
    secp256k1_fe r;
    secp256k1_fe_inv_var(&r, a);
    return r;
}

static inline int fe_is_zero(const secp256k1_fe* a) {
    secp256k1_fe t = *a;
    secp256k1_fe_normalize_var(&t);
    return secp256k1_fe_is_zero(&t);
}

#endif /* RVR_EXT_K256_FE_IMPL */
#endif /* RVR_EXT_K256_IMPL */

#endif /* RVR_EXT_K256_FE_H */
