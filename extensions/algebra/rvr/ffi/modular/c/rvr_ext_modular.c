/* Single-TU wrapper bundling libsecp256k1 (with ECC modules) plus the
 * rvr-ext FFI shims for k256 modular and EC ops. Compiled at lift time
 * via `ModularRvrExtension::c_sources`.
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "openvm.h"
#include "rvr_ext_mod.h"

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "k256 guest limb codecs require a little-endian host"
#endif

static constexpr uint32_t SECP256K1_ELEM_BYTES = 32;
static constexpr uint32_t SECP256K1_ELEM_WORDS =
    SECP256K1_ELEM_BYTES / WORD_SIZE;

/* ── libsecp256k1 amalgamation (ECC modules always enabled) ──────────── */
#include "secp256k1.c"

/* ── k256 field (mod p) helpers ──────────────────────────────────────── */

static inline void bytes_reverse_32(
    uint8_t out[static const SECP256K1_ELEM_BYTES],
    const uint8_t in[static const SECP256K1_ELEM_BYTES]) {
  _Static_assert(SECP256K1_ELEM_BYTES == 4 * sizeof(uint64_t), "");
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

static inline secp256k1_fe fe_read(RvState* state, uint64_t ptr) {
  uint64_t words[SECP256K1_ELEM_WORDS];
  read_mem_u64_range(state, ptr, words, SECP256K1_ELEM_WORDS);
  uint8_t le[SECP256K1_ELEM_BYTES];
  memcpy(le, words, SECP256K1_ELEM_BYTES);
  uint8_t be[SECP256K1_ELEM_BYTES];
  bytes_reverse_32(be, le);
  secp256k1_fe r;
  /* Match the Rust path: guest inputs are reduced mod p. */
  secp256k1_fe_set_b32_mod(&r, be);
  return r;
}

static inline void fe_write(RvState* state, uint64_t ptr,
                            const secp256k1_fe* val) {
  secp256k1_fe t = *val;
  secp256k1_fe_normalize_var(&t);
  uint8_t be[SECP256K1_ELEM_BYTES];
  secp256k1_fe_get_b32(be, &t);
  uint8_t le[SECP256K1_ELEM_BYTES];
  bytes_reverse_32(le, be);
  uint64_t words[SECP256K1_ELEM_WORDS];
  memcpy(words, le, SECP256K1_ELEM_BYTES);
  write_mem_u64_range(state, ptr, words, SECP256K1_ELEM_WORDS);
}

static inline secp256k1_fe fe_add(secp256k1_fe a, const secp256k1_fe* b) {
  secp256k1_fe_add(&a, b);
  secp256k1_fe_normalize_weak(&a);
  return a;
}

static inline secp256k1_fe fe_sub(const secp256k1_fe* a,
                                  const secp256k1_fe* b) {
  secp256k1_fe neg_b;
  secp256k1_fe_negate(&neg_b, b, 1);
  secp256k1_fe r = *a;
  secp256k1_fe_add(&r, &neg_b);
  secp256k1_fe_normalize_weak(&r);
  return r;
}

static inline secp256k1_fe fe_mul(const secp256k1_fe* a,
                                  const secp256k1_fe* b) {
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

/* ── k256 scalar (mod n) helpers ─────────────────────────────────────── */

static inline secp256k1_scalar scalar_read(RvState* state, uint64_t ptr) {
  uint64_t words[SECP256K1_ELEM_WORDS];
  read_mem_u64_range(state, ptr, words, SECP256K1_ELEM_WORDS);
  uint8_t le[SECP256K1_ELEM_BYTES];
  memcpy(le, words, SECP256K1_ELEM_BYTES);
  uint8_t be[SECP256K1_ELEM_BYTES];
  bytes_reverse_32(be, le);
  secp256k1_scalar r;
  secp256k1_scalar_set_b32(&r, be, NULL);
  return r;
}

static inline void scalar_write(RvState* state, uint64_t ptr,
                                const secp256k1_scalar* val) {
  uint8_t be[SECP256K1_ELEM_BYTES];
  secp256k1_scalar_get_b32(be, val);
  uint8_t le[SECP256K1_ELEM_BYTES];
  bytes_reverse_32(le, be);
  uint64_t words[SECP256K1_ELEM_WORDS];
  memcpy(words, le, SECP256K1_ELEM_BYTES);
  write_mem_u64_range(state, ptr, words, SECP256K1_ELEM_WORDS);
}

/* ── Modular arithmetic: secp256k1 coordinate field (mod p) ──────────── */

__attribute__((preserve_most)) void rvr_ext_mod_add_k256_coord(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_fe a = fe_read(state, rs1_ptr);
  secp256k1_fe b = fe_read(state, rs2_ptr);
  secp256k1_fe r = fe_add(a, &b);
  fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_sub_k256_coord(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_fe a = fe_read(state, rs1_ptr);
  secp256k1_fe b = fe_read(state, rs2_ptr);
  secp256k1_fe r = fe_sub(&a, &b);
  fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_mul_k256_coord(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_fe a = fe_read(state, rs1_ptr);
  secp256k1_fe b = fe_read(state, rs2_ptr);
  secp256k1_fe r = fe_mul(&a, &b);
  fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_div_k256_coord(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_fe a = fe_read(state, rs1_ptr);
  secp256k1_fe b = fe_read(state, rs2_ptr);
  debug_assume(!fe_is_zero(&b));
  secp256k1_fe b_inv = fe_inv(&b);
  secp256k1_fe r = fe_mul(&a, &b_inv);
  fe_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) bool rvr_ext_mod_iseq_k256_coord(
    RvState* state, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_fe a = fe_read(state, rs1_ptr);
  secp256k1_fe b = fe_read(state, rs2_ptr);
  return secp256k1_fe_equal(&a, &b) != 0;
}

/* ── Modular arithmetic: secp256k1 scalar field (mod n) ──────────────── */

__attribute__((preserve_most)) void rvr_ext_mod_add_k256_scalar(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_scalar a = scalar_read(state, rs1_ptr);
  secp256k1_scalar b = scalar_read(state, rs2_ptr);
  secp256k1_scalar r;
  secp256k1_scalar_add(&r, &a, &b);
  scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_sub_k256_scalar(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_scalar a = scalar_read(state, rs1_ptr);
  secp256k1_scalar b = scalar_read(state, rs2_ptr);
  secp256k1_scalar neg_b;
  secp256k1_scalar_negate(&neg_b, &b);
  secp256k1_scalar r;
  secp256k1_scalar_add(&r, &a, &neg_b);
  scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_mul_k256_scalar(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_scalar a = scalar_read(state, rs1_ptr);
  secp256k1_scalar b = scalar_read(state, rs2_ptr);
  secp256k1_scalar r;
  secp256k1_scalar_mul(&r, &a, &b);
  scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) void rvr_ext_mod_div_k256_scalar(
    RvState* state, uint64_t rd_ptr, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_scalar a = scalar_read(state, rs1_ptr);
  secp256k1_scalar b = scalar_read(state, rs2_ptr);
  debug_assume(!secp256k1_scalar_is_zero(&b));
  secp256k1_scalar b_inv;
  secp256k1_scalar_inverse_var(&b_inv, &b);
  secp256k1_scalar r;
  secp256k1_scalar_mul(&r, &a, &b_inv);
  scalar_write(state, rd_ptr, &r);
}

__attribute__((preserve_most)) bool rvr_ext_mod_iseq_k256_scalar(
    RvState* state, uint64_t rs1_ptr, uint64_t rs2_ptr) {
  secp256k1_scalar a = scalar_read(state, rs1_ptr);
  secp256k1_scalar b = scalar_read(state, rs2_ptr);
  return secp256k1_scalar_eq(&a, &b) != 0;
}

/* ── EC ops: secp256k1 (always present in modular, regardless of whether
 * the ECC extension is configured at lift time) ──────────────────────── */

__attribute__((preserve_most)) void rvr_ext_ec_add_proj_k256(RvState* state,
                                                             uint64_t rd_ptr,
                                                             uint64_t rs1_ptr,
                                                             uint64_t rs2_ptr);
__attribute__((preserve_most)) void rvr_ext_ec_double_proj_k256(RvState* state,
                                                                uint64_t rd_ptr,
                                                                uint64_t rs1_ptr);

/* Complete projective point addition (a = 0), ePrint 2015/1060 Algorithm 7.
 * Mirrors the circuit's `ec_add_proj_impl_a0`; `3b = 21` for secp256k1 (b = 7). */
__attribute__((preserve_most)) void rvr_ext_ec_add_proj_k256(RvState* state,
                                                             uint64_t rd_ptr,
                                                             uint64_t rs1_ptr,
                                                             uint64_t rs2_ptr) {
  secp256k1_fe x1 = fe_read(state, rs1_ptr);
  secp256k1_fe y1 = fe_read(state, rs1_ptr + SECP256K1_ELEM_BYTES);
  secp256k1_fe z1 = fe_read(state, rs1_ptr + 2 * SECP256K1_ELEM_BYTES);
  secp256k1_fe x2 = fe_read(state, rs2_ptr);
  secp256k1_fe y2 = fe_read(state, rs2_ptr + SECP256K1_ELEM_BYTES);
  secp256k1_fe z2 = fe_read(state, rs2_ptr + 2 * SECP256K1_ELEM_BYTES);

  secp256k1_fe b3;
  secp256k1_fe_set_int(&b3, 21);

  secp256k1_fe t0 = fe_mul(&x1, &x2);
  secp256k1_fe t1 = fe_mul(&y1, &y2);
  secp256k1_fe t2 = fe_mul(&z1, &z2);
  secp256k1_fe t3 = fe_add(x1, &y1);
  secp256k1_fe t4 = fe_add(x2, &y2);
  t3 = fe_mul(&t3, &t4);
  t4 = fe_add(t0, &t1);
  t3 = fe_sub(&t3, &t4);
  t4 = fe_add(y1, &z1);
  secp256k1_fe x3 = fe_add(y2, &z2);
  t4 = fe_mul(&t4, &x3);
  x3 = fe_add(t1, &t2);
  t4 = fe_sub(&t4, &x3);
  x3 = fe_add(x1, &z1);
  secp256k1_fe y3 = fe_add(x2, &z2);
  x3 = fe_mul(&x3, &y3);
  y3 = fe_add(t0, &t2);
  y3 = fe_sub(&x3, &y3);
  x3 = fe_add(fe_add(t0, &t0), &t0);
  t2 = fe_mul(&b3, &t2);
  secp256k1_fe z3 = fe_add(t1, &t2);
  t1 = fe_sub(&t1, &t2);
  y3 = fe_mul(&b3, &y3);
  secp256k1_fe x3_out = fe_mul(&t4, &y3);
  t2 = fe_mul(&t3, &t1);
  x3_out = fe_sub(&t2, &x3_out);
  y3 = fe_mul(&y3, &x3);
  t1 = fe_mul(&t1, &z3);
  secp256k1_fe y3_out = fe_add(t1, &y3);
  x3 = fe_mul(&x3, &t3);
  z3 = fe_mul(&z3, &t4);
  secp256k1_fe z3_out = fe_add(z3, &x3);

  fe_write(state, rd_ptr, &x3_out);
  fe_write(state, rd_ptr + SECP256K1_ELEM_BYTES, &y3_out);
  fe_write(state, rd_ptr + 2 * SECP256K1_ELEM_BYTES, &z3_out);
}

/* Complete projective point doubling (a = 0), ePrint 2015/1060 Algorithm 9.
 * Mirrors the circuit's `ec_double_proj_impl_a0`; `3b = 21` for secp256k1. */
__attribute__((preserve_most)) void rvr_ext_ec_double_proj_k256(RvState* state,
                                                                uint64_t rd_ptr,
                                                                uint64_t rs1_ptr) {
  secp256k1_fe x1 = fe_read(state, rs1_ptr);
  secp256k1_fe y1 = fe_read(state, rs1_ptr + SECP256K1_ELEM_BYTES);
  secp256k1_fe z1 = fe_read(state, rs1_ptr + 2 * SECP256K1_ELEM_BYTES);

  secp256k1_fe b3;
  secp256k1_fe_set_int(&b3, 21);

  secp256k1_fe t0 = fe_mul(&y1, &y1);
  secp256k1_fe z3 = fe_add(t0, &t0);
  z3 = fe_add(z3, &z3);
  z3 = fe_add(z3, &z3);
  secp256k1_fe t1 = fe_mul(&y1, &z1);
  secp256k1_fe t2 = fe_mul(&z1, &z1);
  t2 = fe_mul(&b3, &t2);
  secp256k1_fe x3 = fe_mul(&t2, &z3);
  secp256k1_fe y3 = fe_add(t0, &t2);
  z3 = fe_mul(&t1, &z3);
  t1 = fe_add(t2, &t2);
  t2 = fe_add(t1, &t2);
  t0 = fe_sub(&t0, &t2);
  y3 = fe_mul(&t0, &y3);
  y3 = fe_add(x3, &y3);
  t1 = fe_mul(&x1, &y1);
  x3 = fe_mul(&t0, &t1);
  x3 = fe_add(x3, &x3);

  fe_write(state, rd_ptr, &x3);
  fe_write(state, rd_ptr + SECP256K1_ELEM_BYTES, &y3);
  fe_write(state, rd_ptr + 2 * SECP256K1_ELEM_BYTES, &z3);
}
