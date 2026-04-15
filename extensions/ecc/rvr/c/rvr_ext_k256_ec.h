/* k256 EC ops (add_ne, double). Emit code only when RVR_EXT_K256_IMPL is set.
 * Include after "rvr_ext_k256_fe.h".
 * This header defines the functions inline and is intended to be included
 * exactly once from algebra's rvr_ext_k256.c so libsecp256k1 stays in one TU. */

#ifndef RVR_EXT_K256_EC_H
#define RVR_EXT_K256_EC_H

#ifdef RVR_EXT_K256_IMPL
#ifndef RVR_EXT_K256_EC_IMPL
#define RVR_EXT_K256_EC_IMPL

__attribute__((preserve_most)) void rvr_ext_ec_add_ne_k256(RvState* restrict state,
                                                           uint32_t rd_ptr, uint32_t rs1_ptr,
                                                           uint32_t rs2_ptr) {
    secp256k1_fe x1 = fe_read(state, rs1_ptr);
    secp256k1_fe y1 = fe_read(state, rs1_ptr + SECP256K1_ELEM_BYTES);
    secp256k1_fe x2 = fe_read(state, rs2_ptr);
    secp256k1_fe y2 = fe_read(state, rs2_ptr + SECP256K1_ELEM_BYTES);

    /* lambda = (y2 - y1) / (x2 - x1) */
    secp256k1_fe dy = fe_sub(&y2, &y1);
    secp256k1_fe dx = fe_sub(&x2, &x1);
    debug_assume(!fe_is_zero(&dx));
    secp256k1_fe dx_inv = fe_inv(&dx);
    secp256k1_fe lambda = fe_mul(&dy, &dx_inv);

    /* x3 = lambda^2 - x1 - x2 */
    secp256k1_fe lsq = fe_mul(&lambda, &lambda);
    secp256k1_fe x3 = fe_sub(&lsq, &x1);
    x3 = fe_sub(&x3, &x2);

    /* y3 = lambda * (x1 - x3) - y1 */
    secp256k1_fe dx13 = fe_sub(&x1, &x3);
    secp256k1_fe y3 = fe_mul(&lambda, &dx13);
    y3 = fe_sub(&y3, &y1);

    fe_write(state, rd_ptr, &x3);
    fe_write(state, rd_ptr + SECP256K1_ELEM_BYTES, &y3);
}

__attribute__((preserve_most)) void rvr_ext_ec_double_k256(RvState* restrict state,
                                                           uint32_t rd_ptr, uint32_t rs1_ptr) {
    secp256k1_fe x1 = fe_read(state, rs1_ptr);
    secp256k1_fe y1 = fe_read(state, rs1_ptr + SECP256K1_ELEM_BYTES);

    /* lambda = 3*x1^2 / (2*y1)  (a = 0 for secp256k1) */
    secp256k1_fe x1sq = fe_mul(&x1, &x1);
    secp256k1_fe three_x1sq = fe_add(fe_add(x1sq, &x1sq), &x1sq);
    secp256k1_fe two_y1 = fe_add(y1, &y1);
    debug_assume(!fe_is_zero(&two_y1));
    secp256k1_fe two_y1_inv = fe_inv(&two_y1);
    secp256k1_fe lambda = fe_mul(&three_x1sq, &two_y1_inv);

    /* x3 = lambda^2 - 2*x1 */
    secp256k1_fe lsq = fe_mul(&lambda, &lambda);
    secp256k1_fe two_x1 = fe_add(x1, &x1);
    secp256k1_fe x3 = fe_sub(&lsq, &two_x1);

    /* y3 = lambda * (x1 - x3) - y1 */
    secp256k1_fe dx = fe_sub(&x1, &x3);
    secp256k1_fe y3 = fe_mul(&lambda, &dx);
    y3 = fe_sub(&y3, &y1);

    fe_write(state, rd_ptr, &x3);
    fe_write(state, rd_ptr + SECP256K1_ELEM_BYTES, &y3);
}

#endif /* RVR_EXT_K256_EC_IMPL */
#endif /* RVR_EXT_K256_IMPL */

#endif /* RVR_EXT_K256_EC_H */
