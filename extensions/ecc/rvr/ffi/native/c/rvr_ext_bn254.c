/* RVR BN254 scalar multiplication using MCL's native C API. */

#include "openvm.h"
#include "rvr_ext_ecc.h"

#include <mcl/bn_c384_256.h>

#include <stdint.h>
#include <string.h>

/* MCL and the guest both encode BN254 values as little-endian 32-byte limbs. */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "BN254 guest limb codecs require a little-endian host"
#endif

static constexpr uint32_t BN254_FIELD_BYTES = 32;
static constexpr uint32_t BN254_FIELD_WORDS = BN254_FIELD_BYTES / WORD_SIZE;
static constexpr uint32_t BN254_SCALAR_BYTES = 32;
static constexpr uint32_t BN254_SCALAR_WORDS = BN254_SCALAR_BYTES / WORD_SIZE;

static bool mcl_bn254_ready;

static bool is_mcl_bn254_ready(void) {
    return mclBn_getCurveType() == MCL_BN_SNARK1 && mclBn_getFpByteSize() == BN254_FIELD_BYTES &&
           mclBn_getFrByteSize() == BN254_SCALAR_BYTES;
}

/* MCL initialization is process-global and not thread-safe; run it at load, before the compiled
 * RVR instance can be shared across threads. */
__attribute__((constructor)) static void initialize_mcl_bn254_at_load(void) {
    mcl_bn254_ready =
        is_mcl_bn254_ready() ||
        (mclBn_init(MCL_BN_SNARK1, MCLBN_COMPILED_TIME_VAR) == 0 && is_mcl_bn254_ready());
}

__attribute__((preserve_most)) void rvr_ext_ec_mul_bn254(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    /* MCL curve selection is global. Trap if it changed after initialization. */
    if (unlikely(!mcl_bn254_ready || !is_mcl_bn254_ready())) {
        __builtin_trap();
    }

    uint64_t x[BN254_FIELD_WORDS];
    uint64_t y[BN254_FIELD_WORDS];
    uint64_t scalar_words[BN254_SCALAR_WORDS];
    read_mem_u64_range(state, rs1_ptr, x, BN254_FIELD_WORDS);
    read_mem_u64_range(state, rs1_ptr + BN254_FIELD_BYTES, y, BN254_FIELD_WORDS);
    read_mem_u64_range(state, rs2_ptr, scalar_words, BN254_SCALAR_WORDS);

    mclBnG1 base;
    static constexpr uint64_t ZERO[BN254_FIELD_WORDS] = {};
    if (memcmp(x, ZERO, sizeof(x)) == 0 && memcmp(y, ZERO, sizeof(y)) == 0) {
        /* Keep execution defined for the VM identity value. The EC_MUL contract excludes it. */
        mclBnG1_clear(&base);
    } else {
        int x_ok = mclBnFp_setLittleEndianMod(&base.x, x, BN254_FIELD_BYTES);
        int y_ok = mclBnFp_setLittleEndianMod(&base.y, y, BN254_FIELD_BYTES);
        /* The decoder accepts inputs of up to 64 bytes. */
        assert_assume(x_ok == 0 && y_ok == 0);
        mclBnFp_setInt32(&base.z, 1);
        if (!mclBnG1_isValid(&base)) {
            /* Keep execution defined for an invalid point. The EC_MUL contract excludes it. */
            mclBnG1_clear(&base);
        }
    }

    /* EC_MUL uses scalar | 1. Valid inputs are already odd and below the group order. */
    scalar_words[0] |= 1;
    mclBnFr scalar;
    int scalar_ok = mclBnFr_setLittleEndianMod(&scalar, scalar_words, BN254_SCALAR_BYTES);
    /* The decoder accepts inputs of up to 64 bytes. */
    assert_assume(scalar_ok == 0);

    mclBnG1 product;
    mclBnG1_mul(&product, &base, &scalar);

    uint64_t output_x[BN254_FIELD_WORDS] = {};
    uint64_t output_y[BN254_FIELD_WORDS] = {};
    if (!mclBnG1_isZero(&product)) {
        mclBnG1 normalized;
        mclBnG1_normalize(&normalized, &product);
        mclSize x_size = mclBnFp_getLittleEndian(output_x, BN254_FIELD_BYTES, &normalized.x);
        mclSize y_size = mclBnFp_getLittleEndian(output_y, BN254_FIELD_BYTES, &normalized.y);
        /* Each field value fits in 32 bytes. The zero-filled buffers add any required padding. */
        assert_assume(
            x_size > 0 && x_size <= BN254_FIELD_BYTES && y_size > 0 && y_size <= BN254_FIELD_BYTES
        );
    }
    write_mem_u64_range(state, rd_ptr, output_x, BN254_FIELD_WORDS);
    write_mem_u64_range(state, rd_ptr + BN254_FIELD_BYTES, output_y, BN254_FIELD_WORDS);
}
