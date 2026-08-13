/* RVR BN254 scalar multiplication using MCL's native C API. */

#include "openvm.h"
#include "rvr_ext_ecc.h"

#include <mcl/bn_c384_256.h>

#include <stdatomic.h>
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

static constexpr uint32_t MCL_UNINITIALIZED = 0;
static constexpr uint32_t MCL_INITIALIZING = 1;
static constexpr uint32_t MCL_READY = 2;
static constexpr uint32_t MCL_FAILED = 3;
static atomic_uint mcl_init_state;

/* MCL initialization is process-global and not thread-safe. */
static void ensure_mcl_bn254_initialized(void) {
    uint32_t state = atomic_load_explicit(&mcl_init_state, memory_order_acquire);
    if (likely(state == MCL_READY)) {
        return;
    }

    uint32_t expected = MCL_UNINITIALIZED;
    if (atomic_compare_exchange_strong_explicit(
            &mcl_init_state, &expected, MCL_INITIALIZING, memory_order_acq_rel, memory_order_acquire
        )) {
        bool initialized = mclBn_init(MCL_BN_SNARK1, MCLBN_COMPILED_TIME_VAR) == 0 &&
                           mclBn_getFpByteSize() == 32 && mclBn_getFrByteSize() == 32;
        state = initialized ? MCL_READY : MCL_FAILED;
        atomic_store_explicit(&mcl_init_state, state, memory_order_release);
    } else {
        do {
            state = atomic_load_explicit(&mcl_init_state, memory_order_acquire);
        } while (state == MCL_INITIALIZING);
    }

    if (unlikely(state != MCL_READY)) {
        __builtin_trap();
    }
}

__attribute__((preserve_most)) void rvr_ext_ec_mul_bn254(
    RvState *restrict state,
    uint64_t rd_ptr,
    uint64_t rs1_ptr,
    uint64_t rs2_ptr
) {
    ensure_mcl_bn254_initialized();

    uint64_t x[BN254_FIELD_WORDS];
    uint64_t y[BN254_FIELD_WORDS];
    uint64_t scalar_words[BN254_SCALAR_WORDS];
    read_mem_u64_range(state, rs1_ptr, x, BN254_FIELD_WORDS);
    read_mem_u64_range(state, rs1_ptr + BN254_FIELD_BYTES, y, BN254_FIELD_WORDS);
    read_mem_u64_range(state, rs2_ptr, scalar_words, BN254_SCALAR_WORDS);

    mclBnG1 base;
    static constexpr uint64_t ZERO[BN254_FIELD_WORDS] = {};
    if (memcmp(x, ZERO, sizeof(x)) == 0 && memcmp(y, ZERO, sizeof(y)) == 0) {
        mclBnG1_clear(&base);
    } else {
        int x_ok = mclBnFp_setLittleEndianMod(&base.x, x, BN254_FIELD_BYTES);
        int y_ok = mclBnFp_setLittleEndianMod(&base.y, y, BN254_FIELD_BYTES);
        assert_assume(x_ok == 0 && y_ok == 0);
        mclBnFp_setInt32(&base.z, 1);
        if (!mclBnG1_isValid(&base)) {
            mclBnG1_clear(&base);
        }
    }

    scalar_words[0] |= 1;
    mclBnFr scalar;
    int scalar_ok = mclBnFr_setLittleEndianMod(&scalar, scalar_words, BN254_SCALAR_BYTES);
    assert_assume(scalar_ok == 0);

    mclBnG1 product;
    mclBnG1_mul(&product, &base, &scalar);

    uint64_t output_x[BN254_FIELD_WORDS] = {};
    uint64_t output_y[BN254_FIELD_WORDS] = {};
    if (!mclBnG1_isZero(&product)) {
        mclBnG1 normalized;
        mclBnG1_normalize(&normalized, &product);
        mclSize x_size = mclBnFp_serialize(output_x, BN254_FIELD_BYTES, &normalized.x);
        mclSize y_size = mclBnFp_serialize(output_y, BN254_FIELD_BYTES, &normalized.y);
        assert_assume(x_size == BN254_FIELD_BYTES && y_size == BN254_FIELD_BYTES);
    }
    write_mem_u64_range(state, rd_ptr, output_x, BN254_FIELD_WORDS);
    write_mem_u64_range(state, rd_ptr + BN254_FIELD_BYTES, output_y, BN254_FIELD_WORDS);
}
