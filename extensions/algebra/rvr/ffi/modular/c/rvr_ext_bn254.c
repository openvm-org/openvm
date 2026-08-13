/* RVR BN254 scalar multiplication using MCL's native C API. */

#include "openvm.h"

#include <mcl/bn_c384_256.h>

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

static constexpr uint32_t BN254_FIELD_WORDS = 4;
static constexpr uint32_t BN254_POINT_WORDS = 2 * BN254_FIELD_WORDS;
static constexpr uint32_t BN254_SCALAR_WORDS = 4;

typedef struct {
    uint64_t limb0;
    uint64_t limb1;
    uint64_t limb2;
    uint64_t limb3;
} Bn254FieldBytes;

typedef struct {
    Bn254FieldBytes x;
    Bn254FieldBytes y;
} Bn254PointBytes;

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

    Bn254PointBytes point;
    Bn254FieldBytes scalar_bytes;
    read_mem_u64_range(state, rs1_ptr, (uint64_t *)&point, BN254_POINT_WORDS);
    read_mem_u64_range(state, rs2_ptr, (uint64_t *)&scalar_bytes, BN254_SCALAR_WORDS);

    mclBnG1 base;
    static constexpr Bn254PointBytes IDENTITY = {};
    if (memcmp(&point, &IDENTITY, sizeof(point)) == 0) {
        mclBnG1_clear(&base);
    } else {
        int x_ok = mclBnFp_setLittleEndianMod(&base.x, &point.x, sizeof(point.x));
        int y_ok = mclBnFp_setLittleEndianMod(&base.y, &point.y, sizeof(point.y));
        assert_assume(x_ok == 0 && y_ok == 0);
        mclBnFp_setInt32(&base.z, 1);
        if (!mclBnG1_isValid(&base)) {
            mclBnG1_clear(&base);
        }
    }

    scalar_bytes.limb0 |= 1;
    mclBnFr scalar;
    int scalar_ok = mclBnFr_setLittleEndianMod(&scalar, &scalar_bytes, sizeof(scalar_bytes));
    assert_assume(scalar_ok == 0);

    mclBnG1 product;
    mclBnG1_mul(&product, &base, &scalar);

    Bn254PointBytes output = {};
    if (!mclBnG1_isZero(&product)) {
        mclBnG1 normalized;
        mclBnG1_normalize(&normalized, &product);
        mclSize x_size = mclBnFp_serialize(&output.x, sizeof(output.x), &normalized.x);
        mclSize y_size = mclBnFp_serialize(&output.y, sizeof(output.y), &normalized.y);
        assert_assume(x_size == 32 && y_size == 32);
    }
    write_mem_u64_range(state, rd_ptr, (const uint64_t *)&output, BN254_POINT_WORDS);
}
