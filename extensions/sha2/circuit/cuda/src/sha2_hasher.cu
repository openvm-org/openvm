#define SHA2_DEFINE_DEVICE_CONSTANTS

#include "block_hasher/columns.cuh"
#include "block_hasher/record.cuh"
#include "block_hasher/variant.cuh"
#include "fp.h"
#include "launcher.cuh"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace riscv;
using namespace sha2;

// === Utility helpers for SHA-2 block hasher ===
template <typename V>
__device__ __forceinline__ typename V::Word word_from_bytes_be(const uint8_t *bytes) {
    typename V::Word acc = 0;
#pragma unroll
    for (int i = 0; i < static_cast<int>(V::WORD_U8S); i++) {
        acc = (acc << 8) | static_cast<typename V::Word>(bytes[i]);
    }
    return acc;
}

template <typename V>
__device__ __forceinline__ typename V::Word word_from_bytes_le(const uint8_t *bytes) {
    typename V::Word acc = 0;
#pragma unroll
    for (int i = static_cast<int>(V::WORD_U8S) - 1; i >= 0; i--) {
        acc = (acc << 8) | static_cast<typename V::Word>(bytes[i]);
    }
    return acc;
}

template <typename V>
__device__ __forceinline__ uint32_t word_to_u16_limb(typename V::Word w, int limb) {
    return static_cast<uint32_t>((w >> (16 * limb)) & static_cast<typename V::Word>(0xFFFF));
}

template <typename V>
__device__ __forceinline__ uint32_t word_to_u8_limb(typename V::Word w, int limb) {
    return static_cast<uint32_t>((w >> (8 * limb)) & static_cast<typename V::Word>(0xFF));
}

// === Shared helpers mirroring the CPU tracegen structure ===
template <typename V> struct Sha2TraceHelper {
    Encoder row_idx_encoder;

    __device__ Sha2TraceHelper()
        : row_idx_encoder(static_cast<uint32_t>(V::ROWS_PER_BLOCK + 1), 2, false) {}

    __device__ __forceinline__ size_t base_a(uint32_t row_idx) const {
        return SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.a[row_idx]);
    }

    __device__ __forceinline__ size_t base_e(uint32_t row_idx) const {
        return SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.e[row_idx]);
    }

    __device__ __forceinline__ size_t base_carry_a(uint32_t row_idx) const {
        return SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.carry_a[row_idx]);
    }

    __device__ __forceinline__ size_t base_carry_e(uint32_t row_idx) const {
        return SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.carry_e[row_idx]);
    }

    __device__ __forceinline__ void read_w(RowSlice inner, uint32_t j, Fp *w_limbs) const {
        size_t base = SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.w[j]);
        for (int limb = 0; limb < V::WORD_U16S; limb++) {
            w_limbs[limb] = Fp::zero();
            for (int bit = 0; bit < 16; bit++) {
                w_limbs[limb] += inner[base + bit] * Fp(1 << bit);
            }
            base += 16;
        }
    }

    __device__ __forceinline__ Fp read_carry_fp(RowSlice inner, uint32_t i, uint32_t limb) const {
        size_t base = SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.carry_or_buffer[i]);
        Fp low = inner[base + limb * 2];
        Fp high = inner[base + limb * 2 + 1];
        return low + high + high; // low + 2 * high
    }

    __device__ __forceinline__ void read_word_bits(
        RowSlice inner,
        size_t base,
        Fp *dst_bits
    ) const {
#pragma unroll
        for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
            dst_bits[bit] = inner[base + bit];
        }
    }

    __device__ __forceinline__ void read_w_bits(RowSlice inner, uint32_t j, Fp *dst_bits) const {
        read_word_bits(inner, SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.w[j]), dst_bits);
    }

    // Helpers to get bits from either local or next inner row without pre-loading arrays
    __device__ __forceinline__ Fp
    get_a_bit(RowSlice local_inner, RowSlice next_inner, uint32_t round_idx, uint32_t bit) const {
        if (round_idx < V::ROUNDS_PER_ROW) {
            return local_inner[base_a(round_idx) + bit];
        } else {
            return next_inner[base_a(round_idx - V::ROUNDS_PER_ROW) + bit];
        }
    }

    __device__ __forceinline__ Fp
    get_e_bit(RowSlice local_inner, RowSlice next_inner, uint32_t round_idx, uint32_t bit) const {
        if (round_idx < V::ROUNDS_PER_ROW) {
            return local_inner[base_e(round_idx) + bit];
        } else {
            return next_inner[base_e(round_idx - V::ROUNDS_PER_ROW) + bit];
        }
    }

    __device__ __forceinline__ Fp
    get_w_bit(RowSlice local_inner, RowSlice next_inner, uint32_t round_idx, uint32_t bit) const {
        size_t base;
        if (round_idx < V::ROUNDS_PER_ROW) {
            base = SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.w[round_idx]);
            return local_inner[base + bit];
        } else {
            base = SHA2_COL_INDEX(
                V, Sha2RoundCols, message_schedule.w[round_idx - V::ROUNDS_PER_ROW]
            );
            return next_inner[base + bit];
        }
    }

    __device__ __forceinline__ Fp xor_fp(Fp a, Fp b) const { return a + b - Fp(2) * a * b; }

    __device__ __forceinline__ Fp xor_fp(Fp a, Fp b, Fp c) const { return xor_fp(xor_fp(a, b), c); }

    __device__ __forceinline__ Fp ch_fp(Fp x, Fp y, Fp z) const { return x * y + z - x * z; }

    __device__ __forceinline__ Fp maj_fp(Fp x, Fp y, Fp z) const {
        return x * y + x * z + y * z - Fp(2) * x * y * z;
    }

    __device__ __forceinline__ void rotr_bits(const Fp *src, uint32_t rot, Fp *dst) const {
#pragma unroll
        for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
            dst[bit] = src[(bit + rot) % V::WORD_BITS];
        }
    }

    __device__ __forceinline__ void shr_bits(const Fp *src, uint32_t shift, Fp *dst) const {
#pragma unroll
        for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
            dst[bit] = (bit + shift < V::WORD_BITS) ? src[bit + shift] : Fp::zero();
        }
    }

    __device__ __forceinline__ void big_sig0_bits(const Fp *src, Fp *dst) const {
        if (V::WORD_BITS == 32) {
            Fp r2[V::WORD_BITS], r13[V::WORD_BITS], r22[V::WORD_BITS];
            rotr_bits(src, 2, r2);
            rotr_bits(src, 13, r13);
            rotr_bits(src, 22, r22);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r2[bit], r13[bit], r22[bit]);
            }
        } else {
            Fp r28[V::WORD_BITS], r34[V::WORD_BITS], r39[V::WORD_BITS];
            rotr_bits(src, 28, r28);
            rotr_bits(src, 34, r34);
            rotr_bits(src, 39, r39);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r28[bit], r34[bit], r39[bit]);
            }
        }
    }

    __device__ __forceinline__ void big_sig1_bits(const Fp *src, Fp *dst) const {
        if (V::WORD_BITS == 32) {
            Fp r6[V::WORD_BITS], r11[V::WORD_BITS], r25[V::WORD_BITS];
            rotr_bits(src, 6, r6);
            rotr_bits(src, 11, r11);
            rotr_bits(src, 25, r25);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r6[bit], r11[bit], r25[bit]);
            }
        } else {
            Fp r14[V::WORD_BITS], r18[V::WORD_BITS], r41[V::WORD_BITS];
            rotr_bits(src, 14, r14);
            rotr_bits(src, 18, r18);
            rotr_bits(src, 41, r41);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r14[bit], r18[bit], r41[bit]);
            }
        }
    }

    __device__ __forceinline__ void small_sig0_bits(const Fp *src, Fp *dst) const {
        if (V::WORD_BITS == 32) {
            Fp r7[V::WORD_BITS], r18[V::WORD_BITS], s3[V::WORD_BITS];
            rotr_bits(src, 7, r7);
            rotr_bits(src, 18, r18);
            shr_bits(src, 3, s3);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r7[bit], r18[bit], s3[bit]);
            }
        } else {
            Fp r1[V::WORD_BITS], r8[V::WORD_BITS], s7[V::WORD_BITS];
            rotr_bits(src, 1, r1);
            rotr_bits(src, 8, r8);
            shr_bits(src, 7, s7);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r1[bit], r8[bit], s7[bit]);
            }
        }
    }

    __device__ __forceinline__ void small_sig1_bits(const Fp *src, Fp *dst) const {
        if (V::WORD_BITS == 32) {
            Fp r17[V::WORD_BITS], r19[V::WORD_BITS], s10[V::WORD_BITS];
            rotr_bits(src, 17, r17);
            rotr_bits(src, 19, r19);
            shr_bits(src, 10, s10);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r17[bit], r19[bit], s10[bit]);
            }
        } else {
            Fp r19[V::WORD_BITS], r61[V::WORD_BITS], s6[V::WORD_BITS];
            rotr_bits(src, 19, r19);
            rotr_bits(src, 61, r61);
            shr_bits(src, 6, s6);
#pragma unroll
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                dst[bit] = xor_fp(r19[bit], r61[bit], s6[bit]);
            }
        }
    }

    __device__ __forceinline__ Fp compose_u16_limb(const Fp *bits, uint32_t limb) const {
        Fp acc = Fp::zero();
#pragma unroll
        for (uint32_t bit = 0; bit < 16; bit++) {
            acc += bits[limb * 16 + bit] * Fp(1u << bit);
        }
        return acc;
    }

    __device__ __noinline__ void write_flags_round(
        RowSlice inner_row,
        uint32_t row_idx,
        uint32_t global_block_idx
    ) const {
        SHA2INNER_WRITE_ROUND(V, inner_row, flags.is_round_row, Fp::one());
        SHA2INNER_WRITE_ROUND(
            V,
            inner_row,
            flags.is_first_4_rows,
            (row_idx < static_cast<uint32_t>(V::MESSAGE_ROWS)) ? Fp::one() : Fp::zero()
        );
        SHA2INNER_WRITE_ROUND(V, inner_row, flags.is_digest_row, Fp::zero());
        RowSlice row_idx_flags =
            inner_row.slice_from(SHA2_COL_INDEX(V, Sha2RoundCols, flags.row_idx));
        row_idx_encoder.write_flag_pt(row_idx_flags, row_idx);
        SHA2INNER_WRITE_ROUND(V, inner_row, flags.global_block_idx, Fp(global_block_idx));
    }

    __device__ __noinline__ void write_flags_digest(
        RowSlice inner_row,
        uint32_t row_idx,
        uint32_t global_block_idx
    ) const {
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.is_round_row, Fp::zero());
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.is_first_4_rows, Fp::zero());
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.is_digest_row, Fp::one());
        RowSlice row_idx_flags =
            inner_row.slice_from(SHA2_COL_INDEX(V, Sha2DigestCols, flags.row_idx));
        row_idx_encoder.write_flag_pt(row_idx_flags, row_idx);
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.global_block_idx, Fp(global_block_idx));
    }

    __device__ __noinline__ void generate_carry_ae(RowSlice local_inner, RowSlice next_inner)
        const {
        // Process one round at a time instead of pre-loading all 2*ROUNDS_PER_ROW rows.
        // This reduces per-function stack from ~2KB to ~512B for SHA-256.
        const Fp pow16_inv = inv(Fp(1u << 16));

        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            // Read a[i+3] and e[i+3] bits into small temp arrays for sig functions
            Fp a3_bits[V::WORD_BITS];
            Fp e3_bits[V::WORD_BITS];
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                a3_bits[bit] = get_a_bit(local_inner, next_inner, i + 3, bit);
                e3_bits[bit] = get_e_bit(local_inner, next_inner, i + 3, bit);
            }

            Fp sig_a[V::WORD_BITS];
            Fp sig_e[V::WORD_BITS];
            Fp maj_abc[V::WORD_BITS];
            Fp ch_efg[V::WORD_BITS];

            big_sig0_bits(a3_bits, sig_a);
            big_sig1_bits(e3_bits, sig_e);
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                maj_abc[bit] = maj_fp(
                    a3_bits[bit],
                    get_a_bit(local_inner, next_inner, i + 2, bit),
                    get_a_bit(local_inner, next_inner, i + 1, bit)
                );
                ch_efg[bit] = ch_fp(
                    e3_bits[bit],
                    get_e_bit(local_inner, next_inner, i + 2, bit),
                    get_e_bit(local_inner, next_inner, i + 1, bit)
                );
            }

            Fp prev_carry_a = Fp::zero();
            Fp prev_carry_e = Fp::zero();
            for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                // Compute e[i], a[i], a[i+4], e[i+4] limbs on-the-fly
                Fp e_i_limb = Fp::zero();
                Fp a_i_limb = Fp::zero();
                Fp a_i4_limb = Fp::zero();
                Fp e_i4_limb = Fp::zero();
#pragma unroll 1
                for (uint32_t bit = 0; bit < 16; bit++) {
                    Fp scale(1u << bit);
                    e_i_limb += get_e_bit(local_inner, next_inner, i, limb * 16 + bit) * scale;
                    a_i_limb += get_a_bit(local_inner, next_inner, i, limb * 16 + bit) * scale;
                    a_i4_limb +=
                        get_a_bit(local_inner, next_inner, i + 4, limb * 16 + bit) * scale;
                    e_i4_limb +=
                        get_e_bit(local_inner, next_inner, i + 4, limb * 16 + bit) * scale;
                }

                Fp t1_sum =
                    e_i_limb + compose_u16_limb(sig_e, limb) + compose_u16_limb(ch_efg, limb);
                Fp t2_sum = compose_u16_limb(sig_a, limb) + compose_u16_limb(maj_abc, limb);

                Fp e_sum = a_i_limb + t1_sum +
                           (limb == 0 ? Fp::zero() : next_inner[base_carry_e(i) + limb - 1]);
                Fp a_sum = t1_sum + t2_sum +
                           (limb == 0 ? Fp::zero() : next_inner[base_carry_a(i) + limb - 1]);
                Fp carry_e = (e_sum - e_i4_limb) * pow16_inv;
                Fp carry_a = (a_sum - a_i4_limb) * pow16_inv;

                SHA2INNER_WRITE_ROUND(V, next_inner, work_vars.carry_e[i][limb], Fp(carry_e));
                SHA2INNER_WRITE_ROUND(V, next_inner, work_vars.carry_a[i][limb], Fp(carry_a));

                prev_carry_e = carry_e;
                prev_carry_a = carry_a;
            }
        }
    }

    __device__ __noinline__ void generate_intermed_4(RowSlice local_inner, RowSlice next_inner)
        const {
        // Process one round at a time instead of pre-loading all 2*ROUNDS_PER_ROW rows.
        // This reduces per-function stack from ~1.2KB to ~256B for SHA-256.
        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            // Read w[i+1] bits on-the-fly for small_sig0_bits
            Fp w_i1_bits[V::WORD_BITS];
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                w_i1_bits[bit] = get_w_bit(local_inner, next_inner, i + 1, bit);
            }

            Fp sig_bits[V::WORD_BITS];
            small_sig0_bits(w_i1_bits, sig_bits);

            for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                // Compute w[i] limb on-the-fly
                Fp w_i_limb = Fp::zero();
#pragma unroll 1
                for (uint32_t bit = 0; bit < 16; bit++) {
                    w_i_limb += get_w_bit(local_inner, next_inner, i, limb * 16 + bit) *
                                Fp(1u << bit);
                }
                Fp val = w_i_limb + compose_u16_limb(sig_bits, limb);
                SHA2INNER_WRITE_ROUND(V, next_inner, schedule_helper.intermed_4[i][limb], val);
            }
        }
    }

    __device__ __noinline__ void generate_intermed_12(RowSlice local_inner, RowSlice next_inner)
        const {
        // Process one round at a time instead of pre-loading all 2*ROUNDS_PER_ROW rows.
        // This reduces per-function stack from ~1.2KB to ~256B for SHA-256.
        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            // Read w[i+2] bits on-the-fly for small_sig1_bits
            Fp w_i2_bits[V::WORD_BITS];
            for (uint32_t bit = 0; bit < V::WORD_BITS; bit++) {
                w_i2_bits[bit] = get_w_bit(local_inner, next_inner, i + 2, bit);
            }

            Fp sig_bits[V::WORD_BITS];
            small_sig1_bits(w_i2_bits, sig_bits);

            for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                Fp carry = read_carry_fp(next_inner, i, limb);
                Fp prev_carry = (limb > 0) ? read_carry_fp(next_inner, i, limb - 1) : Fp::zero();

                // w7_limb: use pre-stored w_3 helper for i<3, else compute w[i-3] on-the-fly
                Fp w7_limb;
                if (i < 3) {
                    w7_limb = local_inner[SHA2_COL_INDEX(
                        V, Sha2RoundCols, schedule_helper.w_3[i][limb]
                    )];
                } else {
                    // i-3 is in local_inner (i-3 < ROUNDS_PER_ROW since i<ROUNDS_PER_ROW)
                    w7_limb = Fp::zero();
#pragma unroll 1
                    for (uint32_t bit = 0; bit < 16; bit++) {
                        w7_limb +=
                            get_w_bit(local_inner, next_inner, i - 3, limb * 16 + bit) *
                            Fp(1u << bit);
                    }
                }

                // w_cur = w[i+4] limb on-the-fly (i+4 >= ROUNDS_PER_ROW, so from next_inner)
                Fp w_cur = Fp::zero();
#pragma unroll 1
                for (uint32_t bit = 0; bit < 16; bit++) {
                    w_cur +=
                        get_w_bit(local_inner, next_inner, i + 4, limb * 16 + bit) * Fp(1u << bit);
                }

                Fp sum =
                    compose_u16_limb(sig_bits, limb) + w7_limb - carry * Fp(1u << 16) - w_cur +
                    prev_carry;
                Fp intermed = -sum;
                SHA2INNER_WRITE_ROUND(
                    V, local_inner, schedule_helper.intermed_12[i][limb], intermed
                );
            }
        }
    }

    __device__ __noinline__ void generate_default_row(
        RowSlice inner_row,
        const typename V::Word *first_block_prev_hash,
        Fp *carry_a,
        Fp *carry_e,
        uint32_t global_block_idx,
        size_t trace_height
    ) const {
        RowSlice row_idx_flags =
            inner_row.slice_from(SHA2_COL_INDEX(V, Sha2RoundCols, flags.row_idx));
        row_idx_encoder.write_flag_pt(row_idx_flags, V::ROWS_PER_BLOCK);
        SHA2INNER_WRITE_ROUND(V, inner_row, flags.global_block_idx, Fp(global_block_idx));

        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            uint32_t a_idx = V::ROUNDS_PER_ROW - i - 1;
            uint32_t e_idx = V::ROUNDS_PER_ROW - i + 3;
            SHA2INNER_WRITE_BITS_ROUND(V, inner_row, work_vars.a[i], first_block_prev_hash[a_idx]);
            SHA2INNER_WRITE_BITS_ROUND(V, inner_row, work_vars.e[i], first_block_prev_hash[e_idx]);
        }

        if (carry_a && carry_e) {
            for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    SHA2INNER_WRITE_ROUND(
                        V,
                        inner_row,
                        work_vars.carry_a[i][limb],
                        carry_a[(i * V::WORD_U16S + limb) * trace_height]
                    );
                    SHA2INNER_WRITE_ROUND(
                        V,
                        inner_row,
                        work_vars.carry_e[i][limb],
                        carry_e[(i * V::WORD_U16S + limb) * trace_height]
                    );
                }
            }
        }
    }

    __device__ void generate_missing_cells(
        Fp *trace,
        size_t trace_height,
        uint32_t block_idx
    ) const {
        trace += 1; // skip the first row of the trace
        uint32_t block_row_base = block_idx * V::ROWS_PER_BLOCK;
        uint32_t last_round_row = block_row_base + (V::ROUND_ROWS - 2);
        uint32_t digest_row = block_row_base + (V::ROUND_ROWS - 1);
        uint32_t next_block_row_base = block_row_base + V::ROUND_ROWS;

        if (last_round_row >= trace_height || digest_row >= trace_height ||
            next_block_row_base >= trace_height) {
            return;
        }

        RowSlice last_round_row_slice(trace + last_round_row, trace_height);
        RowSlice last_round_inner =
            last_round_row_slice.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
        RowSlice digest_row_slice(trace + digest_row, trace_height);
        RowSlice digest_inner = digest_row_slice.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
        RowSlice next_row_slice(trace + next_block_row_base, trace_height);
        RowSlice next_inner = next_row_slice.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

        generate_intermed_12(last_round_inner, digest_inner);
        generate_intermed_12(digest_inner, next_inner);
        generate_intermed_4(digest_inner, next_inner);
    }
};

// ===== BLOCK HASHER KERNELS =====
template <typename V>
__global__ void sha2_hash_computation(
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    typename V::Word *prev_hashes,
    uint32_t total_num_blocks
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_num_blocks) {
        return;
    }

    if (block_idx >= num_records) {
        return;
    }

    uint32_t offset = record_offsets[block_idx];
    Sha2BlockRecordMut<V> record(records + offset);
#pragma unroll
    for (int i = 0; i < static_cast<int>(V::HASH_WORDS); i++) {
        const uint8_t *ptr = record.prev_state + i * V::WORD_U8S;
        prev_hashes[block_idx * V::HASH_WORDS + i] = word_from_bytes_le<V>(ptr);
    }
}

template <typename V>
__global__ void sha2_first_pass_tracegen(
    Fp *trace,
    size_t trace_height,
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t total_num_blocks,
    typename V::Word *prev_hashes,
    uint32_t /*ptr_max_bits*/,
    uint32_t * /*range_checker_ptr*/,
    uint32_t /*range_checker_num_bins*/,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t /*timestamp_max_bits*/
) {
    uint32_t global_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_block_idx >= total_num_blocks) {
        return;
    }

    uint32_t record_idx = global_block_idx;
    if (record_idx >= num_records) {
        return;
    }

    uint32_t trace_start_row = global_block_idx * V::ROWS_PER_BLOCK;
    if (trace_start_row + V::ROWS_PER_BLOCK > trace_height) {
        return;
    }

    Sha2TraceHelper<V> helper;
    Sha2BlockRecordMut<V> record(records + record_offsets[record_idx]);
    const typename V::Word *prev_hash = prev_hashes + global_block_idx * V::HASH_WORDS;
    const typename V::Word *next_block_prev_hash =
        prev_hashes + ((global_block_idx + 1) % total_num_blocks) * V::HASH_WORDS;

    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);

    // Use a 16-entry circular buffer instead of w_schedule[ROUNDS_PER_BLOCK].
    // Each schedule word w[t] is stored at w_buf[t & (BLOCK_WORDS - 1)].
    // This reduces stack from ROUNDS_PER_BLOCK*sizeof(Word) to BLOCK_WORDS*sizeof(Word).
    // Correctness: the maximum lookback is 16 entries (w[t-16]), so 16 slots suffice.
    typename V::Word w_buf[V::BLOCK_WORDS] = {};
#pragma unroll
    for (int i = 0; i < static_cast<int>(V::BLOCK_WORDS); i++) {
        w_buf[i] = word_from_bytes_be<V>(record.message_bytes + i * V::WORD_U8S);
    }

    typename V::Word a = prev_hash[0];
    typename V::Word b = prev_hash[1];
    typename V::Word c = prev_hash[2];
    typename V::Word d = prev_hash[3];
    typename V::Word e = prev_hash[4];
    typename V::Word f = prev_hash[5];
    typename V::Word g = prev_hash[6];
    typename V::Word h = prev_hash[7];

    for (uint32_t row_in_block = 0; row_in_block < V::ROWS_PER_BLOCK; row_in_block++) {
        uint32_t absolute_row = trace_start_row + row_in_block;
        if (absolute_row >= trace_height) {
            return;
        }

        RowSlice row(trace + absolute_row, trace_height);
        row.fill_zero(0, Sha2Layout<V>::WIDTH);

        if (row_in_block < V::ROUND_ROWS) {
            SHA2_WRITE_ROUND(V, row, request_id, Fp(record_idx));
            RowSlice inner_row = row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
            helper.write_flags_round(inner_row, row_in_block, global_block_idx + 1);

            for (uint32_t j = 0; j < V::ROUNDS_PER_ROW; j++) {
                uint32_t t = row_in_block * V::ROUNDS_PER_ROW + j;
                typename V::Word w_val;
                if (t < V::BLOCK_WORDS) {
                    w_val = w_buf[t & (V::BLOCK_WORDS - 1)];
                } else {
                    typename V::Word nums[4] = {
                        sha2::small_sig1<V>(w_buf[(t - 2) & (V::BLOCK_WORDS - 1)]),
                        w_buf[(t - 7) & (V::BLOCK_WORDS - 1)],
                        sha2::small_sig0<V>(w_buf[(t - 15) & (V::BLOCK_WORDS - 1)]),
                        w_buf[(t - 16) & (V::BLOCK_WORDS - 1)],
                    };
                    w_val = nums[0] + nums[1] + nums[2] + nums[3];
                    w_buf[t & (V::BLOCK_WORDS - 1)] = w_val;

#pragma unroll
                    for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                        uint32_t sum = 0;
#pragma unroll
                        for (auto num : nums) {
                            sum += word_to_u16_limb<V>(num, limb);
                        }
                        if (limb > 0) {
                            size_t carry_base = SHA2_COL_INDEX(
                                V, Sha2RoundCols, message_schedule.carry_or_buffer[j]
                            );
                            sum += inner_row[carry_base + limb * 2 - 2].asUInt32() +
                                   (inner_row[carry_base + limb * 2 - 1].asUInt32() << 1);
                        }
                        uint32_t carry = (sum - word_to_u16_limb<V>(w_val, limb)) >> 16;
                        SHA2INNER_WRITE_ROUND(
                            V,
                            inner_row,
                            message_schedule.carry_or_buffer[j][limb * 2],
                            Fp(carry & 1)
                        );
                        SHA2INNER_WRITE_ROUND(
                            V,
                            inner_row,
                            message_schedule.carry_or_buffer[j][limb * 2 + 1],
                            Fp((carry >> 1) & 1)
                        );
                    }
                }

                SHA2_WRITE_BITS(V, inner_row, Sha2RoundCols, message_schedule.w[j], w_val);

                typename V::Word t1 =
                    h + sha2::big_sig1<V>(e) + sha2::ch<V>(e, f, g) + V::K(t) + w_val;
                typename V::Word t2 = sha2::big_sig0<V>(a) + sha2::maj<V>(a, b, c);

                typename V::Word new_e = d + t1;
                typename V::Word new_a = t1 + t2;

                SHA2_WRITE_BITS(V, inner_row, Sha2RoundCols, work_vars.e[j], new_e);
                SHA2_WRITE_BITS(V, inner_row, Sha2RoundCols, work_vars.a[j], new_a);

#pragma unroll
                for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                    uint32_t t1_limb = word_to_u16_limb<V>(h, limb) +
                                       word_to_u16_limb<V>(sha2::big_sig1<V>(e), limb) +
                                       word_to_u16_limb<V>(sha2::ch<V>(e, f, g), limb) +
                                       word_to_u16_limb<V>(V::K(t), limb) +
                                       word_to_u16_limb<V>(w_val, limb);
                    uint32_t t2_limb = word_to_u16_limb<V>(sha2::big_sig0<V>(a), limb) +
                                       word_to_u16_limb<V>(sha2::maj<V>(a, b, c), limb);

                    uint32_t prev_carry_e =
                        (limb > 0) ? inner_row[SHA2_COL_INDEX(
                                                   V, Sha2RoundCols, work_vars.carry_e[j][limb - 1]
                                               )]
                                         .asUInt32()
                                   : 0;
                    uint32_t prev_carry_a =
                        (limb > 0) ? inner_row[SHA2_COL_INDEX(
                                                   V, Sha2RoundCols, work_vars.carry_a[j][limb - 1]
                                               )]
                                         .asUInt32()
                                   : 0;
                    uint32_t e_sum = t1_limb + word_to_u16_limb<V>(d, limb) + prev_carry_e;
                    uint32_t a_sum = t1_limb + t2_limb + prev_carry_a;
                    uint32_t c_e = (e_sum - word_to_u16_limb<V>(new_e, limb)) >> 16;
                    uint32_t c_a = (a_sum - word_to_u16_limb<V>(new_a, limb)) >> 16;
                    SHA2INNER_WRITE_ROUND(V, inner_row, work_vars.carry_e[j][limb], Fp(c_e));
                    SHA2INNER_WRITE_ROUND(V, inner_row, work_vars.carry_a[j][limb], Fp(c_a));
                    bitwise_lookup.add_range(c_a, c_e);
                }

                if (row_in_block > 0) {
                    typename V::Word w_4 = w_buf[(t - 4) & (V::BLOCK_WORDS - 1)];
                    typename V::Word sig0_w3 =
                        sha2::small_sig0<V>(w_buf[(t - 3) & (V::BLOCK_WORDS - 1)]);
#pragma unroll
                    for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                        uint32_t val =
                            word_to_u16_limb<V>(w_4, limb) + word_to_u16_limb<V>(sig0_w3, limb);
                        SHA2INNER_WRITE_ROUND(
                            V, inner_row, schedule_helper.intermed_4[j][limb], Fp(val)
                        );
                    }
                    if (j < V::ROUNDS_PER_ROW - 1) {
                        typename V::Word w3 = w_buf[(t - 3) & (V::BLOCK_WORDS - 1)];
#pragma unroll
                        for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                            SHA2INNER_WRITE_ROUND(
                                V,
                                inner_row,
                                schedule_helper.w_3[j][limb],
                                Fp(word_to_u16_limb<V>(w3, limb))
                            );
                        }
                    }
                }

                h = g;
                g = f;
                f = e;
                e = new_e;
                d = c;
                c = b;
                b = a;
                a = new_a;
            }
        } else {
            uint32_t digest_row_idx = V::ROUND_ROWS;
            SHA2_WRITE_DIGEST(V, row, request_id, Fp(record_idx));

            RowSlice inner_row = row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
            helper.write_flags_digest(inner_row, digest_row_idx, global_block_idx + 1);

            for (uint32_t j = 0; j < V::ROUNDS_PER_ROW - 1; j++) {
                typename V::Word val =
                    w_buf[(row_in_block * V::ROUNDS_PER_ROW + j - 3) & (V::BLOCK_WORDS - 1)];
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    SHA2INNER_WRITE_DIGEST(
                        V,
                        inner_row,
                        schedule_helper.w_3[j][limb],
                        Fp(word_to_u16_limb<V>(val, limb))
                    );
                }
            }

            // Use a per-iteration scalar instead of final_hash[HASH_WORDS] array to
            // avoid stack allocation. The column index in the macro uses struct field
            // addressing (not the local variable), so this rename is correct.
            for (int i = 0; i < static_cast<int>(V::HASH_WORDS); i++) {
                typename V::Word work_val =
                    (i == 0)
                        ? a
                        : (i == 1
                               ? b
                               : (i == 2
                                      ? c
                                      : (i == 3 ? d
                                                : (i == 4 ? e : (i == 5 ? f : (i == 6 ? g : h))))));
                typename V::Word fh_val = prev_hash[i] + work_val;
                for (uint32_t limb = 0; limb < V::WORD_U8S; limb++) {
                    SHA2INNER_WRITE_DIGEST(
                        V,
                        inner_row,
                        final_hash[i][limb],
                        Fp(word_to_u8_limb<V>(fh_val, limb))
                    );
                }
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    SHA2INNER_WRITE_DIGEST(
                        V,
                        inner_row,
                        prev_hash[i][limb],
                        Fp(word_to_u16_limb<V>(prev_hash[i], limb))
                    );
                }

                for (uint32_t limb = 0; limb < V::WORD_U8S; limb += 2) {
                    uint32_t b0 = word_to_u8_limb<V>(fh_val, limb);
                    uint32_t b1 = word_to_u8_limb<V>(fh_val, limb + 1);
                    bitwise_lookup.add_range(b0, b1);
                }
            }

            for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
                uint32_t a_idx = V::ROUNDS_PER_ROW - i - 1;
                uint32_t e_idx = V::ROUNDS_PER_ROW - i + 3;
                SHA2_WRITE_BITS(
                    V, inner_row, Sha2DigestCols, hash.a[i], next_block_prev_hash[a_idx]
                );
                SHA2_WRITE_BITS(
                    V, inner_row, Sha2DigestCols, hash.e[i], next_block_prev_hash[e_idx]
                );
            }
        }
    }

    for (uint32_t row_in_block = 0; row_in_block < V::ROWS_PER_BLOCK - 1; row_in_block++) {
        uint32_t absolute_row = trace_start_row + row_in_block;
        RowSlice local_row(trace + absolute_row, trace_height);
        RowSlice next_row(trace + absolute_row + 1, trace_height);
        RowSlice local_inner = local_row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
        RowSlice next_inner = next_row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

        if (row_in_block > 0) {
            for (uint32_t j = 0; j < V::ROUNDS_PER_ROW; j++) {
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    Fp intermed_4_val = local_inner[SHA2_COL_INDEX(
                        V, Sha2RoundCols, schedule_helper.intermed_4[j][limb]
                    )];
                    if (row_in_block + 1 == V::ROWS_PER_BLOCK - 1) {
                        SHA2INNER_WRITE_DIGEST(
                            V, next_inner, schedule_helper.intermed_8[j][limb], intermed_4_val
                        );
                    } else {
                        SHA2INNER_WRITE_ROUND(
                            V, next_inner, schedule_helper.intermed_8[j][limb], intermed_4_val
                        );
                    }

                    if (row_in_block >= 2 && row_in_block < V::ROWS_PER_BLOCK - 3) {
                        Fp intermed_8_val = local_inner[SHA2_COL_INDEX(
                            V, Sha2RoundCols, schedule_helper.intermed_8[j][limb]
                        )];
                        SHA2INNER_WRITE_ROUND(
                            V, next_inner, schedule_helper.intermed_12[j][limb], intermed_8_val
                        );
                    }
                }
            }
        }

        if (row_in_block == V::ROWS_PER_BLOCK - 2) {
            helper.generate_carry_ae(local_inner, next_inner);
            helper.generate_intermed_4(local_inner, next_inner);
        }

        if (row_in_block < V::MESSAGE_ROWS - 1) {
            helper.generate_intermed_12(local_inner, next_inner);
        }
    }
}

template <typename V>
__global__ void sha2_fill_first_dummy_row(Fp *trace, size_t trace_height, size_t rows_used) {
    uint32_t row_idx = rows_used;
    uint32_t total_blocks = rows_used / V::ROWS_PER_BLOCK;
    uint32_t padding_global_block_idx = total_blocks + 1;

    uint32_t digest_row = V::ROUND_ROWS;
    if (digest_row >= trace_height) {
        return;
    }

    RowSlice digest(trace + digest_row, trace_height);
    RowSlice digest_inner = digest.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

    typename V::Word prev_hash[V::HASH_WORDS];
    for (uint32_t i = 0; i < V::HASH_WORDS; i++) {
        typename V::Word acc = 0;
        for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
            size_t base = SHA2_COL_INDEX(V, Sha2DigestCols, prev_hash[i][limb]);
            uint32_t limb_val = digest_inner[base].asUInt32();
            acc |= static_cast<typename V::Word>(limb_val) << (16 * limb);
        }
        prev_hash[i] = acc;
    }

    RowSlice row(trace + row_idx, trace_height);
    uint32_t intermed_4_offset =
        SHA2_COL_INDEX(V, Sha2BlockHasherRoundCols, inner.schedule_helper.intermed_4);
    uint32_t intermed_8_offset =
        SHA2_COL_INDEX(V, Sha2BlockHasherRoundCols, inner.schedule_helper.intermed_8);
    row.fill_zero(0, intermed_4_offset);
    row.fill_zero(intermed_8_offset, Sha2Layout<V>::WIDTH - intermed_8_offset);
    SHA2_WRITE_ROUND(V, row, request_id, Fp::zero());
    RowSlice inner_row = row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

    Sha2TraceHelper<V> helper;
    helper.generate_default_row(
        inner_row,
        prev_hash,
        nullptr,
        nullptr,
        padding_global_block_idx,
        trace_height
    );

    helper.generate_carry_ae(inner_row, inner_row);
}

template <typename V>
__global__ void sha2_second_pass_dependencies(Fp *trace, size_t trace_height, size_t total_blocks) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) {
        return;
    }

    Sha2TraceHelper<V> helper;
    helper.generate_missing_cells(trace, trace_height, block_idx);
}

template <typename V>
__global__ void sha2_fill_invalid_rows(
    Fp *d_trace,
    size_t trace_height,
    size_t rows_used,
    typename V::Word *d_prev_hashes
) {
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t first_dummy_row_idx = rows_used;
    uint32_t total_blocks = rows_used / V::ROWS_PER_BLOCK;
    uint32_t padding_global_block_idx = total_blocks + 1;
    // skip the first dummy row, since it is already filled
    uint32_t row_idx = first_dummy_row_idx + thread_idx + 1;
    if (row_idx >= trace_height) {
        return;
    }

    RowSlice first_dummy_row(d_trace + first_dummy_row_idx, trace_height);
    RowSlice first_dummy_row_inner = first_dummy_row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

    Fp *first_dummy_row_carry_a =
        &first_dummy_row_inner[SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.carry_a)];
    Fp *first_dummy_row_carry_e =
        &first_dummy_row_inner[SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.carry_e)];

    RowSlice dst(d_trace + row_idx, trace_height);
    dst.fill_zero(0, Sha2Layout<V>::WIDTH);
    RowSlice dst_inner = dst.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

    Sha2TraceHelper<V> helper;
    helper.generate_default_row(
        dst_inner,
        &d_prev_hashes[0],
        first_dummy_row_carry_a,
        first_dummy_row_carry_e,
        padding_global_block_idx,
        trace_height
    );
}

template <typename V>
__global__ void sha2_fill_wraparound(Fp *trace, size_t trace_height) {
    if (trace_height < 2) {
        return;
    }

    Sha2TraceHelper<V> helper;
    RowSlice first_row(trace, trace_height);
    RowSlice first_inner = first_row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
    RowSlice last_row(trace + (trace_height - 1), trace_height);
    RowSlice last_inner = last_row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

    helper.generate_intermed_4(last_inner, first_inner);
    helper.generate_intermed_12(last_inner, first_inner);
}

// ===== HOST LAUNCHER FUNCTIONS =====

template <typename V>
int launch_sha2_hash_computation(
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    typename V::Word *d_prev_hashes,
    uint32_t total_num_blocks
) {
    auto [grid_size, block_size] = kernel_launch_params(num_records, 256);

    sha2_hash_computation<V><<<grid_size, block_size>>>(
        d_records, num_records, d_record_offsets, d_prev_hashes, total_num_blocks
    );

    return CHECK_KERNEL();
}

template <typename V>
int launch_sha2_first_pass_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t total_num_blocks,
    typename V::Word *d_prev_hashes,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    auto [grid_size, block_size] = kernel_launch_params(total_num_blocks, 256);

    sha2_first_pass_tracegen<V><<<grid_size, block_size>>>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        total_num_blocks,
        d_prev_hashes,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );

    return CHECK_KERNEL();
}

template <typename V>
int launch_sha2_second_pass_dependencies(Fp *d_trace, size_t trace_height, size_t rows_used) {
    size_t total_blocks = rows_used / V::ROWS_PER_BLOCK;
    auto [grid_size, block_size] = kernel_launch_params(total_blocks, 256);
    sha2_second_pass_dependencies<V>
        <<<grid_size, block_size>>>(d_trace, trace_height, total_blocks);
    if (auto err = CHECK_KERNEL() != 0) {
        return err;
    }

    sha2_fill_wraparound<V><<<1, 1>>>(d_trace, trace_height);
    return CHECK_KERNEL();
}

template <typename V>
int launch_sha2_fill_invalid_rows(
    Fp *d_trace,
    size_t trace_height,
    size_t rows_used,
    typename V::Word *d_prev_hashes
) {
    sha2_fill_first_dummy_row<V><<<1, 1>>>(d_trace, trace_height, rows_used);
    if (CHECK_KERNEL() != 0) {
        return -1;
    }

    auto [grid_size, block_size] = kernel_launch_params(trace_height - rows_used, 256);
    sha2_fill_invalid_rows<V>
        <<<grid_size, block_size>>>(d_trace, trace_height, rows_used, d_prev_hashes);
    return CHECK_KERNEL();
}

// Explicit instantiations for SHA-256 and SHA-512
extern "C" {
int launch_sha256_hash_computation(
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t *d_prev_hashes,
    uint32_t total_num_blocks
) {
    return launch_sha2_hash_computation<Sha256Variant>(
        d_records,
        num_records,
        d_record_offsets,
        reinterpret_cast<uint32_t *>(d_prev_hashes),
        total_num_blocks
    );
}

int launch_sha512_hash_computation(
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint64_t *d_prev_hashes,
    uint32_t total_num_blocks
) {
    return launch_sha2_hash_computation<Sha512Variant>(
        d_records, num_records, d_record_offsets, d_prev_hashes, total_num_blocks
    );
}

int launch_sha256_first_pass_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t total_num_blocks,
    uint32_t *d_prev_hashes,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    return launch_sha2_first_pass_tracegen<Sha256Variant>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        total_num_blocks,
        d_prev_hashes,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
}

int launch_sha512_first_pass_tracegen(
    Fp *d_trace,
    size_t trace_height,
    uint8_t *d_records,
    size_t num_records,
    size_t *d_record_offsets,
    uint32_t total_num_blocks,
    uint64_t *d_prev_hashes,
    uint32_t ptr_max_bits,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits
) {
    return launch_sha2_first_pass_tracegen<Sha512Variant>(
        d_trace,
        trace_height,
        d_records,
        num_records,
        d_record_offsets,
        total_num_blocks,
        d_prev_hashes,
        ptr_max_bits,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
}

int launch_sha256_second_pass_dependencies(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_second_pass_dependencies<Sha256Variant>(d_trace, trace_height, rows_used);
}
int launch_sha512_second_pass_dependencies(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_second_pass_dependencies<Sha512Variant>(d_trace, trace_height, rows_used);
}
int launch_sha256_fill_invalid_rows(
    Fp *d_trace,
    size_t trace_height,
    size_t rows_used,
    uint32_t *d_prev_hashes
) {
    return launch_sha2_fill_invalid_rows<Sha256Variant>(
        d_trace, trace_height, rows_used, d_prev_hashes
    );
}
int launch_sha512_fill_invalid_rows(
    Fp *d_trace,
    size_t trace_height,
    size_t rows_used,
    uint64_t *d_prev_hashes
) {
    return launch_sha2_fill_invalid_rows<Sha512Variant>(
        d_trace, trace_height, rows_used, d_prev_hashes
    );
}
}
