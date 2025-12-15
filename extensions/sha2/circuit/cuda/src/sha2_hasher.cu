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

// Read a word from bits stored in a row slice
template <typename V>
__device__ __forceinline__ typename V::Word word_from_bits(RowSlice row, size_t col_index) {
    typename V::Word result = 0;
#pragma unroll
    for (size_t i = 0; i < V::WORD_BITS; i++) {
        Fp bit = row[col_index + i];
        result |= (static_cast<typename V::Word>(bit.asUInt32()) << i);
    }
    return result;
}

// === Shared helpers mirroring the CPU tracegen structure ===
template <typename V> struct Sha2TraceHelper {
    Encoder row_idx_encoder;

    __device__ Sha2TraceHelper()
        : row_idx_encoder(static_cast<uint32_t>(V::ROWS_PER_BLOCK + 1), 2, false) {}

    __device__ __forceinline__ size_t base_a(bool is_digest, uint32_t row_idx) const {
        return is_digest
                   ? SHA2_COL_INDEX(V, Sha2DigestCols, hash.a[row_idx])
                   : SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.a[row_idx]);
    }

    __device__ __forceinline__ size_t base_e(bool is_digest, uint32_t row_idx) const {
        return is_digest
                   ? SHA2_COL_INDEX(V, Sha2DigestCols, hash.e[row_idx])
                   : SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.e[row_idx]);
    }

    __device__ __forceinline__ size_t base_carry_a(bool is_digest, uint32_t row_idx) const {
        return is_digest
                   ? SHA2_COL_INDEX(V, Sha2DigestCols, hash.carry_a[row_idx])
                   : SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.carry_a[row_idx]);
    }

    __device__ __forceinline__ size_t base_carry_e(bool is_digest, uint32_t row_idx) const {
        return is_digest
                   ? SHA2_COL_INDEX(V, Sha2DigestCols, hash.carry_e[row_idx])
                   : SHA2_COL_INDEX(V, Sha2RoundCols, work_vars.carry_e[row_idx]);
    }

    __device__ __forceinline__ typename V::Word read_a(RowSlice inner, uint32_t row_idx, bool is_digest) const {
        return word_from_bits<V>(inner, base_a(is_digest, row_idx));
    }

    __device__ __forceinline__ typename V::Word read_e(RowSlice inner, uint32_t row_idx, bool is_digest) const {
        return word_from_bits<V>(inner, base_e(is_digest, row_idx));
    }

    __device__ __forceinline__ typename V::Word read_w(RowSlice inner, uint32_t j) const {
        size_t base = SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.w[j]);
        return word_from_bits<V>(inner, base);
    }

    __device__ void write_flags_round(RowSlice inner_row, uint32_t row_idx, uint32_t global_block_idx) const {
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
        SHA2INNER_WRITE_ROUND(V, inner_row, flags.local_block_idx, Fp(0));
    }

    __device__ void write_flags_digest(RowSlice inner_row, uint32_t row_idx, uint32_t global_block_idx) const {
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.is_round_row, Fp::zero());
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.is_first_4_rows, Fp::zero());
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.is_digest_row, Fp::one());
        RowSlice row_idx_flags =
            inner_row.slice_from(SHA2_COL_INDEX(V, Sha2DigestCols, flags.row_idx));
        row_idx_encoder.write_flag_pt(row_idx_flags, row_idx);
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.global_block_idx, Fp(global_block_idx));
        SHA2INNER_WRITE_DIGEST(V, inner_row, flags.local_block_idx, Fp(0));
    }

    __device__ void generate_carry_ae(RowSlice local_inner, RowSlice next_inner, bool next_is_digest) const {
        typename V::Word a_rows[2 * V::ROUNDS_PER_ROW];
        typename V::Word e_rows[2 * V::ROUNDS_PER_ROW];
#pragma unroll
        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            typename V::Word a_val = read_a(local_inner, i, false);
            typename V::Word e_val = read_e(local_inner, i, false);
            a_rows[i] = a_val;
            a_rows[i + V::ROUNDS_PER_ROW] = read_a(next_inner, i, next_is_digest);
            e_rows[i] = e_val;
            e_rows[i + V::ROUNDS_PER_ROW] = read_e(next_inner, i, next_is_digest);
        }

        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            typename V::Word cur_a = a_rows[i + 4];
            typename V::Word cur_e = e_rows[i + 4];
            typename V::Word sig_a = sha2::big_sig0<V>(a_rows[i + 3]);
            typename V::Word sig_e = sha2::big_sig1<V>(e_rows[i + 3]);
            typename V::Word maj_abc = sha2::maj<V>(a_rows[i + 3], a_rows[i + 2], a_rows[i + 1]);
            typename V::Word ch_efg = sha2::ch<V>(e_rows[i + 3], e_rows[i + 2], e_rows[i + 1]);
            typename V::Word d_val = a_rows[i];
            typename V::Word h_val = e_rows[i];

            uint32_t prev_carry_a = 0;
            uint32_t prev_carry_e = 0;
            for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                uint32_t t1_sum = word_to_u16_limb<V>(h_val, limb) +
                                  word_to_u16_limb<V>(sig_e, limb) +
                                  word_to_u16_limb<V>(ch_efg, limb);
                uint32_t t2_sum = word_to_u16_limb<V>(sig_a, limb) +
                                  word_to_u16_limb<V>(maj_abc, limb);
                uint32_t d_limb = word_to_u16_limb<V>(d_val, limb);
                uint32_t cur_a_limb = word_to_u16_limb<V>(cur_a, limb);
                uint32_t cur_e_limb = word_to_u16_limb<V>(cur_e, limb);

                uint32_t e_sum = d_limb + t1_sum + prev_carry_e;
                uint32_t a_sum = t1_sum + t2_sum + prev_carry_a;
                uint32_t carry_e = (e_sum - cur_e_limb) >> 16;
                uint32_t carry_a = (a_sum - cur_a_limb) >> 16;

                if (next_is_digest) {
                    SHA2INNER_WRITE_DIGEST(V, next_inner, hash.carry_e[i][limb], Fp(carry_e));
                    SHA2INNER_WRITE_DIGEST(V, next_inner, hash.carry_a[i][limb], Fp(carry_a));
                } else {
                    SHA2INNER_WRITE_ROUND(V, next_inner, work_vars.carry_e[i][limb], Fp(carry_e));
                    SHA2INNER_WRITE_ROUND(V, next_inner, work_vars.carry_a[i][limb], Fp(carry_a));
                }

                prev_carry_e = carry_e;
                prev_carry_a = carry_a;
            }
        }
    }

    __device__ void generate_intermed_4(
        RowSlice local_inner, RowSlice next_inner, bool next_is_digest
    ) const {
        typename V::Word w_vals[2 * V::ROUNDS_PER_ROW];
#pragma unroll
        for (uint32_t j = 0; j < V::ROUNDS_PER_ROW; j++) {
            w_vals[j] = read_w(local_inner, j);
            w_vals[j + V::ROUNDS_PER_ROW] = read_w(next_inner, j);
        }

        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            typename V::Word sig_w = sha2::small_sig0<V>(w_vals[i + 1]);
#pragma unroll
            for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                uint32_t val =
                    word_to_u16_limb<V>(w_vals[i], limb) + word_to_u16_limb<V>(sig_w, limb);
                if (next_is_digest) {
                    SHA2INNER_WRITE_DIGEST(V, next_inner, schedule_helper.intermed_4[i][limb], Fp(val));
                } else {
                    SHA2INNER_WRITE_ROUND(V, next_inner, schedule_helper.intermed_4[i][limb], Fp(val));
                }
            }
        }
    }

    __device__ void generate_intermed_12(
        RowSlice local_inner, RowSlice next_inner, bool local_is_digest
    ) const {
        typename V::Word w_vals[2 * V::ROUNDS_PER_ROW];
#pragma unroll
        for (uint32_t j = 0; j < V::ROUNDS_PER_ROW; j++) {
            w_vals[j] = read_w(local_inner, j);
            w_vals[j + V::ROUNDS_PER_ROW] = read_w(next_inner, j);
        }

        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            typename V::Word sig_w2 = sha2::small_sig1<V>(w_vals[i + 2]);
            typename V::Word w7 =
                (i < 3)
                    ? [=]() {
                          typename V::Word acc = 0;
                          for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                              size_t base =
                                  SHA2_COL_INDEX(V, Sha2RoundCols, schedule_helper.w_3[i][limb]);
                              uint32_t limb_val = local_inner[base].asUInt32();
                              acc |= static_cast<typename V::Word>(limb_val) << (16 * limb);
                          }
                          return acc;
                      }()
                    : w_vals[i - 3];
            typename V::Word w_cur = w_vals[i + 4];
            for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                size_t base = SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.carry_or_buffer) +
                              i * V::WORD_U8S * 2;
                uint32_t carry = next_inner[base + limb * 2].asUInt32() +
                                 (next_inner[base + limb * 2 + 1].asUInt32() << 1);
                uint32_t prev_carry =
                    (limb > 0)
                        ? next_inner[base + limb * 2 - 2].asUInt32() +
                              (next_inner[base + limb * 2 - 1].asUInt32() << 1)
                        : 0;
                int64_t sum = static_cast<int64_t>(word_to_u16_limb<V>(sig_w2, limb)) +
                              static_cast<int64_t>(word_to_u16_limb<V>(w7, limb)) -
                              static_cast<int64_t>(carry << 16) -
                              static_cast<int64_t>(word_to_u16_limb<V>(w_cur, limb)) +
                              static_cast<int64_t>(prev_carry);
                uint32_t intermed = static_cast<uint32_t>(-sum);
                if (local_is_digest) {
                    SHA2INNER_WRITE_DIGEST(V, local_inner, schedule_helper.intermed_12[i][limb], intermed);
                } else {
                    SHA2INNER_WRITE_ROUND(V, local_inner, schedule_helper.intermed_12[i][limb], intermed);
                }
            }
        }
    }

    __device__ void generate_default_row(
        RowSlice inner_row,
        const typename V::Word *first_block_prev_hash,
        const uint32_t *carry_a,
        const uint32_t *carry_e
    ) const {
        RowSlice row_idx_flags =
            inner_row.slice_from(SHA2_COL_INDEX(V, Sha2RoundCols, flags.row_idx));
        row_idx_encoder.write_flag_pt(row_idx_flags, V::ROWS_PER_BLOCK);

        for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
            uint32_t a_idx = V::ROUNDS_PER_ROW - i - 1;
            uint32_t e_idx = V::ROUNDS_PER_ROW - i + 3;
            SHA2INNER_WRITE_ROUND(V, inner_row, work_vars.a[i], first_block_prev_hash[a_idx]);
            SHA2INNER_WRITE_ROUND(V, inner_row, work_vars.e[i], first_block_prev_hash[e_idx]);
        }

        if (carry_a && carry_e) {
            for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    SHA2INNER_WRITE_ROUND(
                        V,
                        inner_row,
                        work_vars.carry_a[i][limb],
                        Fp(carry_a[i * V::WORD_U16S + limb])
                    );
                    SHA2INNER_WRITE_ROUND(
                        V,
                        inner_row,
                        work_vars.carry_e[i][limb],
                        Fp(carry_e[i * V::WORD_U16S + limb])
                    );
                }
            }
        }
    }

    __device__ void generate_missing_cells(Fp *trace, size_t trace_height, uint32_t block_idx) const {
        uint32_t block_row_base = block_idx * V::ROWS_PER_BLOCK;
        uint32_t last_round_row = block_row_base + (V::ROUND_ROWS - 2);
        uint32_t digest_row = block_row_base + (V::ROUND_ROWS - 1);
        uint32_t next_block_row_base = block_row_base + V::ROUND_ROWS;

        if (last_round_row >= trace_height || digest_row >= trace_height ||
            next_block_row_base >= trace_height) {
            return;
        }

        RowSlice last_round_row_slice(trace + last_round_row, trace_height);
        RowSlice last_round_inner = last_round_row_slice.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
        RowSlice digest_row_slice(trace + digest_row, trace_height);
        RowSlice digest_inner = digest_row_slice.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);
        RowSlice next_row_slice(trace + next_block_row_base, trace_height);
        RowSlice next_inner = next_row_slice.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

        generate_intermed_12(last_round_inner, digest_inner, false);
        generate_intermed_12(digest_inner, next_inner, false);
        generate_intermed_4(digest_inner, next_inner, true);
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
    uint32_t *block_to_record_idx,
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

    uint32_t record_idx = block_to_record_idx[global_block_idx];
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

    typename V::Word w_schedule[V::ROUNDS_PER_BLOCK] = {};
#pragma unroll
    for (int i = 0; i < static_cast<int>(V::BLOCK_WORDS); i++) {
        w_schedule[i] = word_from_bytes_be<V>(record.message_bytes + i * V::WORD_U8S);
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
                    w_val = w_schedule[t];
                } else {
                    typename V::Word nums[4] = {
                        sha2::small_sig1<V>(w_schedule[t - 2]),
                        w_schedule[t - 7],
                        sha2::small_sig0<V>(w_schedule[t - 15]),
                        w_schedule[t - 16],
                    };
                    w_val = nums[0] + nums[1] + nums[2] + nums[3];
                    w_schedule[t] = w_val;

#pragma unroll
                    for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                        uint32_t sum = 0;
#pragma unroll
                        for (auto num : nums) {
                            sum += word_to_u16_limb<V>(num, limb);
                        }
                        if (limb > 0) {
                            size_t carry_base =
                                SHA2_COL_INDEX(V, Sha2RoundCols, message_schedule.carry_or_buffer) +
                                j * V::WORD_U8S * 2;
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

                    uint32_t prev_carry_e = (limb > 0)
                                                ? inner_row[SHA2_COL_INDEX(
                                                                V,
                                                                Sha2RoundCols,
                                                                work_vars.carry_e[j][limb - 1]
                                                            )]
                                                      .asUInt32()
                                                : 0;
                    uint32_t prev_carry_a = (limb > 0)
                                                ? inner_row[SHA2_COL_INDEX(
                                                                V,
                                                                Sha2RoundCols,
                                                                work_vars.carry_a[j][limb - 1]
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
                    typename V::Word w_4 = w_schedule[t - 4];
                    typename V::Word sig0_w3 = sha2::small_sig0<V>(w_schedule[t - 3]);
#pragma unroll
                    for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                        uint32_t val =
                            word_to_u16_limb<V>(w_4, limb) + word_to_u16_limb<V>(sig0_w3, limb);
                        SHA2INNER_WRITE_ROUND(
                            V, inner_row, schedule_helper.intermed_4[j][limb], Fp(val)
                        );
                    }
                    if (j < V::ROUNDS_PER_ROW - 1) {
                        typename V::Word w3 = w_schedule[t - 3];
#pragma unroll
                        for (int limb = 0; limb < static_cast<int>(V::WORD_U16S); limb++) {
                            SHA2INNER_WRITE_ROUND(
                                V, inner_row, schedule_helper.w_3[j][limb], Fp(word_to_u16_limb<V>(w3, limb))
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
                uint32_t idx = (V::ROUND_ROWS - 1) * V::ROUNDS_PER_ROW + j;
                typename V::Word val = w_schedule[idx - 3];
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    SHA2INNER_WRITE_DIGEST(
                        V, inner_row, schedule_helper.w_3[j][limb], Fp(word_to_u16_limb<V>(val, limb))
                    );
                }
            }

            typename V::Word final_hash[V::HASH_WORDS];
            for (int i = 0; i < static_cast<int>(V::HASH_WORDS); i++) {
                typename V::Word work_val =
                    (i == 0)
                        ? a
                        : (i == 1
                               ? b
                               : (i == 2 ? c : (i == 3 ? d : (i == 4 ? e : (i == 5 ? f : (i == 6 ? g : h))))));
                final_hash[i] = prev_hash[i] + work_val;
                for (uint32_t limb = 0; limb < V::WORD_U8S; limb++) {
                    SHA2INNER_WRITE_DIGEST(
                        V, inner_row, final_hash[i][limb], Fp(word_to_u8_limb<V>(final_hash[i], limb))
                    );
                }
                for (uint32_t limb = 0; limb < V::WORD_U16S; limb++) {
                    SHA2INNER_WRITE_DIGEST(
                        V, inner_row, prev_hash[i][limb], Fp(word_to_u16_limb<V>(prev_hash[i], limb))
                    );
                }

                for (uint32_t limb = 0; limb < V::WORD_U8S; limb += 2) {
                    uint32_t b0 = word_to_u8_limb<V>(final_hash[i], limb);
                    uint32_t b1 = word_to_u8_limb<V>(final_hash[i], limb + 1);
                    bitwise_lookup.add_range(b0, b1);
                }
            }

            for (uint32_t i = 0; i < V::ROUNDS_PER_ROW; i++) {
                uint32_t a_idx = V::ROUNDS_PER_ROW - i - 1;
                uint32_t e_idx = V::ROUNDS_PER_ROW - i + 3;
                SHA2_WRITE_BITS(V, inner_row, Sha2DigestCols, hash.a[i], next_block_prev_hash[a_idx]);
                SHA2_WRITE_BITS(V, inner_row, Sha2DigestCols, hash.e[i], next_block_prev_hash[e_idx]);
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
            helper.generate_carry_ae(local_inner, next_inner, true);
            helper.generate_intermed_4(local_inner, next_inner, true);
        }

        if (row_in_block < V::MESSAGE_ROWS - 1) {
            helper.generate_intermed_12(local_inner, next_inner, false);
        }
    }
}

template <typename V>
__global__ void sha2_fill_first_dummy_row(Fp *trace, size_t trace_height, size_t rows_used) {
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row_idx = rows_used + thread_idx;
    if (row_idx >= trace_height) {
        return;
    }

    if (row_idx != rows_used) {
        return;
    }

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
    row.fill_zero(0, Sha2Layout<V>::WIDTH);
    SHA2_WRITE_ROUND(V, row, request_id, Fp::zero());
    RowSlice inner_row = row.slice_from(Sha2Layout<V>::INNER_COLUMN_OFFSET);

    Sha2TraceHelper<V> helper;
    helper.generate_default_row(inner_row, prev_hash, nullptr, nullptr);

    helper.generate_carry_ae(inner_row, inner_row, false);
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
__global__ void sha2_fill_invalid_rows(Fp *d_trace, size_t trace_height, size_t rows_used) {
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row_idx = rows_used + thread_idx;
    if (row_idx >= trace_height) {
        return;
    }

    RowSlice src(d_trace + rows_used - 1, trace_height);
    RowSlice dst(d_trace + row_idx, trace_height);
    for (size_t c = 0; c < Sha2Layout<V>::WIDTH; c++) {
        dst[c] = src[c];
    }
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
    uint32_t *d_block_to_record_idx,
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
        d_block_to_record_idx,
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
int launch_sha2_fill_first_dummy_row(Fp *d_trace, size_t trace_height, size_t rows_used) {
    auto [grid_size, block_size] = kernel_launch_params(trace_height - rows_used, 256);
    sha2_fill_first_dummy_row<V><<<grid_size, block_size>>>(d_trace, trace_height, rows_used);
    return CHECK_KERNEL();
}

template <typename V>
int launch_sha2_second_pass_dependencies(Fp *d_trace, size_t trace_height, size_t rows_used) {
    size_t total_blocks = rows_used / V::ROWS_PER_BLOCK;
    auto [grid_size, block_size] = kernel_launch_params(total_blocks, 256);
    sha2_second_pass_dependencies<V>
        <<<grid_size, block_size>>>(d_trace, trace_height, total_blocks);
    return CHECK_KERNEL();
}

template <typename V>
int launch_sha2_fill_invalid_rows(Fp *d_trace, size_t trace_height, size_t rows_used) {
    auto [grid_size, block_size] = kernel_launch_params(trace_height - rows_used, 256);
    sha2_fill_invalid_rows<V><<<grid_size, block_size>>>(d_trace, trace_height, rows_used);
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
    uint32_t *d_block_to_record_idx,
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
        d_block_to_record_idx,
        total_num_blocks,
        reinterpret_cast<uint32_t *>(d_prev_hashes),
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
    uint32_t *d_block_to_record_idx,
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
        d_block_to_record_idx,
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

int launch_sha256_fill_first_dummy_row(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_fill_first_dummy_row<Sha256Variant>(d_trace, trace_height, rows_used);
}
int launch_sha512_fill_first_dummy_row(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_fill_first_dummy_row<Sha512Variant>(d_trace, trace_height, rows_used);
}

int launch_sha256_second_pass_dependencies(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_second_pass_dependencies<Sha256Variant>(d_trace, trace_height, rows_used);
}
int launch_sha512_second_pass_dependencies(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_second_pass_dependencies<Sha512Variant>(d_trace, trace_height, rows_used);
}
int launch_sha256_fill_invalid_rows(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_fill_invalid_rows<Sha256Variant>(d_trace, trace_height, rows_used);
}
int launch_sha512_fill_invalid_rows(Fp *d_trace, size_t trace_height, size_t rows_used) {
    return launch_sha2_fill_invalid_rows<Sha512Variant>(d_trace, trace_height, rows_used);
}
}
