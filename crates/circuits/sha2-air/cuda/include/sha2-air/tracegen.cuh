#pragma once

#include "columns.cuh"
#include "primitives/constants.h"
#include "primitives/encoder.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "utils.cuh"

using namespace riscv;
using namespace sha256;

__device__ void generate_carry_ae(RowSlice local_row, RowSlice next_row) {
    Fp a_bits[SHA256_ROUNDS_PER_ROW * 2][SHA256_WORD_BITS];
    Fp e_bits[SHA256_ROUNDS_PER_ROW * 2][SHA256_WORD_BITS];
    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            a_bits[i][bit] = local_row[COL_INDEX(Sha256RoundCols, work_vars.a[i][bit])];
            e_bits[i][bit] = local_row[COL_INDEX(Sha256RoundCols, work_vars.e[i][bit])];
        }
    }

    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            a_bits[i + SHA256_ROUNDS_PER_ROW][bit] =
                next_row[COL_INDEX(Sha256RoundCols, work_vars.a[i][bit])];
            e_bits[i + SHA256_ROUNDS_PER_ROW][bit] =
                next_row[COL_INDEX(Sha256RoundCols, work_vars.e[i][bit])];
        }
    }

    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        Fp sig_a_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            sig_a_bits[bit] =
                (a_bits[i + 3][(bit + 2) & 31] + a_bits[i + 3][(bit + 13) & 31] -
                 Fp(2) * a_bits[i + 3][(bit + 2) & 31] * a_bits[i + 3][(bit + 13) & 31]) +
                a_bits[i + 3][(bit + 22) & 31] -
                Fp(2) *
                    (a_bits[i + 3][(bit + 2) & 31] + a_bits[i + 3][(bit + 13) & 31] -
                     Fp(2) * a_bits[i + 3][(bit + 2) & 31] * a_bits[i + 3][(bit + 13) & 31]) *
                    a_bits[i + 3][(bit + 22) & 31];
        }

        Fp sig_e_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            sig_e_bits[bit] =
                (e_bits[i + 3][(bit + 6) & 31] + e_bits[i + 3][(bit + 11) & 31] -
                 Fp(2) * e_bits[i + 3][(bit + 6) & 31] * e_bits[i + 3][(bit + 11) & 31]) +
                e_bits[i + 3][(bit + 25) & 31] -
                Fp(2) *
                    (e_bits[i + 3][(bit + 6) & 31] + e_bits[i + 3][(bit + 11) & 31] -
                     Fp(2) * e_bits[i + 3][(bit + 6) & 31] * e_bits[i + 3][(bit + 11) & 31]) *
                    e_bits[i + 3][(bit + 25) & 31];
        }

        Fp maj_abc_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            maj_abc_bits[bit] =
                a_bits[i + 3][bit] * a_bits[i + 2][bit] + a_bits[i + 3][bit] * a_bits[i + 1][bit] +
                a_bits[i + 2][bit] * a_bits[i + 1][bit] -
                Fp(2) * a_bits[i + 3][bit] * a_bits[i + 2][bit] * a_bits[i + 1][bit];
        }

        Fp ch_efg_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            ch_efg_bits[bit] = e_bits[i + 3][bit] * e_bits[i + 2][bit] + e_bits[i + 1][bit] -
                               e_bits[i + 3][bit] * e_bits[i + 1][bit];
        }
        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            Fp t1_limb_sum = Fp::zero();
#pragma unroll 1
            for (int bit = 0; bit < 16; bit++) {
                t1_limb_sum += (e_bits[i][j * 16 + bit] + sig_e_bits[j * 16 + bit] +
                                ch_efg_bits[j * 16 + bit]) *
                               Fp(1 << bit);
            }
            Fp t2_limb_sum = Fp::zero();
#pragma unroll 1
            for (int bit = 0; bit < 16; bit++) {
                t2_limb_sum +=
                    (sig_a_bits[j * 16 + bit] + maj_abc_bits[j * 16 + bit]) * Fp(1 << bit);
            }

            Fp d_limb = Fp::zero();
            Fp cur_a_limb = Fp::zero();
            Fp cur_e_limb = Fp::zero();
#pragma unroll 1
            for (int bit = 0; bit < 16; bit++) {
                d_limb += (a_bits[i][j * 16 + bit] * Fp(1 << bit));
                cur_a_limb += (a_bits[i + 4][j * 16 + bit] * Fp(1 << bit));
                cur_e_limb += (e_bits[i + 4][j * 16 + bit] * Fp(1 << bit));
            }
            Fp prev_carry_e =
                (j == 0) ? Fp::zero()
                         : next_row[COL_INDEX(Sha256RoundCols, work_vars.carry_e[i][j - 1])];
            Fp carry_e_numerator = d_limb + t1_limb_sum + prev_carry_e - cur_e_limb;
            Fp carry_e = carry_e_numerator.mul_2exp_neg_n(16);

            Fp prev_carry_a =
                (j == 0) ? Fp::zero()
                         : next_row[COL_INDEX(Sha256RoundCols, work_vars.carry_a[i][j - 1])];

            Fp carry_a_numerator = t1_limb_sum + t2_limb_sum + prev_carry_a - cur_a_limb;
            Fp carry_a = carry_a_numerator.mul_2exp_neg_n(16);
            SHA256INNER_WRITE_ROUND(next_row, work_vars.carry_e[i][j], carry_e);
            SHA256INNER_WRITE_ROUND(next_row, work_vars.carry_a[i][j], carry_a);
        }
    }
}

__device__ void generate_intermed_4(RowSlice local_row, RowSlice next_row) {
    Fp w_bits[SHA256_ROUNDS_PER_ROW * 2][SHA256_WORD_BITS];

    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            w_bits[i][bit] = local_row[COL_INDEX(Sha256RoundCols, message_schedule.w[i][bit])];
        }
    }

    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            w_bits[i + SHA256_ROUNDS_PER_ROW][bit] =
                next_row[COL_INDEX(Sha256RoundCols, message_schedule.w[i][bit])];
        }
    }

    Fp w_limbs[SHA256_ROUNDS_PER_ROW * 2][SHA256_WORD_U16S];
    for (int i = 0; i < SHA256_ROUNDS_PER_ROW * 2; i++) {
        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            w_limbs[i][j] = Fp::zero();
            for (int bit = 0; bit < 16; bit++) {
                w_limbs[i][j] = w_limbs[i][j] + w_bits[i][j * 16 + bit] * Fp(1 << bit);
            }
        }
    }

    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        Fp sig_w_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            sig_w_bits[bit] =
                (w_bits[i + 1][(bit + 7) & 31] + w_bits[i + 1][(bit + 18) & 31] -
                 Fp(2) * w_bits[i + 1][(bit + 7) & 31] * w_bits[i + 1][(bit + 18) & 31]) +
                ((bit + 3 < 32) ? w_bits[i + 1][bit + 3] : Fp::zero()) -
                Fp(2) *
                    (w_bits[i + 1][(bit + 7) & 31] + w_bits[i + 1][(bit + 18) & 31] -
                     Fp(2) * w_bits[i + 1][(bit + 7) & 31] * w_bits[i + 1][(bit + 18) & 31]) *
                    ((bit + 3 < 32) ? w_bits[i + 1][bit + 3] : Fp::zero());
        }

        Fp sig_w_limbs[SHA256_WORD_U16S];
        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            sig_w_limbs[j] = Fp::zero();
            for (int bit = 0; bit < 16; bit++) {
                sig_w_limbs[j] = sig_w_limbs[j] + sig_w_bits[j * 16 + bit] * Fp(1 << bit);
            }
        }

        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            SHA256INNER_WRITE_ROUND(
                next_row, schedule_helper.intermed_4[i][j], w_limbs[i][j] + sig_w_limbs[j]
            );
        }
    }
}

__device__ void generate_intermed_12(RowSlice local_row, RowSlice next_row) {
    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        Fp sig_w_2_limbs[SHA256_WORD_U16S];
        Fp w_i_plus_2_bits[SHA256_WORD_BITS];

        if (i + 2 < SHA256_ROUNDS_PER_ROW) {
            for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
                w_i_plus_2_bits[bit] =
                    local_row[COL_INDEX(Sha256RoundCols, message_schedule.w[i + 2][bit])];
            }
        } else {
            for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
                w_i_plus_2_bits[bit] = next_row[COL_INDEX(
                    Sha256RoundCols, message_schedule.w[i + 2 - SHA256_ROUNDS_PER_ROW][bit]
                )];
            }
        }

        Fp sig_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            sig_bits[bit] =
                (w_i_plus_2_bits[(bit + 17) & 31] + w_i_plus_2_bits[(bit + 19) & 31] -
                 Fp(2) * w_i_plus_2_bits[(bit + 17) & 31] * w_i_plus_2_bits[(bit + 19) & 31]) +
                ((bit + 10 < 32) ? w_i_plus_2_bits[bit + 10] : Fp::zero()) -
                Fp(2) *
                    (w_i_plus_2_bits[(bit + 17) & 31] + w_i_plus_2_bits[(bit + 19) & 31] -
                     Fp(2) * w_i_plus_2_bits[(bit + 17) & 31] * w_i_plus_2_bits[(bit + 19) & 31]) *
                    ((bit + 10 < 32) ? w_i_plus_2_bits[bit + 10] : Fp::zero());
        }

        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            sig_w_2_limbs[j] = Fp::zero();
            for (int bit = 0; bit < 16; bit++) {
                sig_w_2_limbs[j] += sig_bits[j * 16 + bit] * Fp(1 << bit);
            }
        }

        Fp w_7_limbs[SHA256_WORD_U16S];
        if (i < 3) {
            w_7_limbs[0] = local_row[COL_INDEX(Sha256RoundCols, schedule_helper.w_3[i][0])];
            w_7_limbs[1] = local_row[COL_INDEX(Sha256RoundCols, schedule_helper.w_3[i][1])];
        } else {
            Fp w_i_minus_3_bits[SHA256_WORD_BITS];
            for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
                w_i_minus_3_bits[bit] =
                    local_row[COL_INDEX(Sha256RoundCols, message_schedule.w[i - 3][bit])];
            }
            for (int j = 0; j < SHA256_WORD_U16S; j++) {
                w_7_limbs[j] = Fp::zero();
                for (int bit = 0; bit < 16; bit++) {
                    w_7_limbs[j] += w_i_minus_3_bits[j * 16 + bit] * Fp(1 << bit);
                }
            }
        }

        Fp w_cur_limbs[SHA256_WORD_U16S];
        Fp w_cur_bits[SHA256_WORD_BITS];
        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            w_cur_bits[bit] = next_row[COL_INDEX(Sha256RoundCols, message_schedule.w[i][bit])];
        }
        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            w_cur_limbs[j] = Fp::zero();
            for (int bit = 0; bit < 16; bit++) {
                w_cur_limbs[j] += w_cur_bits[j * 16 + bit] * Fp(1 << bit);
            }
        }

        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            Fp carry =
                next_row[COL_INDEX(Sha256RoundCols, message_schedule.carry_or_buffer[i][j * 2])] +
                Fp(2) * next_row[COL_INDEX(
                            Sha256RoundCols, message_schedule.carry_or_buffer[i][j * 2 + 1]
                        )];

            Fp prev_carry = Fp::zero();
            if (j > 0) {
                prev_carry =
                    next_row[COL_INDEX(
                        Sha256RoundCols, message_schedule.carry_or_buffer[i][j * 2 - 2]
                    )] +
                    Fp(2) * next_row[COL_INDEX(
                                Sha256RoundCols, message_schedule.carry_or_buffer[i][j * 2 - 1]
                            )];
            }

            Fp sum =
                sig_w_2_limbs[j] + w_7_limbs[j] - carry * Fp(1 << 16) - w_cur_limbs[j] + prev_carry;
            SHA256INNER_WRITE_ROUND(local_row, schedule_helper.intermed_12[i][j], -sum);
        }
    }
}

__device__ void get_block_hash(
    uint32_t hash[SHA256_HASH_WORDS],
    const uint8_t input[SHA256_BLOCK_U8S]
) {
    uint32_t work_vars[SHA256_HASH_WORDS];
    memcpy(work_vars, hash, SHA256_HASH_WORDS * sizeof(uint32_t));

    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = u32_from_bytes_be(input + i * 4);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = small_sig1(w[i - 2]) + w[i - 7] + small_sig0(w[i - 15]) + w[i - 16];
    }

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = work_vars[7] + big_sig1(work_vars[4]) +
                      ch(work_vars[4], work_vars[5], work_vars[6]) + SHA256_K[i] + w[i];
        uint32_t t2 = big_sig0(work_vars[0]) + maj(work_vars[0], work_vars[1], work_vars[2]);

        uint32_t a = work_vars[0];
        uint32_t b = work_vars[1];
        uint32_t c = work_vars[2];
        uint32_t d = work_vars[3];
        uint32_t e = work_vars[4];
        uint32_t f = work_vars[5];
        uint32_t g = work_vars[6];
        uint32_t h = work_vars[7];

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;

        work_vars[0] = a;
        work_vars[1] = b;
        work_vars[2] = c;
        work_vars[3] = d;
        work_vars[4] = e;
        work_vars[5] = f;
        work_vars[6] = g;
        work_vars[7] = h;
    }

    for (int i = 0; i < SHA256_HASH_WORDS; i++) {
        hash[i] += work_vars[i];
    }
}

__device__ void generate_block_trace(
    Fp *trace,
    size_t trace_height,
    const uint32_t input[SHA256_BLOCK_WORDS],
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    const uint32_t prev_hash[SHA256_HASH_WORDS],
    bool is_last_block,
    uint32_t global_block_idx,
    uint32_t local_block_idx
) {
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);
    Encoder row_idx_encoder(18, 2, false);

    uint32_t message_schedule[64];
    uint32_t work_vars[SHA256_HASH_WORDS];

    memcpy(message_schedule, input, SHA256_BLOCK_WORDS * sizeof(uint32_t));
    memcpy(work_vars, prev_hash, SHA256_HASH_WORDS * sizeof(uint32_t));

    for (int i = 0; i < SHA256_ROWS_PER_BLOCK; i++) {
        RowSlice row_slice(trace + i, trace_height);

        if (i < 16) {
            SHA256INNER_WRITE_ROUND(row_slice, flags.is_round_row, Fp::one());
            SHA256INNER_WRITE_ROUND(
                row_slice, flags.is_first_4_rows, (i < 4) ? Fp::one() : Fp::zero()
            );
            SHA256INNER_WRITE_ROUND(row_slice, flags.is_digest_row, Fp::zero());
            SHA256INNER_WRITE_ROUND(
                row_slice, flags.is_last_block, is_last_block ? Fp::one() : Fp::zero()
            );

            RowSlice row_idx_flags =
                row_slice.slice_from(COL_INDEX(Sha256RoundCols, flags.row_idx));
            row_idx_encoder.write_flag_pt(row_idx_flags, i);

            SHA256INNER_WRITE_ROUND(row_slice, flags.global_block_idx, global_block_idx);
            SHA256INNER_WRITE_ROUND(row_slice, flags.local_block_idx, local_block_idx);

            if (i < 4) {
                for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                    COL_WRITE_BITS(
                        row_slice,
                        Sha256RoundCols,
                        message_schedule.w[j],
                        input[i * SHA256_ROUNDS_PER_ROW + j]
                    );
                }
            } else {
                for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                    int idx = i * SHA256_ROUNDS_PER_ROW + j;

                    uint32_t w = small_sig1(message_schedule[idx - 2]) + message_schedule[idx - 7] +
                                 small_sig0(message_schedule[idx - 15]) +
                                 message_schedule[idx - 16];

                    COL_WRITE_BITS(row_slice, Sha256RoundCols, message_schedule.w[j], w);

                    for (int k = 0; k < SHA256_WORD_U16S; k++) {
                        uint32_t sum = u32_to_u16_limb(small_sig1(message_schedule[idx - 2]), k) +
                                       u32_to_u16_limb(message_schedule[idx - 7], k) +
                                       u32_to_u16_limb(small_sig0(message_schedule[idx - 15]), k) +
                                       u32_to_u16_limb(message_schedule[idx - 16], k);

                        if (k > 0) {
                            sum += row_slice[COL_INDEX(
                                                 Sha256RoundCols,
                                                 message_schedule.carry_or_buffer[j][k * 2 - 2]
                                             )]
                                       .asUInt32() +
                                   2 * row_slice[COL_INDEX(
                                                     Sha256RoundCols,
                                                     message_schedule.carry_or_buffer[j][k * 2 - 1]
                                                 )]
                                           .asUInt32();
                        }

                        uint32_t carry = (sum - u32_to_u16_limb(w, k)) >> 16;
                        SHA256INNER_WRITE_ROUND(
                            row_slice, message_schedule.carry_or_buffer[j][k * 2], Fp(carry & 1)
                        );
                        SHA256INNER_WRITE_ROUND(
                            row_slice,
                            message_schedule.carry_or_buffer[j][k * 2 + 1],
                            Fp(carry >> 1)
                        );
                    }
                    message_schedule[idx] = w;
                }
            }

            for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                int idx = i * SHA256_ROUNDS_PER_ROW + j;

                uint32_t t1 = work_vars[7] + big_sig1(work_vars[4]) +
                              ch(work_vars[4], work_vars[5], work_vars[6]) + SHA256_K[idx] +
                              message_schedule[idx];
                uint32_t t2 =
                    big_sig0(work_vars[0]) + maj(work_vars[0], work_vars[1], work_vars[2]);
                uint32_t e = work_vars[3] + t1;
                uint32_t a = t1 + t2;

                COL_WRITE_BITS(row_slice, Sha256RoundCols, work_vars.a[j], a);
                COL_WRITE_BITS(row_slice, Sha256RoundCols, work_vars.e[j], e);

                uint32_t carry_a_values[SHA256_WORD_U16S] = {0};
                uint32_t carry_e_values[SHA256_WORD_U16S] = {0};

                for (int k = 0; k < SHA256_WORD_U16S; k++) {
                    uint32_t t1_limb =
                        u32_to_u16_limb(work_vars[7], k) +
                        u32_to_u16_limb(big_sig1(work_vars[4]), k) +
                        u32_to_u16_limb(ch(work_vars[4], work_vars[5], work_vars[6]), k) +
                        u32_to_u16_limb(SHA256_K[idx], k) +
                        u32_to_u16_limb(message_schedule[idx], k);

                    uint32_t t2_limb =
                        u32_to_u16_limb(big_sig0(work_vars[0]), k) +
                        u32_to_u16_limb(maj(work_vars[0], work_vars[1], work_vars[2]), k);

                    uint32_t e_limb = t1_limb + u32_to_u16_limb(work_vars[3], k);
                    uint32_t a_limb = t1_limb + t2_limb;

                    if (k > 0) {
                        a_limb += carry_a_values[k - 1];
                        e_limb += carry_e_values[k - 1];
                    }

                    carry_a_values[k] = (a_limb - u32_to_u16_limb(a, k)) >> 16;
                    carry_e_values[k] = (e_limb - u32_to_u16_limb(e, k)) >> 16;

                    SHA256INNER_WRITE_ROUND(row_slice, work_vars.carry_a[j][k], carry_a_values[k]);
                    SHA256INNER_WRITE_ROUND(row_slice, work_vars.carry_e[j][k], carry_e_values[k]);

                    bitwise_lookup.add_range(carry_a_values[k], carry_e_values[k]);
                }

                work_vars[7] = work_vars[6];
                work_vars[6] = work_vars[5];
                work_vars[5] = work_vars[4];
                work_vars[4] = e;
                work_vars[3] = work_vars[2];
                work_vars[2] = work_vars[1];
                work_vars[1] = work_vars[0];
                work_vars[0] = a;
            }

            if (i == 0) {
                for (int j = 0; j < SHA256_ROUNDS_PER_ROW - 1; j++) {
                    for (int k = 0; k < SHA256_WORD_U16S; k++) {
                        SHA256INNER_WRITE_ROUND(row_slice, schedule_helper.w_3[j][k], Fp::zero());
                    }
                }

                for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                    for (int k = 0; k < SHA256_WORD_U16S; k++) {
                        SHA256INNER_WRITE_ROUND(
                            row_slice, schedule_helper.intermed_4[j][k], Fp::zero()
                        );
                        SHA256INNER_WRITE_ROUND(
                            row_slice, schedule_helper.intermed_8[j][k], Fp::zero()
                        );
                        SHA256INNER_WRITE_ROUND(
                            row_slice, schedule_helper.intermed_12[j][k], Fp::zero()
                        );
                    }
                }
            } else if (i > 0) {
                for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                    uint32_t idx = i * SHA256_ROUNDS_PER_ROW + j;

                    uint32_t w_4 = message_schedule[idx - 4];
                    uint32_t sig_0_w_3 = small_sig0(message_schedule[idx - 3]);

                    SHA256INNER_WRITE_ROUND(
                        row_slice,
                        schedule_helper.intermed_4[j][0],
                        Fp(u32_to_u16_limb(w_4, 0) + u32_to_u16_limb(sig_0_w_3, 0))
                    );
                    SHA256INNER_WRITE_ROUND(
                        row_slice,
                        schedule_helper.intermed_4[j][1],
                        Fp(u32_to_u16_limb(w_4, 1) + u32_to_u16_limb(sig_0_w_3, 1))
                    );

                    if (j < SHA256_ROUNDS_PER_ROW - 1) {
                        uint32_t w_3 = message_schedule[idx - 3];
                        SHA256INNER_WRITE_ROUND(
                            row_slice, schedule_helper.w_3[j][0], u32_to_u16_limb(w_3, 0)
                        );
                        SHA256INNER_WRITE_ROUND(
                            row_slice, schedule_helper.w_3[j][1], u32_to_u16_limb(w_3, 1)
                        );
                    }
                }
            }
        } else {
            for (int j = 0; j < SHA256_ROUNDS_PER_ROW - 1; j++) {
                uint32_t w_3 = message_schedule[i * SHA256_ROUNDS_PER_ROW + j - 3];
                SHA256INNER_WRITE_DIGEST(
                    row_slice, schedule_helper.w_3[j][0], u32_to_u16_limb(w_3, 0)
                );
                SHA256INNER_WRITE_DIGEST(
                    row_slice, schedule_helper.w_3[j][1], u32_to_u16_limb(w_3, 1)
                );
            }

            SHA256INNER_WRITE_DIGEST(row_slice, flags.is_round_row, Fp::zero());
            SHA256INNER_WRITE_DIGEST(row_slice, flags.is_first_4_rows, Fp::zero());
            SHA256INNER_WRITE_DIGEST(row_slice, flags.is_digest_row, Fp::one());
            SHA256INNER_WRITE_DIGEST(
                row_slice, flags.is_last_block, is_last_block ? Fp::one() : Fp::zero()
            );

            RowSlice row_idx_flags =
                row_slice.slice_from(COL_INDEX(Sha256DigestCols, flags.row_idx));
            row_idx_encoder.write_flag_pt(row_idx_flags, 16);

            SHA256INNER_WRITE_DIGEST(row_slice, flags.global_block_idx, global_block_idx);
            SHA256INNER_WRITE_DIGEST(row_slice, flags.local_block_idx, local_block_idx);

            uint32_t final_hash[SHA256_HASH_WORDS];
            for (int j = 0; j < SHA256_HASH_WORDS; j++) {
                final_hash[j] = work_vars[j] + prev_hash[j];
            }

            for (int j = 0; j < SHA256_HASH_WORDS; j++) {
                uint8_t *hash_bytes = (uint8_t *)&final_hash[j];
                SHA256INNER_WRITE_ARRAY_DIGEST(row_slice, final_hash[j], hash_bytes);

#pragma unroll
                for (int chunk = 0; chunk < SHA256_WORD_U8S; chunk += 2) {
                    bitwise_lookup.add_range(
                        (uint32_t)hash_bytes[chunk], (uint32_t)hash_bytes[chunk + 1]
                    );
                }
            }

            for (int j = 0; j < SHA256_HASH_WORDS; j++) {
                SHA256INNER_WRITE_DIGEST(
                    row_slice, prev_hash[j][0], u32_to_u16_limb(prev_hash[j], 0)
                );
                SHA256INNER_WRITE_DIGEST(
                    row_slice, prev_hash[j][1], u32_to_u16_limb(prev_hash[j], 1)
                );
            }

            uint32_t hash[SHA256_HASH_WORDS];
            if (is_last_block) {
                for (int j = 0; j < SHA256_HASH_WORDS; j++) {
                    hash[j] = SHA256_H[j];
                }
            } else {
                for (int j = 0; j < SHA256_HASH_WORDS; j++) {
                    hash[j] = final_hash[j];
                }
            }

            for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                COL_WRITE_BITS(
                    row_slice, Sha256DigestCols, hash.a[j], hash[SHA256_ROUNDS_PER_ROW - j - 1]
                );
                COL_WRITE_BITS(
                    row_slice, Sha256DigestCols, hash.e[j], hash[SHA256_ROUNDS_PER_ROW - j + 3]
                );
            }
        }
    }
    for (int i = 0; i < SHA256_ROWS_PER_BLOCK - 1; i++) {
        RowSlice local_row(trace + i, trace_height);
        RowSlice next_row(trace + i + 1, trace_height);

        for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
            for (int k = 0; k < SHA256_WORD_U16S; k++) {
                Fp intermed_4_val =
                    local_row[COL_INDEX(Sha256RoundCols, schedule_helper.intermed_4[j][k])];
                SHA256INNER_WRITE_ROUND(next_row, schedule_helper.intermed_8[j][k], intermed_4_val);
            }
        }

        if (i >= 2 && i <= 13) {
            for (int j = 0; j < SHA256_ROUNDS_PER_ROW; j++) {
                for (int k = 0; k < SHA256_WORD_U16S; k++) {
                    Fp intermed_8_val =
                        local_row[COL_INDEX(Sha256RoundCols, schedule_helper.intermed_8[j][k])];
                    SHA256INNER_WRITE_ROUND(
                        next_row, schedule_helper.intermed_12[j][k], intermed_8_val
                    );
                }
            }
        }

        if (i == SHA256_ROWS_PER_BLOCK - 2) {

            generate_carry_ae(local_row, next_row);
            generate_intermed_4(local_row, next_row);
        }

        if (i <= 2) {
            generate_intermed_12(local_row, next_row);
        }
    }
}

__device__ void generate_default_row(RowSlice row_slice) {
    Encoder row_idx_encoder(18, 2, false);
    RowSlice row_idx_flags = row_slice.slice_from(COL_INDEX(Sha256RoundCols, flags.row_idx));
    row_idx_encoder.write_flag_pt(row_idx_flags, 17);

    for (int i = 0; i < SHA256_ROUNDS_PER_ROW; i++) {
        uint32_t a_word = SHA256_H[3 - i];
        uint32_t e_word = SHA256_H[7 - i];

        for (int bit = 0; bit < SHA256_WORD_BITS; bit++) {
            SHA256INNER_WRITE_ROUND(row_slice, work_vars.a[i][bit], Fp((a_word >> bit) & 1));
            SHA256INNER_WRITE_ROUND(row_slice, work_vars.e[i][bit], Fp((e_word >> bit) & 1));
        }

#pragma unroll
        for (int j = 0; j < SHA256_WORD_U16S; j++) {
            SHA256INNER_WRITE_ROUND(
                row_slice, work_vars.carry_a[i][j], SHA256_INVALID_CARRY_A[i][j]
            );
            SHA256INNER_WRITE_ROUND(
                row_slice, work_vars.carry_e[i][j], SHA256_INVALID_CARRY_E[i][j]
            );
        }
    }
}

__device__ void generate_missing_cells(Fp *trace_chunk, size_t trace_height) {
    RowSlice row15(trace_chunk + 15, trace_height);
    RowSlice row16(trace_chunk + 16, trace_height);
    RowSlice row17(trace_chunk + 17, trace_height);

    generate_intermed_12(row15, row16);
    generate_intermed_12(row16, row17);
    generate_intermed_4(row16, row17);
}

__global__ void sha256_second_pass_dependencies(
    Fp *inner_trace_start,
    size_t trace_height,
    size_t total_sha256_blocks
) {
    uint32_t sha256_block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sha256_block_idx >= total_sha256_blocks) {
        return;
    }

    Fp *block_start = inner_trace_start + (sha256_block_idx * SHA256_ROWS_PER_BLOCK);
    generate_missing_cells(block_start, trace_height);
}
