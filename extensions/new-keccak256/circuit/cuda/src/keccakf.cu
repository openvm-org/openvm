#include "keccakf/keccakf.cuh"
#include "keccakf/p3_generation.cuh"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"

using namespace keccakf;
using namespace riscv;
using namespace new_keccak;

#define KECCAKF_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakfVmCols, FIELD, VALUE)
#define KECCAKF_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakfVmCols, FIELD, VALUES)
#define KECCAKF_FILL_ZERO(FIELD) COL_FILL_ZERO(row, KeccakfVmCols, FIELD)
#define KECCAKF_SLICE(FIELD) row.slice_from(COL_INDEX(KeccakfVmCols, FIELD))

// Single kernel that processes one block (24 rows) per thread
__global__ void keccakf_tracegen(
    Fp *d_trace,
    size_t height,
    uint32_t num_records,
    uint32_t blocks_to_fill,
    DeviceBufferConstView<KeccakfVmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    auto block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= blocks_to_fill) {
        return;
    }

    __align__(16) uint64_t current_state[5][5] = {0};
    __align__(16) uint64_t initial_state[5][5] = {0};
    uint64_t postimage_u64[25] = {0};

    if (block_idx < num_records) {
        auto const &rec = d_records[block_idx];

        // Convert preimage bytes to u64 state and transpose for p3 compatibility
#pragma unroll 5
        for (auto x = 0; x < 5; x++) {
#pragma unroll 5
            for (auto y = 0; y < 5; y++) {
                uint64_t val = 0;
#pragma unroll 8
                for (auto j = 0; j < 8; j++) {
                    val |= static_cast<uint64_t>(rec.preimage_buffer_bytes[(y + 5 * x) * 8 + j]) << (j * 8);
                }
                current_state[x][y] = val;  // Store transposed (x-major for p3)
                initial_state[x][y] = val;
            }
        }

        // Compute postimage - load preimage into postimage_u64
#pragma unroll 5
        for (auto i = 0; i < 25; i++) {
            uint64_t val = 0;
#pragma unroll 8
            for (auto j = 0; j < 8; j++) {
                val |= static_cast<uint64_t>(rec.preimage_buffer_bytes[i * 8 + j]) << (j * 8);
            }
            postimage_u64[i] = val;
        }

        // Keccakf permutation
#pragma unroll 1
        for (auto round = 0; round < NUM_ROUNDS; round++) {
            // Theta
            uint64_t c[5];
#pragma unroll 5
            for (auto x = 0; x < 5; x++) {
                c[x] = postimage_u64[x] ^ postimage_u64[x + 5] ^ postimage_u64[x + 10] ^
                       postimage_u64[x + 15] ^ postimage_u64[x + 20];
            }
#pragma unroll 5
            for (auto x = 0; x < 5; x++) {
                auto d = c[(x + 4) % 5] ^ ROTL64(c[(x + 1) % 5], 1);
                postimage_u64[x] ^= d;
                postimage_u64[x + 5] ^= d;
                postimage_u64[x + 10] ^= d;
                postimage_u64[x + 15] ^= d;
                postimage_u64[x + 20] ^= d;
            }

            // Rho and Pi
            uint64_t temp[25];
#pragma unroll 5
            for (auto y = 0; y < 5; y++) {
#pragma unroll 5
                for (auto x = 0; x < 5; x++) {
                    temp[y + 5 * ((2 * x + 3 * y) % 5)] = ROTL64(postimage_u64[x + 5 * y], R[x][y]);
                }
            }

            // Chi
#pragma unroll 5
            for (auto y = 0; y < 5; y++) {
#pragma unroll 5
                for (auto x = 0; x < 5; x++) {
                    postimage_u64[x + 5 * y] = temp[x + 5 * y] ^ ((~temp[(x + 1) % 5 + 5 * y]) & temp[(x + 2) % 5 + 5 * y]);
                }
            }

            // Iota
            postimage_u64[0] ^= RC[round];
        }
    }

    auto last_rounds = height - (blocks_to_fill - 1) * NUM_ROUNDS;
    uint16_t *initial_state_limbs = reinterpret_cast<uint16_t *>(initial_state);
    RowSlice prev_round_row(d_trace + block_idx * NUM_ROUNDS, height);

    // Process all 24 rounds for this block
    for (uint32_t round_idx = 0; round_idx < NUM_ROUNDS; round_idx++) {
        if ((block_idx == (blocks_to_fill - 1)) && (round_idx >= last_rounds)) {
            break;
        }

        RowSlice row(d_trace + block_idx * NUM_ROUNDS + round_idx, height);

        // Fill p3 permutation columns
        if (round_idx == 0) {
            COL_WRITE_ARRAY(row, KeccakfVmCols, inner.a, initial_state_limbs);
            COL_WRITE_ARRAY(row, KeccakfVmCols, inner.preimage, initial_state_limbs);
        } else {
            COL_WRITE_ARRAY(row, KeccakfVmCols, inner.preimage, initial_state_limbs);
            // Copy previous row's output to this row's input (a)
            for (auto y = 0; y < 5; y++) {
#pragma unroll 5
                for (auto x = 0; x < 5; x++) {
#pragma unroll 4
                    for (auto limb = 0; limb < U64_LIMBS; limb++) {
                        COL_WRITE_VALUE(row, KeccakfVmCols, inner.a[y][x][limb],
                            ((x == 0) && (y == 0))
                                ? prev_round_row[COL_INDEX(KeccakfVmCols, inner.a_prime_prime_prime_0_0_limbs[limb])]
                                : prev_round_row[COL_INDEX(KeccakfVmCols, inner.a_prime_prime[y][x][limb])]);
                    }
                }
            }
        }
        generate_trace_row_for_round(row, round_idx, current_state);

        if (block_idx < num_records) {
            auto const &rec = d_records[block_idx];

            MemoryAuxColsFactory mem_helper(
                VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
                timestamp_max_bits
            );
            BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr, bitwise_num_bits);

            // Fill the state hi columns - extract hi byte directly from u64
            // preimage_state_hi[i] = high byte of i-th u16 = byte at position 2*i+1
#pragma unroll 8
            for (auto i = 0; i < KECCAK_STATE_U16; i++) {
                KECCAKF_WRITE(preimage_state_hi[i],
                    static_cast<uint8_t>(rec.preimage_buffer_bytes[2 * i + 1]));
            }
            // postimage_state_hi[i] = high byte of i-th u16 from postimage_u64
            // i-th u16 is in postimage_u64[i/4], at limb position (i%4), hi byte offset is 8
#pragma unroll 8
            for (auto i = 0; i < KECCAK_STATE_U16; i++) {
                KECCAKF_WRITE(postimage_state_hi[i],
                    static_cast<uint8_t>((postimage_u64[i / 4] >> ((i % 4) * 16 + 8)) & 0xFF));
            }

            // Fill the instruction columns
            KECCAKF_WRITE(instruction.pc, rec.pc);
            KECCAKF_WRITE(instruction.is_enabled, 1);
            KECCAKF_WRITE(instruction.rd_ptr, rec.rd_ptr);
            KECCAKF_WRITE(instruction.buffer_ptr, rec.buffer);
            KECCAKF_WRITE_ARRAY(instruction.buffer_ptr_limbs, reinterpret_cast<const uint8_t *>(&rec.buffer));

            // Fill timestamp
            KECCAKF_WRITE(timestamp, round_idx > 0 ? rec.timestamp + 1 + NUM_BUFFER_WORDS : rec.timestamp);

            KECCAKF_WRITE(is_enabled_is_first_round, round_idx == 0);
            KECCAKF_WRITE(is_enabled_is_final_round, round_idx == NUM_ROUNDS - 1);
            KECCAKF_WRITE(inner._export, Fp::zero());

            // Fill the register reads
            if (round_idx == 0) {
                auto ts = rec.timestamp;
                mem_helper.fill(
                    KECCAKF_SLICE(mem_oc.register_aux_cols[0].base),
                    rec.register_aux_cols[0].prev_timestamp,
                    ts
                );
                ts++;

                // Buffer reads (50 words of 4 bytes each)
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    mem_helper.fill(
                        KECCAKF_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base),
                        rec.buffer_read_aux_cols[t].prev_timestamp,
                        ts
                    );
                    ts++;
                }

                // Range check for buffer pointer
                constexpr uint32_t MSL_RSHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
                constexpr uint32_t RV32_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
                bitwise_lookup.add_range(
                    (rec.buffer >> MSL_RSHIFT) << (RV32_TOTAL_BITS - pointer_max_bits),
                    (rec.buffer >> MSL_RSHIFT) << (RV32_TOTAL_BITS - pointer_max_bits)
                );
            } else {
                mem_helper.fill_zero(KECCAKF_SLICE(mem_oc.register_aux_cols[0].base));
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    mem_helper.fill_zero(KECCAKF_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base));
                }
            }

            // Fill the buffer writes
            if (round_idx == NUM_ROUNDS - 1) {
                auto write_ts = rec.timestamp + 1 + NUM_BUFFER_WORDS;
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    mem_helper.fill(
                        KECCAKF_SLICE(mem_oc.buffer_bytes_write_aux_cols[t].base),
                        rec.buffer_write_aux_cols[t].prev_timestamp,
                        write_ts
                    );
                    KECCAKF_WRITE_ARRAY(mem_oc.buffer_bytes_write_aux_cols[t].prev_data,
                        rec.buffer_write_aux_cols[t].prev_data);
                    write_ts++;
                }
            } else {
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    KECCAKF_FILL_ZERO(mem_oc.buffer_bytes_write_aux_cols[t]);
                }
            }
        } else {
            // inner rows are filled above. Ensure export flag is 0
            KECCAKF_WRITE(inner._export, Fp::zero());
            row.fill_zero(
                COL_INDEX(KeccakfVmCols, preimage_state_hi),
                sizeof(KeccakfVmCols<uint8_t>) - COL_INDEX(KeccakfVmCols, preimage_state_hi)
            );
        }

        prev_round_row = row;
    }
}

#undef KECCAKF_WRITE
#undef KECCAKF_WRITE_ARRAY
#undef KECCAKF_FILL_ZERO
#undef KECCAKF_SLICE

extern "C" int _keccakf_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<KeccakfVmRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(KeccakfVmCols<uint8_t>));

    uint32_t num_records = d_records.len();
    uint32_t blocks_to_fill = div_ceil(height, uint32_t(NUM_ROUNDS));

    auto [grid, block] = kernel_launch_params(blocks_to_fill, 256);
    keccakf_tracegen<<<grid, block>>>(
        d_trace,
        height,
        num_records,
        blocks_to_fill,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        pointer_max_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
