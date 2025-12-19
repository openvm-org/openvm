#include "keccakf/keccakf.cuh"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"

using namespace keccakf;
using namespace riscv;

__device__ __constant__ uint8_t R[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14},
};

__device__ __constant__ uint64_t RC[NUM_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ void generate_trace_row_for_round(
    RowSlice row,
    uint32_t round,
    uint64_t current_state[5][5]
) {
    COL_FILL_ZERO(row, KeccakfVmCols, inner.step_flags);
    COL_WRITE_VALUE(row, KeccakfVmCols, inner.step_flags[round], 1);

    // Populate C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4]).
    uint64_t state_c[5];
#pragma unroll 5
    for (auto x = 0; x < 5; x++) {
        state_c[x] = current_state[0][x] ^ current_state[1][x] ^ current_state[2][x] ^
                     current_state[3][x] ^ current_state[4][x];
        COL_WRITE_BITS(row, KeccakfVmCols, inner.c[x], state_c[x]);
    }

    // Populate C'[x, z] = xor(C[x, z], C[x - 1, z], ROTL1(C[x + 1, z - 1])).
    uint64_t state_c_prime[5];
#pragma unroll 5
    for (auto x = 0; x < 5; x++) {
        state_c_prime[x] = state_c[x] ^ state_c[(x + 4) % 5] ^ ROTL64(state_c[(x + 1) % 5], 1);
        COL_WRITE_BITS(row, KeccakfVmCols, inner.c_prime[x], state_c_prime[x]);
    }

    // Populate A'. To avoid shifting indices, we rewrite
    //     A'[x, y, z] = xor(A[x, y, z], C[x - 1, z], C[x + 1, z - 1])
    // as
    //     A'[x, y, z] = xor(A[x, y, z], C[x, z], C'[x, z]).
    for (auto x = 0; x < 5; x++) {
#pragma unroll 5
        for (auto y = 0; y < 5; y++) {
            current_state[y][x] ^= state_c[x] ^ state_c_prime[x];
            COL_WRITE_BITS(row, KeccakfVmCols, inner.a_prime[y][x], current_state[y][x]);
        }
    }

    // Rotate the current state to get the B array.
    uint64_t state_b[5][5];
    for (auto i = 0; i < 5; i++) {
#pragma unroll 5
        for (auto j = 0; j < 5; j++) {
            auto new_i = (i + 3 * j) % 5;
            auto new_j = i;
            state_b[j][i] = ROTL64(current_state[new_j][new_i], R[new_i][new_j]);
        }
    }

    // Populate A'' as A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    for (auto i = 0; i < 5; i++) {
#pragma unroll 5
        for (auto j = 0; j < 5; j++) {
            current_state[i][j] =
                state_b[i][j] ^ ((~state_b[i][(j + 1) % 5]) & state_b[i][(j + 2) % 5]);
        }
    }
    uint16_t *state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
    COL_WRITE_ARRAY(row, KeccakfVmCols, inner.a_prime_prime, state_limbs);

    COL_WRITE_BITS(row, KeccakfVmCols, inner.a_prime_prime_0_0_bits, current_state[0][0]);

    // A''[0, 0] is additionally xor'd with RC.
    current_state[0][0] ^= RC[round];

    state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
    COL_WRITE_ARRAY(row, KeccakfVmCols, inner.a_prime_prime_prime_0_0_limbs, state_limbs);
}

// Kernel that processes blocks (24 rows per record), one block per thread
__global__ void keccakf_p3_tracegen(
    Fp *d_trace,
    size_t height,
    uint32_t num_records,
    uint32_t blocks_to_fill,  // includes dummy rows
    DeviceBufferConstView<KeccakfVmRecord> d_records
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= blocks_to_fill) {
        return;
    }

    RowSlice row(d_trace + block_idx * NUM_ROUNDS, height);

    __align__(16) uint64_t current_state[5][5] = {0};
    __align__(16) uint64_t initial_state[5][5] = {0};

    if (block_idx < num_records) {
        auto const &rec = d_records[block_idx];

        // Convert preimage bytes to u64 state and transpose for p3 compatibility
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                uint64_t val = 0;
                int src_idx = y + 5 * x;  // Source is in y-major order
                for (int j = 0; j < 8; j++) {
                    val |= static_cast<uint64_t>(rec.preimage_buffer_bytes[src_idx * 8 + j]) << (j * 8);
                }
                current_state[x][y] = val;  // Store transposed (x-major for p3)
                initial_state[x][y] = val;
            }
        }
    }

    uint16_t *initial_state_limbs = reinterpret_cast<uint16_t *>(initial_state);

    // First round
    COL_WRITE_ARRAY(row, KeccakfVmCols, inner.a, initial_state_limbs);
    COL_WRITE_ARRAY(row, KeccakfVmCols, inner.preimage, initial_state_limbs);
    generate_trace_row_for_round(row, 0, current_state);

    // Handle the last round (might be beyond height if this is the last partial block)
    auto last_rounds = height - (blocks_to_fill - 1) * NUM_ROUNDS;
    RowSlice prev_round_row = row;

#pragma unroll 8
    for (uint32_t round = 1; round < NUM_ROUNDS; round++) {
        if ((block_idx == (blocks_to_fill - 1)) && (round >= last_rounds)) {
            break;
        }

        RowSlice round_row(prev_round_row.ptr + 1, height);

        // Copy preimage to this row
        COL_WRITE_ARRAY(round_row, KeccakfVmCols, inner.preimage, initial_state_limbs);

        // Copy previous row's output to this row's input (a)
        for (int y = 0; y < 5; y++) {
#pragma unroll 5
            for (int x = 0; x < 5; x++) {
#pragma unroll 4
                for (int limb = 0; limb < U64_LIMBS; limb++) {
                    // For [0,0], use a_prime_prime_prime_0_0_limbs
                    // For others, use a_prime_prime
                    Fp value = ((x == 0) && (y == 0))
                        ? prev_round_row[COL_INDEX(KeccakfVmCols, inner.a_prime_prime_prime_0_0_limbs[limb])]
                        : prev_round_row[COL_INDEX(KeccakfVmCols, inner.a_prime_prime[y][x][limb])];
                    COL_WRITE_VALUE(round_row, KeccakfVmCols, inner.a[y][x][limb], value);
                }
            }
        }

        generate_trace_row_for_round(round_row, round, current_state);
        prev_round_row = round_row;
    }
}

#define KECCAKF_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakfVmCols, FIELD, VALUE)
#define KECCAKF_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakfVmCols, FIELD, VALUES)
#define KECCAKF_FILL_ZERO(FIELD) COL_FILL_ZERO(row, KeccakfVmCols, FIELD)
#define KECCAKF_SLICE(FIELD) row.slice_from(COL_INDEX(KeccakfVmCols, FIELD))

// Kernel that fills VM-specific columns (instruction, memory aux, etc.)
// Processes one record (24 rows) per thread to avoid redundant keccakf computation
__global__ void keccakf_vm_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<KeccakfVmRecord> d_records,
    size_t rows_used,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    auto record_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto num_records = d_records.len();
    auto total_blocks = (height + NUM_ROUNDS - 1) / NUM_ROUNDS;

    if (record_idx >= total_blocks) {
        return;
    }

    MemoryAuxColsFactory mem_helper(
        VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
        timestamp_max_bits
    );
    auto bitwise_lookup = BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits);

    // Compute postimage once per record (instead of once per row)
    uint8_t postimage_bytes[KECCAK_STATE_BYTES] = {0};

    if (record_idx < num_records) {
        auto const &rec = d_records[record_idx];

        // Compute postimage for this record
        uint64_t postimage_u64[25];
#pragma unroll 5
        for (auto i = 0; i < 25; i++) {
            postimage_u64[i] = 0;
            for (auto j = 0; j < 8; j++) {
                postimage_u64[i] |= static_cast<uint64_t>(rec.preimage_buffer_bytes[i * 8 + j]) << (j * 8);
            }
        }

        // Keccakf permutation (computed ONCE per record)
        for (auto round = 0; round < NUM_ROUNDS; round++) {
            // Theta
            uint64_t c[5];
#pragma unroll 5
            for (auto x = 0; x < 5; x++) {
                c[x] = postimage_u64[x] ^ postimage_u64[x + 5] ^ postimage_u64[x + 10] ^
                       postimage_u64[x + 15] ^ postimage_u64[x + 20];
            }
            uint64_t d[5];
#pragma unroll 5
            for (auto x = 0; x < 5; x++) {
                d[x] = c[(x + 4) % 5] ^ ROTL64(c[(x + 1) % 5], 1);
            }
#pragma unroll 5
            for (auto i = 0; i < 25; i++) {
                postimage_u64[i] ^= d[i % 5];
            }

            // Rho and Pi
            uint64_t temp[25];
#pragma unroll 5
            for (auto y = 0; y < 5; y++) {
#pragma unroll 5
                for (auto x = 0; x < 5; x++) {
                    auto idx = x + 5 * y;
                    auto new_x = y;
                    auto new_y = (2 * x + 3 * y) % 5;
                    auto new_idx = new_x + 5 * new_y;
                    temp[new_idx] = ROTL64(postimage_u64[idx], R[x][y]);
                }
            }

            // Chi
#pragma unroll 5
            for (auto y = 0; y < 5; y++) {
#pragma unroll 5
                for (auto x = 0; x < 5; x++) {
                    auto idx = x + 5 * y;
                    postimage_u64[idx] = temp[idx] ^ ((~temp[(x + 1) % 5 + 5 * y]) & temp[(x + 2) % 5 + 5 * y]);
                }
            }

            // Iota
            postimage_u64[0] ^= RC[round];
        }

        // Convert postimage to bytes
#pragma unroll 5
        for (auto i = 0; i < 25; i++) {
            for (auto j = 0; j < 8; j++) {
                postimage_bytes[i * 8 + j] = static_cast<uint8_t>(postimage_u64[i] >> (j * 8));
            }
        }
    }

    // Fill all 24 rows for this record
    auto base_row_idx = record_idx * NUM_ROUNDS;
    auto last_valid_round = (record_idx == total_blocks - 1)
        ? (height - base_row_idx)
        : NUM_ROUNDS;

    for (auto round_idx = 0u; round_idx < last_valid_round; round_idx++) {
        RowSlice row(d_trace + base_row_idx + round_idx, height);

        if (record_idx < num_records) {
            auto const &rec = d_records[record_idx];
            auto is_first_round = (round_idx == 0);
            auto is_final_round = (round_idx == NUM_ROUNDS - 1);

            // Fill the state hi columns
#pragma unroll 8
            for (auto i = 0; i < KECCAK_STATE_U16; i++) {
                KECCAKF_WRITE(preimage_state_hi[i],
                    static_cast<uint8_t>(rec.preimage_buffer_bytes[2 * i + 1]));
            }
#pragma unroll 8
            for (auto i = 0; i < KECCAK_STATE_U16; i++) {
                KECCAKF_WRITE(postimage_state_hi[i], postimage_bytes[2 * i + 1]);
            }

            // Fill the instruction columns
            KECCAKF_WRITE(instruction.pc, rec.pc);
            KECCAKF_WRITE(instruction.is_enabled, 1);
            KECCAKF_WRITE(instruction.rd_ptr, rec.rd_ptr);
            KECCAKF_WRITE(instruction.buffer_ptr, rec.buffer);
            KECCAKF_WRITE_ARRAY(instruction.buffer_ptr_limbs, reinterpret_cast<const uint8_t *>(&rec.buffer));

            // Fill timestamp - matches Rust's timestamp tracking across rows
            KECCAKF_WRITE(timestamp, round_idx > 0 ? rec.timestamp + 1 + NUM_BUFFER_WORDS : rec.timestamp);

            KECCAKF_WRITE(is_enabled_is_first_round, is_first_round);
            KECCAKF_WRITE(is_enabled_is_final_round, is_final_round);
            KECCAKF_WRITE(inner._export, Fp::zero());

            // Fill the register reads
            if (is_first_round) {
                auto timestamp = rec.timestamp;
                mem_helper.fill(
                    KECCAKF_SLICE(mem_oc.register_aux_cols[0].base),
                    rec.register_aux_cols[0].prev_timestamp,
                    timestamp
                );
                timestamp++;

                // Buffer reads (50 words of 4 bytes each)
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    mem_helper.fill(
                        KECCAKF_SLICE(mem_oc.buffer_bytes_read_aux_cols[t].base),
                        rec.buffer_read_aux_cols[t].prev_timestamp,
                        timestamp
                    );
                    timestamp++;
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
            if (is_final_round) {
                auto write_timestamp = rec.timestamp + 1 + NUM_BUFFER_WORDS;
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    mem_helper.fill(
                        KECCAKF_SLICE(mem_oc.buffer_bytes_write_aux_cols[t].base),
                        rec.buffer_write_aux_cols[t].prev_timestamp,
                        write_timestamp
                    );
                    KECCAKF_WRITE_ARRAY(mem_oc.buffer_bytes_write_aux_cols[t].prev_data,
                        rec.buffer_write_aux_cols[t].prev_data);
                    write_timestamp++;
                }
            } else {
                for (auto t = 0u; t < NUM_BUFFER_WORDS; t++) {
                    KECCAKF_FILL_ZERO(mem_oc.buffer_bytes_write_aux_cols[t]);
                }
            }
        } else {
            // inner rows are filled in p3_tracegen kernel. Ensure export flag is 0
            KECCAKF_WRITE(inner._export, Fp::zero());
            row.fill_zero(
                COL_INDEX(KeccakfVmCols, preimage_state_hi),
                sizeof(KeccakfVmCols<uint8_t>) - COL_INDEX(KeccakfVmCols, preimage_state_hi)
            );
        }
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
    size_t rows_used,
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

    // First pass: generate p3 permutation columns (one thread per block of 24 rows)
    {
        auto [grid, block] = kernel_launch_params(blocks_to_fill, 256);
        keccakf_p3_tracegen<<<grid, block>>>(
            d_trace,
            height,
            num_records,
            blocks_to_fill,
            d_records
        );
        int err = CHECK_KERNEL();
        if (err != 0) return err;
    }

    // Second pass: fill VM-specific columns (one thread per record/block of 24 rows)
    {
        auto [grid, block] = kernel_launch_params(blocks_to_fill, 256);
        keccakf_vm_tracegen<<<grid, block>>>(
            d_trace,
            height,
            d_records,
            rows_used,
            d_range_checker_ptr,
            range_checker_num_bins,
            d_bitwise_lookup_ptr,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits
        );
        int err = CHECK_KERNEL();
        if (err != 0) return err;
    }

    return 0;
}
