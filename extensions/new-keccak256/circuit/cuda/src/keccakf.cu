#include "keccakf/keccakf.cuh"
#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"

using namespace keccakf;

// Keccak round constants
__device__ __constant__ uint64_t KECCAK_RC[NUM_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step
__device__ __constant__ uint8_t KECCAK_RHO[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14},
};

// Generate p3 trace row for a single round, following the exact p3_keccak_air layout
__device__ void generate_p3_trace_row_for_round(
    RowSlice row,
    uint32_t round,
    uint64_t current_state[5][5]
) {
    // Write step flags
    COL_FILL_ZERO(row, KeccakfVmCols, inner.step_flags);
    COL_WRITE_VALUE(row, KeccakfVmCols, inner.step_flags[round], 1);

    // Compute C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4])
    uint64_t state_c[5];
#pragma unroll 5
    for (int x = 0; x < 5; x++) {
        state_c[x] = current_state[0][x] ^ current_state[1][x] ^ current_state[2][x] ^
                     current_state[3][x] ^ current_state[4][x];
        COL_WRITE_BITS(row, KeccakfVmCols, inner.c[x], state_c[x]);
    }

    // Compute C'[x] = C[x] ^ C[x-1] ^ ROTL(C[x+1], 1)
    uint64_t state_c_prime[5];
#pragma unroll 5
    for (int x = 0; x < 5; x++) {
        state_c_prime[x] = state_c[x] ^ state_c[(x + 4) % 5] ^ ROTL64(state_c[(x + 1) % 5], 1);
        COL_WRITE_BITS(row, KeccakfVmCols, inner.c_prime[x], state_c_prime[x]);
    }

    // Compute A' and update current_state
    for (int x = 0; x < 5; x++) {
#pragma unroll 5
        for (int y = 0; y < 5; y++) {
            current_state[y][x] ^= state_c[x] ^ state_c_prime[x];
            COL_WRITE_BITS(row, KeccakfVmCols, inner.a_prime[y][x], current_state[y][x]);
        }
    }

    // Compute B (rho and pi)
    uint64_t state_b[5][5];
    for (int i = 0; i < 5; i++) {
#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            int new_i = (i + 3 * j) % 5;
            int new_j = i;
            state_b[j][i] = ROTL64(current_state[new_j][new_i], KECCAK_RHO[new_i][new_j]);
        }
    }

    // Compute A'' (chi)
    for (int i = 0; i < 5; i++) {
#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            current_state[i][j] = state_b[i][j] ^ ((~state_b[i][(j + 1) % 5]) & state_b[i][(j + 2) % 5]);
        }
    }
    uint16_t *state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
    COL_WRITE_ARRAY(row, KeccakfVmCols, inner.a_prime_prime, state_limbs);

    COL_WRITE_BITS(row, KeccakfVmCols, inner.a_prime_prime_0_0_bits, current_state[0][0]);

    // Compute A''' (iota for element [0,0] only)
    current_state[0][0] ^= KECCAK_RC[round];

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
    generate_p3_trace_row_for_round(row, 0, current_state);

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

        generate_p3_trace_row_for_round(round_row, round, current_state);
        prev_round_row = round_row;
    }
}

// Kernel that fills VM-specific columns (instruction, memory aux, etc.)
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
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height) {
        return;
    }

    RowSlice row(d_trace + row_idx, height);

    if (row_idx < rows_used) {
        uint32_t record_idx = row_idx / NUM_ROUNDS;
        uint32_t round_idx = row_idx % NUM_ROUNDS;

        if (record_idx >= d_records.len()) {
            return;
        }

        auto const &rec = d_records[record_idx];

        MemoryAuxColsFactory mem_helper(
            VariableRangeChecker(d_range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr, bitwise_num_bits);

        // Fill preimage_state_hi (high bytes of u16 limbs)
        for (int i = 0; i < KECCAK_STATE_U16; i++) {
            COL_WRITE_VALUE(row, KeccakfVmCols, preimage_state_hi[i],
                static_cast<uint8_t>(rec.preimage_buffer_bytes[2 * i + 1]));
        }

        // Compute postimage for filling postimage_state_hi
        uint64_t postimage_u64[25];
        for (int i = 0; i < 25; i++) {
            postimage_u64[i] = 0;
            for (int j = 0; j < 8; j++) {
                postimage_u64[i] |= static_cast<uint64_t>(rec.preimage_buffer_bytes[i * 8 + j]) << (j * 8);
            }
        }

        // Keccakf permutation
        for (int round = 0; round < NUM_ROUNDS; round++) {
            // Theta
            uint64_t c[5];
            for (int x = 0; x < 5; x++) {
                c[x] = postimage_u64[x] ^ postimage_u64[x + 5] ^ postimage_u64[x + 10] ^
                       postimage_u64[x + 15] ^ postimage_u64[x + 20];
            }
            uint64_t d[5];
            for (int x = 0; x < 5; x++) {
                d[x] = c[(x + 4) % 5] ^ ROTL64(c[(x + 1) % 5], 1);
            }
            for (int i = 0; i < 25; i++) {
                postimage_u64[i] ^= d[i % 5];
            }

            // Rho and Pi
            uint64_t temp[25];
            for (int y = 0; y < 5; y++) {
                for (int x = 0; x < 5; x++) {
                    int idx = x + 5 * y;
                    int new_x = y;
                    int new_y = (2 * x + 3 * y) % 5;
                    int new_idx = new_x + 5 * new_y;
                    temp[new_idx] = ROTL64(postimage_u64[idx], KECCAK_RHO[x][y]);
                }
            }

            // Chi
            for (int y = 0; y < 5; y++) {
                for (int x = 0; x < 5; x++) {
                    int idx = x + 5 * y;
                    postimage_u64[idx] = temp[idx] ^ ((~temp[(x + 1) % 5 + 5 * y]) & temp[(x + 2) % 5 + 5 * y]);
                }
            }

            // Iota
            postimage_u64[0] ^= KECCAK_RC[round];
        }

        // Fill postimage_state_hi
        uint8_t postimage_bytes[KECCAK_STATE_BYTES];
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 8; j++) {
                postimage_bytes[i * 8 + j] = static_cast<uint8_t>(postimage_u64[i] >> (j * 8));
            }
        }
        for (int i = 0; i < KECCAK_STATE_U16; i++) {
            COL_WRITE_VALUE(row, KeccakfVmCols, postimage_state_hi[i], postimage_bytes[2 * i + 1]);
        }

        // Fill instruction columns
        COL_WRITE_VALUE(row, KeccakfVmCols, instruction.pc, rec.pc);
        COL_WRITE_VALUE(row, KeccakfVmCols, instruction.is_enabled, 1);
        COL_WRITE_VALUE(row, KeccakfVmCols, instruction.buffer_ptr, rec.rd_ptr);
        COL_WRITE_VALUE(row, KeccakfVmCols, instruction.buffer, rec.buffer);

        uint8_t buffer_bytes[4];
        memcpy(buffer_bytes, &rec.buffer, 4);
        COL_WRITE_ARRAY(row, KeccakfVmCols, instruction.buffer_limbs, buffer_bytes);

        // Fill timestamp - matches Rust's timestamp tracking across rows
        // After first round, timestamp has been incremented by all first-round memory operations
        uint32_t col_timestamp = rec.timestamp;
        if (round_idx > 0) {
            col_timestamp = rec.timestamp + 1 + NUM_BUFFER_WORDS;  // 1 register + 50 buffer reads = 51
        }
        COL_WRITE_VALUE(row, KeccakfVmCols, timestamp, col_timestamp);

        // Fill step flag derivatives
        bool is_first_round = (round_idx == 0);
        bool is_final_round = (round_idx == NUM_ROUNDS - 1);
        COL_WRITE_VALUE(row, KeccakfVmCols, is_enabled_is_first_round, is_first_round ? 1 : 0);
        COL_WRITE_VALUE(row, KeccakfVmCols, is_enabled_is_final_round, is_final_round ? 1 : 0);

        // Export flag - should be 0 for keccakf
        COL_WRITE_VALUE(row, KeccakfVmCols, inner._export, 0);

        // Fill memory auxiliary columns
        uint32_t timestamp = rec.timestamp;

        if (is_first_round) {
            // Register read
            mem_helper.fill(
                row.slice_from(COL_INDEX(KeccakfVmCols, mem_oc.register_aux_cols[0].base)),
                rec.register_aux_cols[0].prev_timestamp,
                timestamp
            );
            timestamp++;

            // Buffer reads (50 words of 4 bytes each)
            for (uint32_t t = 0; t < NUM_BUFFER_WORDS; t++) {
                mem_helper.fill(
                    row.slice_from(COL_INDEX(KeccakfVmCols, mem_oc.buffer_bytes_read_aux_cols[t].base)),
                    rec.buffer_read_aux_cols[t].prev_timestamp,
                    timestamp
                );
                timestamp++;
            }

            // Range check for buffer pointer
            uint32_t msl_rshift = 24;  // RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)
            uint32_t msl_lshift = 32 - pointer_max_bits;
            bitwise_lookup.add_range(
                (rec.buffer >> msl_rshift) << msl_lshift,
                (rec.buffer >> msl_rshift) << msl_lshift
            );
        } else {
            // Zero fill register aux cols for non-first rounds
            mem_helper.fill_zero(row.slice_from(COL_INDEX(KeccakfVmCols, mem_oc.register_aux_cols[0].base)));
            for (uint32_t t = 0; t < NUM_BUFFER_WORDS; t++) {
                mem_helper.fill_zero(row.slice_from(COL_INDEX(KeccakfVmCols, mem_oc.buffer_bytes_read_aux_cols[t].base)));
            }
        }

        if (is_final_round) {
            // Buffer writes (50 words of 4 bytes each)
            uint32_t write_timestamp = rec.timestamp + 1 + NUM_BUFFER_WORDS;
            for (uint32_t t = 0; t < NUM_BUFFER_WORDS; t++) {
                mem_helper.fill(
                    row.slice_from(COL_INDEX(KeccakfVmCols, mem_oc.buffer_bytes_write_aux_cols[t].base)),
                    rec.buffer_write_aux_cols[t].prev_timestamp,
                    write_timestamp
                );
                COL_WRITE_ARRAY(row, KeccakfVmCols, mem_oc.buffer_bytes_write_aux_cols[t].prev_data,
                    rec.buffer_write_aux_cols[t].prev_data);
                write_timestamp++;
            }
        } else {
            // Zero fill write aux cols for non-final rounds
            for (uint32_t t = 0; t < NUM_BUFFER_WORDS; t++) {
                COL_FILL_ZERO(row, KeccakfVmCols, mem_oc.buffer_bytes_write_aux_cols[t]);
            }
        }
    } else {
        // Dummy rows - just need to fill VM columns with zeros
        // p3 columns are filled by keccakf_p3_tracegen
        row.fill_zero(
            COL_INDEX(KeccakfVmCols, preimage_state_hi),
            sizeof(KeccakfVmCols<uint8_t>) - COL_INDEX(KeccakfVmCols, preimage_state_hi)
        );
    }
}

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

    // Second pass: fill VM-specific columns (one thread per row)
    {
        auto [grid, block] = kernel_launch_params(height, 256);
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
