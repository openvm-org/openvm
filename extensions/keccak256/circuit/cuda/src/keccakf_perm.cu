#include "fp.h"
#include "keccakf_op.cuh" // For KeccakfOpRecord
#include "keccakf_perm.cuh"
#include "launcher.cuh"
#include "p3_keccakf.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace keccakf_perm;
using namespace keccakf_op;
using p3_keccak_air::NUM_ROUNDS;
using p3_keccak_air::U64_LIMBS;

#define KECCAKF_PERM_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakfPermCols, FIELD, VALUE)
#define KECCAKF_PERM_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakfPermCols, FIELD, VALUES)

// Main kernel for KeccakfPermChip trace generation
// Each thread processes one permutation (24 rows)
__global__ void keccakf_perm_tracegen(
    Fp *__restrict__ d_trace, // no alias with d_records, enables store optimization
    size_t height,
    uint32_t num_records,
    uint32_t blocks_to_fill, // = ceil(height / 24)
    DeviceBufferConstView<KeccakfOpRecord> d_records
) {
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= blocks_to_fill) {
        return;
    }

    // Single persistent state (no separate initial_state to reduce register pressure).
    __align__(16) uint64_t current_state[5][5] = {0};
    // Compact preimage as u16 limbs, rewritten each round.
    uint16_t preimage_limbs[5 * 5 * U64_LIMBS] = {0};

    uint32_t timestamp = 0;

    if (block_idx < num_records) {
        auto const &rec = d_records[block_idx];
        timestamp = rec.timestamp;

        // Convert preimage bytes to u64 state with coordinate transposition
        // generate_trace_row_for_round expects current_state[y][x] = A[x][y] (keccak notation)
        // In keccak buffer: A[x][y] is stored at offset (x + 5*y)
        // So current_state[y][x] should get keccak_buffer[x + 5*y]
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                // keccak spec: A[x][y] is at byte offset (x + 5*y) * 8
                int keccak_offset = x + 5 * y;
                uint64_t val = 0;
                for (int j = 0; j < 8; j++) {
                    val |= static_cast<uint64_t>(rec.preimage_buffer_bytes[keccak_offset * 8 + j])
                           << (j * 8);
                }
                current_state[y][x] = val;
            }
        }
        memcpy(
            preimage_limbs,
            reinterpret_cast<uint16_t *>(&current_state[0][0]),
            sizeof(preimage_limbs)
        );
    }

    // Calculate how many rows to fill for this block
    size_t rows_for_this_block = NUM_ROUNDS;
    if (block_idx == blocks_to_fill - 1) {
        // Last block might have fewer rows
        size_t remaining = height - block_idx * NUM_ROUNDS;
        if (remaining < NUM_ROUNDS) {
            rows_for_this_block = remaining;
        }
    }

    // Generate 24 round rows. Dummy blocks (block_idx >= num_records) use zero-initialized
    // state and produce a valid keccak-f trace over zero input. Every column is explicitly
    // written: preimage and a below, intermediates by generate_trace_row_for_round, then
    // export and timestamp at the end.
    for (uint32_t round_idx = 0; round_idx < rows_for_this_block; round_idx++) {
        size_t row_idx = block_idx * NUM_ROUNDS + round_idx;
        RowSlice row(d_trace + row_idx, height);

        // Fill preimage columns (same for all rounds within a permutation)
        KECCAKF_PERM_WRITE_ARRAY(inner.preimage, preimage_limbs);

        // Fill 'a' input state directly from current_state.
        // After iota of the previous round, current_state already holds the correct
        // next-round input, so no prev-row readback is needed.
        uint16_t *state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
        COL_WRITE_ARRAY(row, KeccakfPermCols, inner.a, state_limbs);

        // Generate trace row for this round (updates current_state in-place)
        p3_keccak_air::generate_trace_row_for_round(row, round_idx, current_state);

        // Set export flag and timestamp on last round
        if (block_idx < num_records && round_idx == NUM_ROUNDS - 1) {
            KECCAKF_PERM_WRITE(inner._export, 1);
            KECCAKF_PERM_WRITE(timestamp, timestamp);
        } else {
            KECCAKF_PERM_WRITE(inner._export, 0);
            KECCAKF_PERM_WRITE(timestamp, 0);
        }
    }
}

#undef KECCAKF_PERM_WRITE
#undef KECCAKF_PERM_WRITE_ARRAY

extern "C" int _keccakf_perm_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<KeccakfOpRecord> d_records,
    size_t num_records
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(KeccakfPermCols<uint8_t>));

    uint32_t blocks_to_fill = div_ceil(height, uint32_t(NUM_ROUNDS));

    auto [grid, block] = kernel_launch_params(blocks_to_fill, 256);
    keccakf_perm_tracegen<<<grid, block>>>(
        d_trace, height, static_cast<uint32_t>(num_records), blocks_to_fill, d_records
    );
    return CHECK_KERNEL();
}
