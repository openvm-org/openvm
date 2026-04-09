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

static constexpr uint32_t KECCAK_STATE_WORDS = 25;

// Two-phase keccak-f trace generation for coalesced column-major stores.
// Trace layout is trace[col * height + row]; threads writing adjacent rows coalesce.
//
// Phase 1 (1 thread per permutation):
//   - runs 24 keccak-f rounds (theta/rho/pi/chi/iota)
//   - stores the 25-lane u64 round-input state before each round
//     into scratch: d_round_states[perm][round][lane] (~4.8 KB/perm)
//
// Phase 2 (1 thread per row = 1 round of 1 permutation):
//   - loads round-input state from scratch
//   - recomputes that round to materialize intermediates (c, c', a', a'', ...)
//   - writes all 2634 trace columns

// Phase 1: compute keccak-f, store round-input states to scratch
// each thread processes one permutation (24 rounds)
__global__ void keccakf_perm_phase1(
    uint64_t *__restrict__ d_round_states, // [blocks_to_fill][24][25]
    uint32_t num_records,
    uint32_t blocks_to_fill,
    DeviceBufferConstView<KeccakfOpRecord> d_records
) {
    uint32_t perm_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (perm_idx >= blocks_to_fill) {
        return;
    }

    __align__(16) uint64_t current_state[5][5] = {0};

    if (perm_idx < num_records) {
        auto const &rec = d_records[perm_idx];

        // Convert preimage bytes to u64 state with coordinate transposition
        // generate_trace_row_for_round expects current_state[y][x] = A[x][y] (keccak notation)
        // In keccak buffer: A[x][y] is stored at offset (x + 5*y)
        // So current_state[y][x] should get keccak_buffer[x + 5*y]
        for (uint32_t x = 0; x < 5; x++) {
            for (uint32_t y = 0; y < 5; y++) {
                // keccak spec: A[x][y] is at byte offset (x + 5*y) * 8
                uint32_t keccak_offset = x + 5 * y;
                uint64_t val = 0;
                for (uint32_t j = 0; j < 8; j++) {
                    val |= static_cast<uint64_t>(rec.preimage_buffer_bytes[keccak_offset * 8 + j])
                           << (j * 8);
                }
                current_state[y][x] = val;
            }
        }
    }

    // Store round-input state before each round, then advance
    uint64_t *flat = &current_state[0][0];
    for (uint32_t round_idx = 0; round_idx < NUM_ROUNDS; round_idx++) {
        size_t off = (static_cast<size_t>(perm_idx) * NUM_ROUNDS + round_idx) * KECCAK_STATE_WORDS;
#pragma unroll
        for (uint32_t i = 0; i < KECCAK_STATE_WORDS; i++) {
            d_round_states[off + i] = flat[i];
        }
        p3_keccak_air::apply_round_in_place(round_idx, current_state);
    }
}

// Phase 2: write column-major trace from cached round states
// Each thread writes one row; adjacent threads write adjacent rows (coalesced)
__global__ void keccakf_perm_phase2(
    Fp *__restrict__ d_trace,
    size_t height,
    uint32_t num_records,
    DeviceBufferConstView<KeccakfOpRecord> d_records,
    uint64_t const *__restrict__ d_round_states // [blocks_to_fill][24][25]
) {
    size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height) {
        return;
    }

    uint32_t perm_idx = static_cast<uint32_t>(row_idx / NUM_ROUNDS);
    uint32_t round_idx = static_cast<uint32_t>(row_idx % NUM_ROUNDS);

    // Load round-input state from scratch
    __align__(16) uint64_t current_state[5][5];
    size_t off = (static_cast<size_t>(perm_idx) * NUM_ROUNDS + round_idx) * KECCAK_STATE_WORDS;
    uint64_t *flat = &current_state[0][0];
#pragma unroll
    for (uint32_t i = 0; i < KECCAK_STATE_WORDS; i++) {
        flat[i] = d_round_states[off + i];
    }

    RowSlice row(d_trace + row_idx, height);

    // Fill preimage columns (invariant across rounds, read from round 0 of this permutation)
    size_t preimage_off = static_cast<size_t>(perm_idx) * NUM_ROUNDS * KECCAK_STATE_WORDS;
    auto const *preimage_limbs = reinterpret_cast<uint16_t const *>(&d_round_states[preimage_off]);
    KECCAKF_PERM_WRITE_ARRAY(inner.preimage, preimage_limbs);

    // Fill 'a' input state from current round state
    uint16_t *state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
    COL_WRITE_ARRAY(row, KeccakfPermCols, inner.a, state_limbs);

    // Generate trace row for this round (writes c, c_prime, a_prime, a_prime_prime, etc.)
    p3_keccak_air::generate_trace_row_for_round(row, round_idx, current_state);

    // Set export flag and timestamp on last round of valid records
    if (perm_idx < num_records && round_idx == NUM_ROUNDS - 1) {
        KECCAKF_PERM_WRITE(inner._export, 1);
        KECCAKF_PERM_WRITE(timestamp, d_records[perm_idx].timestamp);
    } else {
        KECCAKF_PERM_WRITE(inner._export, 0);
        KECCAKF_PERM_WRITE(timestamp, 0);
    }
}

#undef KECCAKF_PERM_WRITE
#undef KECCAKF_PERM_WRITE_ARRAY

extern "C" int _keccakf_perm_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<KeccakfOpRecord> d_records,
    size_t num_records,
    uint64_t *d_round_states,
    size_t round_state_words
) {
    assert((height & (height - 1)) == 0);
    assert(width == sizeof(KeccakfPermCols<uint8_t>));

    uint32_t blocks_to_fill = div_ceil(height, uint32_t(NUM_ROUNDS));
    assert(
        round_state_words >= static_cast<size_t>(blocks_to_fill) * NUM_ROUNDS * KECCAK_STATE_WORDS
    );

    // Phase 1: compute keccak-f, store round states to scratch
    auto [p1_grid, p1_block] = kernel_launch_params(blocks_to_fill, 128);
    keccakf_perm_phase1<<<p1_grid, p1_block>>>(
        d_round_states, static_cast<uint32_t>(num_records), blocks_to_fill, d_records
    );
    int result = CHECK_KERNEL();
    if (result != 0) {
        return result;
    }

    // Phase 2: write trace with coalesced stores (one thread per row)
    auto [p2_grid, p2_block] = kernel_launch_params(height, 256);
    keccakf_perm_phase2<<<p2_grid, p2_block>>>(
        d_trace, height, static_cast<uint32_t>(num_records), d_records, d_round_states
    );
    return CHECK_KERNEL();
}
