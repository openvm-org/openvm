#include "fp.h"
#include "arch/rvr/preflight.cuh"
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
using p3_keccak_air::NUM_ROUNDS;

static constexpr uint32_t KECCAK_STATE_WORDS = 25;
static constexpr uint32_t KECCAKF_PERM_REPLAY_ERROR = 821;

#define KECCAKF_PERM_WRITE(FIELD, VALUE) COL_WRITE_VALUE(row, KeccakfPermCols, FIELD, VALUE)
#define KECCAKF_PERM_WRITE_ARRAY(FIELD, VALUES) COL_WRITE_ARRAY(row, KeccakfPermCols, FIELD, VALUES)

__global__ void keccakf_perm_replay_phase1(
    uint64_t *__restrict__ d_round_states,
    uint32_t num_records,
    uint32_t blocks_to_fill,
    uint64_t const *__restrict__ d_preimages
) {
    uint32_t perm_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (perm_idx >= blocks_to_fill) return;

    __align__(16) uint64_t current_state[5][5] = {0};
    if (perm_idx < num_records) {
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                current_state[y][x] =
                    d_preimages[static_cast<size_t>(perm_idx) * KECCAK_STATE_WORDS + x + 5 * y];
            }
        }
    }

    uint64_t *flat = &current_state[0][0];
    for (uint32_t round_idx = 0; round_idx < NUM_ROUNDS; round_idx++) {
        size_t offset =
            (static_cast<size_t>(perm_idx) * NUM_ROUNDS + round_idx) * KECCAK_STATE_WORDS;
#pragma unroll
        for (uint32_t i = 0; i < KECCAK_STATE_WORDS; i++) {
            d_round_states[offset + i] = flat[i];
        }
        p3_keccak_air::apply_round_in_place(round_idx, current_state);
    }
}

__global__ void keccakf_perm_replay_phase2(
    Fp *__restrict__ d_trace,
    size_t height,
    uint32_t num_records,
    DeviceBufferConstView<PreflightProgramEvent> program,
    DeviceBufferConstView<RvrReplayStep> steps,
    size_t step_start,
    uint64_t const *__restrict__ d_round_states,
    uint32_t *error
) {
    size_t row_idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (row_idx >= height) return;

    uint32_t perm_idx = static_cast<uint32_t>(row_idx / NUM_ROUNDS);
    uint32_t round_idx = static_cast<uint32_t>(row_idx % NUM_ROUNDS);
    __align__(16) uint64_t current_state[5][5];
    size_t offset =
        (static_cast<size_t>(perm_idx) * NUM_ROUNDS + round_idx) * KECCAK_STATE_WORDS;
    uint64_t *flat = &current_state[0][0];
#pragma unroll
    for (uint32_t i = 0; i < KECCAK_STATE_WORDS; i++) flat[i] = d_round_states[offset + i];

    RowSlice row(d_trace + row_idx, height);
    size_t preimage_offset = static_cast<size_t>(perm_idx) * NUM_ROUNDS * KECCAK_STATE_WORDS;
    KECCAKF_PERM_WRITE_ARRAY(
        inner.preimage, reinterpret_cast<uint16_t const *>(&d_round_states[preimage_offset])
    );
    COL_WRITE_ARRAY(
        row,
        KeccakfPermCols,
        inner.a,
        reinterpret_cast<uint16_t const *>(&current_state[0][0])
    );
    p3_keccak_air::generate_trace_row_for_round(row, round_idx, current_state);

    if (perm_idx < num_records && round_idx == NUM_ROUNDS - 1) {
        auto const &step = steps[step_start + perm_idx];
        size_t program_index = step.program_index;
        if (program_index >= program.len() || program.len() - program_index <= 1) {
            preflight_set_error(error, KECCAKF_PERM_REPLAY_ERROR);
            KECCAKF_PERM_WRITE(inner._export, 0);
            KECCAKF_PERM_WRITE(timestamp, 0);
            return;
        }
        KECCAKF_PERM_WRITE(inner._export, 1);
        KECCAKF_PERM_WRITE(timestamp, program[program_index].timestamp);
    } else {
        KECCAKF_PERM_WRITE(inner._export, 0);
        KECCAKF_PERM_WRITE(timestamp, 0);
    }
}

extern "C" int _keccakf_perm_replay_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<PreflightProgramEvent> d_program,
    DeviceBufferConstView<RvrReplayStep> d_steps,
    size_t step_start,
    size_t num_steps,
    uint64_t const *d_preimages,
    size_t preimage_words,
    uint64_t *d_round_states,
    size_t round_state_words,
    uint32_t *d_error,
    cudaStream_t stream
) {
    assert(width == sizeof(KeccakfPermCols<uint8_t>));
    assert(step_start <= d_steps.len() && num_steps <= d_steps.len() - step_start);
    assert(preimage_words >= num_steps * KECCAK_STATE_WORDS);
    uint32_t blocks_to_fill = div_ceil(height, uint32_t(NUM_ROUNDS));
    assert(
        round_state_words >= static_cast<size_t>(blocks_to_fill) * NUM_ROUNDS * KECCAK_STATE_WORDS
    );

    auto [p1_grid, p1_block] = kernel_launch_params(blocks_to_fill, 128);
    keccakf_perm_replay_phase1<<<p1_grid, p1_block, 0, stream>>>(
        d_round_states,
        static_cast<uint32_t>(num_steps),
        blocks_to_fill,
        d_preimages
    );
    int result = CHECK_KERNEL();
    if (result != 0) return result;

    auto [p2_grid, p2_block] = kernel_launch_params(height, 256);
    keccakf_perm_replay_phase2<<<p2_grid, p2_block, 0, stream>>>(
        d_trace,
        height,
        static_cast<uint32_t>(num_steps),
        d_program,
        d_steps,
        step_start,
        d_round_states,
        d_error
    );
    return CHECK_KERNEL();
}

#undef KECCAKF_PERM_WRITE
#undef KECCAKF_PERM_WRITE_ARRAY
