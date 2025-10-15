#include "keccak256/keccakvm.cuh"
#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/utils.cuh"

using namespace keccak256;

// FROM https://github.com/mochimodev/cuda-hashing-algos/blob/master/keccak.cu

__device__ __constant__ uint64_t CUDA_KECCAK_CONSTS[KECCAK_ROUND] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

__device__ __forceinline__ uint64_t cuda_keccak_ROTL64(uint64_t a, uint64_t b) {
    return ROTL64(a, b);
}

__device__ void cuda_keccak_permutations(uint64_t *state) {
    int64_t *A = reinterpret_cast<int64_t *>(state);

    int64_t *a00 = A, *a01 = A + 1, *a02 = A + 2, *a03 = A + 3, *a04 = A + 4;
    int64_t *a05 = A + 5, *a06 = A + 6, *a07 = A + 7, *a08 = A + 8, *a09 = A + 9;
    int64_t *a10 = A + 10, *a11 = A + 11, *a12 = A + 12, *a13 = A + 13, *a14 = A + 14;
    int64_t *a15 = A + 15, *a16 = A + 16, *a17 = A + 17, *a18 = A + 18, *a19 = A + 19;
    int64_t *a20 = A + 20, *a21 = A + 21, *a22 = A + 22, *a23 = A + 23, *a24 = A + 24;

    for (int i = 0; i < KECCAK_ROUND; i++) {

        /* Theta */
        int64_t c0 = *a00 ^ *a05 ^ *a10 ^ *a15 ^ *a20;
        int64_t c1 = *a01 ^ *a06 ^ *a11 ^ *a16 ^ *a21;
        int64_t c2 = *a02 ^ *a07 ^ *a12 ^ *a17 ^ *a22;
        int64_t c3 = *a03 ^ *a08 ^ *a13 ^ *a18 ^ *a23;
        int64_t c4 = *a04 ^ *a09 ^ *a14 ^ *a19 ^ *a24;

        int64_t d1 = cuda_keccak_ROTL64(c1, 1) ^ c4;
        int64_t d2 = cuda_keccak_ROTL64(c2, 1) ^ c0;
        int64_t d3 = cuda_keccak_ROTL64(c3, 1) ^ c1;
        int64_t d4 = cuda_keccak_ROTL64(c4, 1) ^ c2;
        int64_t d0 = cuda_keccak_ROTL64(c0, 1) ^ c3;

        *a00 ^= d1;
        *a05 ^= d1;
        *a10 ^= d1;
        *a15 ^= d1;
        *a20 ^= d1;
        *a01 ^= d2;
        *a06 ^= d2;
        *a11 ^= d2;
        *a16 ^= d2;
        *a21 ^= d2;
        *a02 ^= d3;
        *a07 ^= d3;
        *a12 ^= d3;
        *a17 ^= d3;
        *a22 ^= d3;
        *a03 ^= d4;
        *a08 ^= d4;
        *a13 ^= d4;
        *a18 ^= d4;
        *a23 ^= d4;
        *a04 ^= d0;
        *a09 ^= d0;
        *a14 ^= d0;
        *a19 ^= d0;
        *a24 ^= d0;

        /* Rho pi */
        c1 = cuda_keccak_ROTL64(*a01, 1);
        *a01 = cuda_keccak_ROTL64(*a06, 44);
        *a06 = cuda_keccak_ROTL64(*a09, 20);
        *a09 = cuda_keccak_ROTL64(*a22, 61);
        *a22 = cuda_keccak_ROTL64(*a14, 39);
        *a14 = cuda_keccak_ROTL64(*a20, 18);
        *a20 = cuda_keccak_ROTL64(*a02, 62);
        *a02 = cuda_keccak_ROTL64(*a12, 43);
        *a12 = cuda_keccak_ROTL64(*a13, 25);
        *a13 = cuda_keccak_ROTL64(*a19, 8);
        *a19 = cuda_keccak_ROTL64(*a23, 56);
        *a23 = cuda_keccak_ROTL64(*a15, 41);
        *a15 = cuda_keccak_ROTL64(*a04, 27);
        *a04 = cuda_keccak_ROTL64(*a24, 14);
        *a24 = cuda_keccak_ROTL64(*a21, 2);
        *a21 = cuda_keccak_ROTL64(*a08, 55);
        *a08 = cuda_keccak_ROTL64(*a16, 45);
        *a16 = cuda_keccak_ROTL64(*a05, 36);
        *a05 = cuda_keccak_ROTL64(*a03, 28);
        *a03 = cuda_keccak_ROTL64(*a18, 21);
        *a18 = cuda_keccak_ROTL64(*a17, 15);
        *a17 = cuda_keccak_ROTL64(*a11, 10);
        *a11 = cuda_keccak_ROTL64(*a07, 6);
        *a07 = cuda_keccak_ROTL64(*a10, 3);
        *a10 = c1;

        /* Chi */
        c0 = *a00 ^ (~*a01 & *a02);
        c1 = *a01 ^ (~*a02 & *a03);
        *a02 ^= ~*a03 & *a04;
        *a03 ^= ~*a04 & *a00;
        *a04 ^= ~*a00 & *a01;
        *a00 = c0;
        *a01 = c1;

        c0 = *a05 ^ (~*a06 & *a07);
        c1 = *a06 ^ (~*a07 & *a08);
        *a07 ^= ~*a08 & *a09;
        *a08 ^= ~*a09 & *a05;
        *a09 ^= ~*a05 & *a06;
        *a05 = c0;
        *a06 = c1;

        c0 = *a10 ^ (~*a11 & *a12);
        c1 = *a11 ^ (~*a12 & *a13);
        *a12 ^= ~*a13 & *a14;
        *a13 ^= ~*a14 & *a10;
        *a14 ^= ~*a10 & *a11;
        *a10 = c0;
        *a11 = c1;

        c0 = *a15 ^ (~*a16 & *a17);
        c1 = *a16 ^ (~*a17 & *a18);
        *a17 ^= ~*a18 & *a19;
        *a18 ^= ~*a19 & *a15;
        *a19 ^= ~*a15 & *a16;
        *a15 = c0;
        *a16 = c1;

        c0 = *a20 ^ (~*a21 & *a22);
        c1 = *a21 ^ (~*a22 & *a23);
        *a22 ^= ~*a23 & *a24;
        *a23 ^= ~*a24 & *a20;
        *a24 ^= ~*a20 & *a21;
        *a20 = c0;
        *a21 = c1;

        /* Iota */
        *a00 ^= CUDA_KECCAK_CONSTS[i];
    }
}

// END OF FROM https://github.com/mochimodev/cuda-hashing-algos/blob/master/keccak.cu

__global__ void keccakf_kernel(
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t *block_offsets,
    uint32_t total_num_blocks,
    uint64_t *states,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits
) {
    auto record_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (record_idx >= num_records) {
        return;
    }
    auto bitwise_lookup = BitwiseOperationLookup(bitwise_lookup_ptr, bitwise_num_bits);

    auto record_mut = KeccakVmRecordMut(records + record_offsets[record_idx]);
    auto input = record_mut.input_ptr;
    auto record_len = record_mut.header.len;
    auto num_blocks = num_keccak_f(record_len);
    auto block_offset = block_offsets[record_idx];
    auto this_rec_states = reinterpret_cast<uint64_t (*)[KECCAK_STATE_SIZE]>(
        states + block_offset * KECCAK_STATE_SIZE
    );
    auto next_rec_states = reinterpret_cast<uint64_t (*)[KECCAK_STATE_SIZE]>(
        states + (block_offset + total_num_blocks) * KECCAK_STATE_SIZE
    );
    uint64_t state[KECCAK_STATE_SIZE] = {0};

    auto last_block_len = record_len - (num_blocks - 1) * KECCAK_RATE_BYTES;
    for (size_t blk = 0; blk < num_blocks; blk++) {
        uint8_t *chunk = input + blk * KECCAK_RATE_BYTES;
        bool is_last_block = (blk == num_blocks - 1);
        for (int round = 0; round < NUM_ABSORB_ROUNDS; round++) {
            uint8_t i_bytes[8] = {};
            int round_base = round * sizeof(uint64_t);
            // For the last block, zero-initialize and only copy available bytes
            if (is_last_block && round_base < last_block_len) {
                size_t bytes_to_copy = min(sizeof(uint64_t), last_block_len - round_base);
                memcpy(i_bytes, chunk + round_base, bytes_to_copy);
            } else if (!is_last_block) {
                memcpy(i_bytes, chunk + round_base, sizeof(uint64_t));
            }

            // Handle Keccak spec padding
            if (is_last_block) {
                int round_base = round * 8;
#pragma unroll 8
                for (int i = 0; i < 8; i++) {
                    int global_idx = round_base + i;
                    if (global_idx >= last_block_len) {
                        i_bytes[i] = 0;
                        if (global_idx == last_block_len)
                            i_bytes[i] = 0x01;
                        // WARNING: it is possible for i_bytes[i] = 0x81=0b10000001 in the case all padding happens in a single byte
                        if (global_idx == KECCAK_RATE_BYTES - 1)
                            i_bytes[i] ^= 0x80;
                    }
                }
            }

#pragma unroll 8
            for (int i = 0; i < 8; i++) {
                uint8_t byte = i_bytes[i];
                uint8_t s_byte = (state[round] >> (i * 8)) & 0xff;
                if (blk != 0) {
                    bitwise_lookup.add_xor(byte, s_byte);
                }
                state[round] ^= (static_cast<uint64_t>(byte) << (i * 8));
            }
        }
#pragma unroll 8
        for (int i = 0; i < KECCAK_STATE_SIZE; i++) {
            this_rec_states[blk][i] = state[i];
        }
        cuda_keccak_permutations(state);
#pragma unroll 8
        for (int i = 0; i < KECCAK_STATE_SIZE; i++) {
            next_rec_states[blk][i] = state[i];
        }
    }
}

extern "C" int _keccakf_kernel(
    uint8_t *records,
    size_t num_records,
    size_t *record_offsets,
    uint32_t *block_offsets,
    uint32_t total_num_blocks,
    uint64_t *states,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits
) {
    auto [grid, block] = kernel_launch_params(num_records, 256);
    keccakf_kernel<<<grid, block>>>(
        records,
        num_records,
        record_offsets,
        block_offsets,
        total_num_blocks,
        states,
        bitwise_lookup_ptr,
        bitwise_num_bits
    );
    return CHECK_KERNEL();
}
