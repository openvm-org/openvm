#pragma once

#include "primitives/constants.h"
#include "primitives/utils.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace sha256;

__device__ __host__ inline uint32_t get_sha256_num_blocks(uint32_t len) {
    uint32_t bit_len = len * 8;
    uint32_t padded_bit_len = bit_len + 1 + 64;
    return (padded_bit_len + 511) >> 9;
}

struct Sha256VmRecordHeader {
    uint32_t from_pc;
    uint32_t timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t dst_ptr;
    uint32_t src_ptr;
    uint32_t len;
    MemoryReadAuxRecord register_reads_aux[SHA256_REGISTER_READS];
    MemoryWriteBytesAuxRecord<SHA256_WRITE_SIZE> write_aux;
};

struct Sha256VmRecordMut {
    Sha256VmRecordHeader *header;
    uint8_t *input;
    MemoryReadAuxRecord *read_aux;

    __device__ __host__ __forceinline__ static uint32_t next_multiple_of(
        uint32_t value,
        uint32_t alignment
    ) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    __device__ __host__ __forceinline__ Sha256VmRecordMut(uint8_t *record_buf) {
        // Use memcpy for safe unaligned access instead of reinterpret_cast
        header = reinterpret_cast<Sha256VmRecordHeader *>(record_buf);

        uint32_t offset = sizeof(Sha256VmRecordHeader);

        input = record_buf + offset;
        uint32_t num_blocks = get_sha256_num_blocks(header->len);
        uint32_t input_size = num_blocks * SHA256_BLOCK_U8S;

        offset += input_size;
        offset = next_multiple_of(offset, alignof(MemoryReadAuxRecord));

        read_aux = reinterpret_cast<MemoryReadAuxRecord *>(record_buf + offset);
    }
};

__device__ static constexpr uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

__device__ static constexpr uint32_t SHA256_H[8] = {
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
};

__device__ static constexpr uint32_t SHA256_INVALID_CARRY_A[SHA256_ROUNDS_PER_ROW]
                                                           [SHA256_WORD_U16S] = {
                                                               {1230919683, 1162494304},
                                                               {266373122, 1282901987},
                                                               {1519718403, 1008990871},
                                                               {923381762, 330807052},
};

__device__ static constexpr uint32_t SHA256_INVALID_CARRY_E[SHA256_ROUNDS_PER_ROW]
                                                           [SHA256_WORD_U16S] = {
                                                               {204933122, 1994683449},
                                                               {443873282, 1544639095},
                                                               {719953922, 1888246508},
                                                               {194580482, 1075725211},
};

__device__ inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ ((~x) & z); }

__device__ inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ inline uint32_t big_sig0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }

__device__ inline uint32_t big_sig1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }

__device__ inline uint32_t small_sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }

__device__ inline uint32_t small_sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

enum class PaddingFlags : uint32_t {
    NotConsidered = 0,
    NotPadding = 1,
    FirstPadding0 = 2,
    FirstPadding1 = 3,
    FirstPadding2 = 4,
    FirstPadding3 = 5,
    FirstPadding4 = 6,
    FirstPadding5 = 7,
    FirstPadding6 = 8,
    FirstPadding7 = 9,
    FirstPadding8 = 10,
    FirstPadding9 = 11,
    FirstPadding10 = 12,
    FirstPadding11 = 13,
    FirstPadding12 = 14,
    FirstPadding13 = 15,
    FirstPadding14 = 16,
    FirstPadding15 = 17,
    FirstPadding0_LastRow = 18,
    FirstPadding1_LastRow = 19,
    FirstPadding2_LastRow = 20,
    FirstPadding3_LastRow = 21,
    FirstPadding4_LastRow = 22,
    FirstPadding5_LastRow = 23,
    FirstPadding6_LastRow = 24,
    FirstPadding7_LastRow = 25,
    EntirePaddingLastRow = 26,
    EntirePadding = 27,
    COUNT = 28,
};

__device__ inline MemoryReadAuxRecord *get_read_aux_record(
    const Sha256VmRecordMut *record,
    uint32_t block_idx,
    uint32_t read_row_idx
) {
    return &record->read_aux[block_idx * SHA256_NUM_READ_ROWS + read_row_idx];
}

__device__ inline uint16_t u32_to_u16_limb(uint32_t value, uint32_t limb_idx) {
    return ((uint16_t *)&value)[limb_idx];
}

__device__ inline void sha256_pad_input(
    const uint8_t *input,
    uint32_t len,
    uint8_t *padded_output,
    uint32_t num_blocks
) {
#pragma unroll
    for (uint32_t i = 0; i < len; i++) {
        padded_output[i] = input[i];
    }

    padded_output[len] = 0x80;

    uint32_t total_len = num_blocks * SHA256_BLOCK_U8S;
    for (uint32_t i = len + 1; i < total_len - 8; i++) {
        padded_output[i] = 0;
    }

    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        padded_output[total_len - 8 + i] = static_cast<uint8_t>(bit_len >> (8 * (7 - i)));
    }
}