#pragma once

#include <cstdint>
#include "primitives/constants.h"
#include "primitives/utils.cuh"

namespace sha2 {

// Common VM constants across SHA-2 variants.
inline constexpr size_t SHA2_REGISTER_READS = 3;
inline constexpr size_t SHA2_READ_SIZE = 16;
inline constexpr size_t SHA2_WRITE_SIZE = 4;
inline constexpr size_t SHA2_MAIN_READ_SIZE = 4;

template <
    typename WordT,
    size_t WORD_BITS_,
    size_t BLOCK_WORDS_,
    size_t ROUNDS_PER_ROW_,
    size_t ROUNDS_PER_BLOCK_,
    size_t HASH_WORDS_,
    size_t ROW_VAR_CNT_,
    size_t MESSAGE_LENGTH_BITS_>
struct Sha2VariantBase {
    using Word = WordT;

    static constexpr size_t WORD_BITS = WORD_BITS_;
    static constexpr size_t BLOCK_WORDS = BLOCK_WORDS_;
    static constexpr size_t ROUNDS_PER_ROW = ROUNDS_PER_ROW_;
    static constexpr size_t ROUNDS_PER_BLOCK = ROUNDS_PER_BLOCK_;
    static constexpr size_t HASH_WORDS = HASH_WORDS_;
    static constexpr size_t ROW_VAR_CNT = ROW_VAR_CNT_;
    static constexpr size_t MESSAGE_LENGTH_BITS = MESSAGE_LENGTH_BITS_;

    static constexpr size_t WORD_U16S = WORD_BITS / 16;
    static constexpr size_t WORD_U8S = WORD_BITS / 8;
    static constexpr size_t WORD_BYTES = WORD_U8S;
    static constexpr size_t BLOCK_U8S = BLOCK_WORDS * WORD_U8S;
    static constexpr size_t BLOCK_BYTES = BLOCK_U8S;
    static constexpr size_t BLOCK_BITS = BLOCK_WORDS * WORD_BITS;
    static constexpr size_t ROUND_ROWS = ROUNDS_PER_BLOCK / ROUNDS_PER_ROW;
    static constexpr size_t MESSAGE_ROWS = BLOCK_WORDS / ROUNDS_PER_ROW;
    static constexpr size_t ROUNDS_PER_ROW_MINUS_ONE = ROUNDS_PER_ROW - 1;

    static constexpr size_t NUM_READ_ROWS = BLOCK_U8S / SHA2_READ_SIZE;
    static constexpr size_t STATE_BYTES = HASH_WORDS * WORD_U8S;
    static constexpr size_t BLOCK_READS = BLOCK_U8S / SHA2_MAIN_READ_SIZE;
    static constexpr size_t STATE_READS = STATE_BYTES / SHA2_MAIN_READ_SIZE;
    static constexpr size_t STATE_WRITES = STATE_BYTES / SHA2_WRITE_SIZE;
    static constexpr size_t TIMESTAMP_DELTA =
        BLOCK_READS + STATE_READS + STATE_WRITES + SHA2_REGISTER_READS;
};

// SHA-256 constants
static constexpr uint32_t SHA256_K_HOST[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static constexpr uint32_t SHA256_H_HOST[8] = {
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
};

// SHA-512 constants
static constexpr uint64_t SHA512_K_HOST[80] = {
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
};

static constexpr uint64_t SHA512_H_HOST[8] = {
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
};

// Device copies of the constants
__device__ __constant__ uint32_t SHA256_K_DEV[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};
__device__ __constant__ uint32_t SHA256_H_DEV[8] = {
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
};
__device__ __constant__ uint64_t SHA512_K_DEV[80] = {
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
};
__device__ __constant__ uint64_t SHA512_H_DEV[8] = {
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
};

struct Sha256Variant : Sha2VariantBase<uint32_t, 32, 16, 4, 64, 8, 5, 64> {
    static constexpr size_t ROWS_PER_BLOCK = 17;
    static constexpr int SIGMA_A0 = 2;
    static constexpr int SIGMA_A1 = 13;
    static constexpr int SIGMA_A2 = 22;
    static constexpr int SIGMA_E0 = 6;
    static constexpr int SIGMA_E1 = 11;
    static constexpr int SIGMA_E2 = 25;
    static constexpr int SIGMA0_ROT1 = 7;
    static constexpr int SIGMA0_ROT2 = 18;
    static constexpr int SIGMA0_SHR = 3;
    static constexpr int SIGMA1_ROT1 = 17;
    static constexpr int SIGMA1_ROT2 = 19;
    static constexpr int SIGMA1_SHR = 10;

    __device__ __host__ static inline Word K(size_t i) {
#ifdef __CUDA_ARCH__
        return SHA256_K_DEV[i];
#else
        return SHA256_K_HOST[i];
#endif
    }
    __device__ __host__ static inline Word H(size_t i) {
#ifdef __CUDA_ARCH__
        return SHA256_H_DEV[i];
#else
        return SHA256_H_HOST[i];
#endif
    }
};

struct Sha512Variant : Sha2VariantBase<uint64_t, 64, 16, 4, 80, 8, 6, 128> {
    static constexpr size_t ROWS_PER_BLOCK = 21;
    static constexpr int SIGMA_A0 = 28;
    static constexpr int SIGMA_A1 = 34;
    static constexpr int SIGMA_A2 = 39;
    static constexpr int SIGMA_E0 = 14;
    static constexpr int SIGMA_E1 = 18;
    static constexpr int SIGMA_E2 = 41;
    static constexpr int SIGMA0_ROT1 = 1;
    static constexpr int SIGMA0_ROT2 = 8;
    static constexpr int SIGMA0_SHR = 7;
    static constexpr int SIGMA1_ROT1 = 19;
    static constexpr int SIGMA1_ROT2 = 61;
    static constexpr int SIGMA1_SHR = 6;

    __device__ __host__ static inline Word K(size_t i) {
#ifdef __CUDA_ARCH__
        return SHA512_K_DEV[i];
#else
        return SHA512_K_HOST[i];
#endif
    }
    __device__ __host__ static inline Word H(size_t i) {
#ifdef __CUDA_ARCH__
        return SHA512_H_DEV[i];
#else
        return SHA512_H_HOST[i];
#endif
    }
};

template <typename WordT> __device__ __host__ __forceinline__ WordT rotr_generic(WordT value, int n);

template <> inline __device__ __host__ uint32_t rotr_generic<uint32_t>(uint32_t value, int n) {
    return (value >> n) | (value << (32 - n));
}

template <> inline __device__ __host__ uint64_t rotr_generic<uint64_t>(uint64_t value, int n) {
    return (value >> n) | (value << (64 - n));
}

template <typename V>
__device__ __host__ __forceinline__ typename V::Word big_sig0(typename V::Word x) {
    return rotr_generic<typename V::Word>(x, V::SIGMA_A0) ^
           rotr_generic<typename V::Word>(x, V::SIGMA_A1) ^
           rotr_generic<typename V::Word>(x, V::SIGMA_A2);
}

template <typename V>
__device__ __host__ __forceinline__ typename V::Word big_sig1(typename V::Word x) {
    return rotr_generic<typename V::Word>(x, V::SIGMA_E0) ^
           rotr_generic<typename V::Word>(x, V::SIGMA_E1) ^
           rotr_generic<typename V::Word>(x, V::SIGMA_E2);
}

template <typename V>
__device__ __host__ __forceinline__ typename V::Word small_sig0(typename V::Word x) {
    return rotr_generic<typename V::Word>(x, V::SIGMA0_ROT1) ^
           rotr_generic<typename V::Word>(x, V::SIGMA0_ROT2) ^ (x >> V::SIGMA0_SHR);
}

template <typename V>
__device__ __host__ __forceinline__ typename V::Word small_sig1(typename V::Word x) {
    return rotr_generic<typename V::Word>(x, V::SIGMA1_ROT1) ^
           rotr_generic<typename V::Word>(x, V::SIGMA1_ROT2) ^ (x >> V::SIGMA1_SHR);
}

template <typename V>
__device__ __host__ __forceinline__ typename V::Word ch(
    typename V::Word x,
    typename V::Word y,
    typename V::Word z
) {
    return (x & y) ^ ((~x) & z);
}

template <typename V>
__device__ __host__ __forceinline__ typename V::Word maj(
    typename V::Word x,
    typename V::Word y,
    typename V::Word z
) {
    return (x & y) ^ (x & z) ^ (y & z);
}

template <typename V> __device__ __host__ __forceinline__ uint32_t get_num_blocks(uint32_t len) {
    constexpr uint32_t length_bits = static_cast<uint32_t>(V::MESSAGE_LENGTH_BITS);
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    uint64_t padded_bit_len = bit_len + 1 + length_bits;
    return static_cast<uint32_t>((padded_bit_len + (V::BLOCK_BITS - 1)) / V::BLOCK_BITS);
}

} // namespace sha2
