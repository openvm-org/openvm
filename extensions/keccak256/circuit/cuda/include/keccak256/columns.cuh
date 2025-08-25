#pragma once

#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace keccak256;
using namespace p3_keccak_air;

template <typename T> struct KeccakPermCols {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    T step_flags[NUM_ROUNDS];

    /// A register which indicates if a row should be exported, i.e. included in a multiset equality
    /// argument. Should be 1 only for certain rows which are final steps, i.e. with
    /// `step_flags[23] = 1`.
    T _export;

    /// Permutation inputs, stored in y-major order.
    T preimage[5][5][U64_LIMBS];

    T a[5][5][U64_LIMBS];

    /// ```ignore
    /// C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4])
    /// ```
    T c[5][64];

    /// ```ignore
    /// C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1])
    /// ```
    T c_prime[5][64];

    // Note: D is inlined, not stored in the witness.
    /// ```ignore
    /// A'[x, y] = xor(A[x, y], D[x])
    ///          = xor(A[x, y], C[x - 1], ROT(C[x + 1], 1))
    /// ```
    T a_prime[5][5][64];

    /// ```ignore
    /// A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    /// ```
    T a_prime_prime[5][5][U64_LIMBS];

    /// The bits of `A''[0, 0]`.
    T a_prime_prime_0_0_bits[64];

    /// ```ignore
    /// A'''[0, 0, z] = A''[0, 0, z] ^ RC[k, z]
    /// ```
    T a_prime_prime_prime_0_0_limbs[U64_LIMBS];
};

template <typename T> struct KeccakSpongeCols {
    /// Only used on first row of a round to determine whether the state
    /// prior to absorb should be reset to all 0s.
    /// Constrained to be zero if not first round.
    T is_new_start;

    /// Whether the current byte is a padding byte.
    /// If this row represents a full input block, this should contain all 0s.
    T is_padding_byte[KECCAK_RATE_BYTES];

    /// The block being absorbed, which may contain input bytes and padding bytes.
    T block_bytes[KECCAK_RATE_BYTES];

    /// For each of the first [KECCAK_RATE_U16S] `u16` limbs in the state,
    /// the most significant byte of the limb.
    /// Here `state` is the postimage state if last round and the preimage
    /// state if first round. It can be junk if not first or last round.
    T state_hi[KECCAK_RATE_U16S];
};

template <typename T> struct KeccakInstructionCols {
    /// Program counter
    T pc;
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    T is_enabled;
    /// Is enabled and first round of block. Used to lower constraint degree.
    /// is_enabled * inner.step_flags\[0\]
    T is_enabled_first_round;
    /// The starting timestamp to use for memory access in this row.
    /// A single row will do multiple memory accesses.
    T start_timestamp;
    /// Pointer to address space 1 `dst` register
    T dst_ptr;
    /// Pointer to address space 1 `src` register
    T src_ptr;
    /// Pointer to address space 1 `len` register
    T len_ptr;
    // Register values
    /// dst <- \[dst_ptr:4\]_1
    T dst[RV32_REGISTER_NUM_LIMBS];
    /// src <- \[src_ptr:4\]_1
    /// We store src_limbs\[i\] = \[src_ptr + i + 1\]_1 and src = u32(\[src_ptr:4\]_1) from which
    /// \[src_ptr\]_1 can be recovered by linear combination.
    /// We do this because `src` needs to be incremented between keccak-f permutations.
    T src_limbs[RV32_REGISTER_NUM_LIMBS - 1];
    T src;
    /// len <- \[len_ptr:4\]_1
    /// We store len_limbs\[i\] = \[len_ptr + i + 1\]_1 and remaining_len = u32(\[len_ptr:4\]_1)
    /// from which \[len_ptr\]_1 can be recovered by linear combination.
    /// We do this because `remaining_len` needs to be decremented between keccak-f permutations.
    T len_limbs[RV32_REGISTER_NUM_LIMBS - 1];
    /// The remaining length of the unpadded input, in bytes.
    /// If `is_new_start` is true and `is_enabled` is true, this must be equal to `u32(len)`.
    T remaining_len;
};

template <typename T> struct KeccakMemoryCols {
    MemoryReadAuxCols<T> register_aux[KECCAK_REGISTER_READS];
    MemoryReadAuxCols<T> absorb_reads[KECCAK_ABSORB_READS];
    MemoryWriteAuxCols<T, KECCAK_WORD_SIZE> digest_writes[KECCAK_DIGEST_WRITES];
    /// The input bytes are batch read in blocks of private constant KECCAK_WORD_SIZE bytes.
    /// However if the input length is not a multiple of KECCAK_WORD_SIZE, we read into
    /// `partial_block` more bytes than we need. On the other hand `block_bytes` expects
    /// only the partial block of bytes and then the correctly padded bytes.
    /// We will select between `partial_block` and `block_bytes` for what to read from memory.
    /// We never read a full padding block, so the first byte is always ok.
    T partial_block[KECCAK_WORD_SIZE - 1];
};

template <typename T> struct KeccakVmCols {
    /// Columns for keccak-f permutation
    KeccakPermCols<T> inner;
    /// Columns for sponge and padding
    KeccakSpongeCols<T> sponge;
    /// Columns for instruction interface and register access
    KeccakInstructionCols<T> instruction;
    /// Auxiliary columns for offline memory checking
    KeccakMemoryCols<T> mem_oc;
};

constexpr size_t NUM_KECCAK_VM_COLS = sizeof(KeccakVmCols<uint8_t>);
constexpr size_t NUM_KECCAK_PERM_COLS = sizeof(KeccakPermCols<uint8_t>);
constexpr size_t NUM_KECCAK_SPONGE_COLS = sizeof(KeccakSpongeCols<uint8_t>);
constexpr size_t NUM_KECCAK_INSTRUCTION_COLS = sizeof(KeccakInstructionCols<uint8_t>);
constexpr size_t NUM_KECCAK_MEMORY_COLS = sizeof(KeccakMemoryCols<uint8_t>);
