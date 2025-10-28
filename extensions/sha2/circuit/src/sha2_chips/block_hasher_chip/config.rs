use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};

use crate::{Sha256Config, Sha2DigestColsRef, Sha2RoundColsRef, Sha384Config, Sha512Config};

#[repr(u32)]
#[derive(num_enum::TryFromPrimitive, num_enum::IntoPrimitive, Copy, Clone, Debug)]
pub enum Sha2Variant {
    Sha256,
    Sha512,
    Sha384,
}

pub trait Sha2BlockHasherConfig: Send + Sync + Clone {
    // --- Required ---

    type Word: 'static
        + Shr<usize, Output = Self::Word>
        + Shl<usize, Output = Self::Word>
        + BitAnd<Output = Self::Word>
        + Not<Output = Self::Word>
        + BitXor<Output = Self::Word>
        + BitOr<Output = Self::Word>
        + RotateRight
        + WrappingAdd
        + PartialEq
        + From<u32>
        + TryInto<u32, Error: std::fmt::Debug>
        + Copy
        + Send
        + Sync;
    // Differentiate between the SHA-2 variants
    const VARIANT: Sha2Variant;
    /// Number of bits in a SHA word
    const WORD_BITS: usize;
    /// Number of words in a SHA block
    const BLOCK_WORDS: usize;
    /// Number of rows per block
    const ROWS_PER_BLOCK: usize;
    /// Number of rounds per row. Must divide Self::ROUNDS_PER_BLOCK
    const ROUNDS_PER_ROW: usize;
    /// Number of rounds per block. Must be a multiple of Self::ROUNDS_PER_ROW
    const ROUNDS_PER_BLOCK: usize;
    /// Number of words in a SHA hash
    const HASH_WORDS: usize;
    /// Number of vars needed to encode the row index with [Encoder]
    const ROW_VAR_CNT: usize;

    ///  To optimize the trace generation of invalid rows, we precompute those values.
    // these should be appropriately sized for the config
    fn get_invalid_carry_a(round_num: usize) -> &'static [u32];
    fn get_invalid_carry_e(round_num: usize) -> &'static [u32];

    /// We also store the SHA constants K and H
    fn get_k() -> &'static [Self::Word];
    fn get_h() -> &'static [Self::Word];

    // --- Provided ---

    /// Number of 16-bit limbs in a SHA word
    const WORD_U16S: usize = Self::WORD_BITS / 16;
    /// Number of 8-bit limbs in a SHA word
    const WORD_U8S: usize = Self::WORD_BITS / 8;
    /// Number of cells in a SHA block
    const BLOCK_U8S: usize = Self::BLOCK_WORDS * Self::WORD_U8S;
    /// Number of bits in a SHA block
    const BLOCK_BITS: usize = Self::BLOCK_WORDS * Self::WORD_BITS;
    /// Number of rows used for the sha rounds
    const ROUND_ROWS: usize = Self::ROUNDS_PER_BLOCK / Self::ROUNDS_PER_ROW;
    /// Number of rows used for the message
    const MESSAGE_ROWS: usize = Self::BLOCK_WORDS / Self::ROUNDS_PER_ROW;
    /// Number of rounds per row minus one (needed for one of the column structs)
    const ROUNDS_PER_ROW_MINUS_ONE: usize = Self::ROUNDS_PER_ROW - 1;
    /// Width of the Sha2RoundCols
    const ROUND_WIDTH: usize = Sha2RoundColsRef::<u8>::width::<Self>();
    /// Width of the Sha2DigestCols
    const DIGEST_WIDTH: usize = Sha2DigestColsRef::<u8>::width::<Self>();
    /// Width of the Sha2BlockHasherCols
    const WIDTH: usize = if Self::ROUND_WIDTH > Self::DIGEST_WIDTH {
        Self::ROUND_WIDTH
    } else {
        Self::DIGEST_WIDTH
    };
}

impl Sha2BlockHasherConfig for Sha256Config {
    // ==== Do not change these constants! ====
    const VARIANT: Sha2Variant = Sha2Variant::Sha256;
    type Word = u32;
    /// Number of bits in a SHA256 word
    const WORD_BITS: usize = 32;
    /// Number of words in a SHA256 block
    const BLOCK_WORDS: usize = 16;
    /// Number of rows per block
    const ROWS_PER_BLOCK: usize = 17;
    /// Number of rounds per row
    const ROUNDS_PER_ROW: usize = 4;
    /// Number of rounds per block
    const ROUNDS_PER_BLOCK: usize = 64;
    /// Number of words in a SHA256 hash
    const HASH_WORDS: usize = 8;
    /// Number of vars needed to encode the row index with [Encoder]
    const ROW_VAR_CNT: usize = 5;

    fn get_invalid_carry_a(round_num: usize) -> &'static [u32] {
        &SHA256_INVALID_CARRY_A[round_num]
    }
    fn get_invalid_carry_e(round_num: usize) -> &'static [u32] {
        &SHA256_INVALID_CARRY_E[round_num]
    }
    fn get_k() -> &'static [u32] {
        &SHA256_K
    }
    fn get_h() -> &'static [u32] {
        &SHA256_H
    }
}

pub const SHA256_INVALID_CARRY_A: [[u32; Sha256Config::WORD_U16S]; Sha256Config::ROUNDS_PER_ROW] = [
    [1230919683, 1162494304],
    [266373122, 1282901987],
    [1519718403, 1008990871],
    [923381762, 330807052],
];
pub const SHA256_INVALID_CARRY_E: [[u32; Sha256Config::WORD_U16S]; Sha256Config::ROUNDS_PER_ROW] = [
    [204933122, 1994683449],
    [443873282, 1544639095],
    [719953922, 1888246508],
    [194580482, 1075725211],
];

/// SHA256 constant K's
pub const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
/// SHA256 initial hash values
pub const SHA256_H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

impl Sha2BlockHasherConfig for Sha512Config {
    // ==== Do not change these constants! ====
    const VARIANT: Sha2Variant = Sha2Variant::Sha512;
    type Word = u64;
    /// Number of bits in a SHA512 word
    const WORD_BITS: usize = 64;
    /// Number of words in a SHA512 block
    const BLOCK_WORDS: usize = 16;
    /// Number of rows per block
    const ROWS_PER_BLOCK: usize = 21;
    /// Number of rounds per row
    const ROUNDS_PER_ROW: usize = 4;
    /// Number of rounds per block
    const ROUNDS_PER_BLOCK: usize = 80;
    /// Number of words in a SHA512 hash
    const HASH_WORDS: usize = 8;
    /// Number of vars needed to encode the row index with [Encoder]
    const ROW_VAR_CNT: usize = 6;

    fn get_invalid_carry_a(round_num: usize) -> &'static [u32] {
        &SHA512_INVALID_CARRY_A[round_num]
    }
    fn get_invalid_carry_e(round_num: usize) -> &'static [u32] {
        &SHA512_INVALID_CARRY_E[round_num]
    }
    fn get_k() -> &'static [u64] {
        &SHA512_K
    }
    fn get_h() -> &'static [u64] {
        &SHA512_H
    }
}

pub(crate) const SHA512_INVALID_CARRY_A: [[u32; Sha512Config::WORD_U16S];
    Sha512Config::ROUNDS_PER_ROW] = [
    [55971842, 827997017, 993005918, 512731953],
    [227512322, 1697529235, 1936430385, 940122990],
    [1939875843, 1173318562, 826201586, 1513494849],
    [891955202, 1732283693, 1736658755, 223514501],
];

pub(crate) const SHA512_INVALID_CARRY_E: [[u32; Sha512Config::WORD_U16S];
    Sha512Config::ROUNDS_PER_ROW] = [
    [1384427522, 1509509767, 153131516, 102514978],
    [1527552003, 1041677071, 837289497, 843522538],
    [775188482, 1620184630, 744892564, 892058728],
    [1801267202, 1393118048, 1846108940, 830635531],
];

/// SHA512 constant K's
pub const SHA512_K: [u64; 80] = [
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
];
/// SHA512 initial hash values
pub const SHA512_H: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

impl Sha2BlockHasherConfig for Sha384Config {
    // ==== Do not change these constants! ====
    const VARIANT: Sha2Variant = Sha2Variant::Sha384;
    type Word = <Sha512Config as Sha2BlockHasherConfig>::Word;
    /// Number of bits in a SHA384 word
    const WORD_BITS: usize = <Sha512Config as Sha2BlockHasherConfig>::WORD_BITS;
    /// Number of words in a SHA384 block
    const BLOCK_WORDS: usize = <Sha512Config as Sha2BlockHasherConfig>::BLOCK_WORDS;
    /// Number of rows per block
    const ROWS_PER_BLOCK: usize = <Sha512Config as Sha2BlockHasherConfig>::ROWS_PER_BLOCK;
    /// Number of rounds per row
    const ROUNDS_PER_ROW: usize = <Sha512Config as Sha2BlockHasherConfig>::ROUNDS_PER_ROW;
    /// Number of rounds per block
    const ROUNDS_PER_BLOCK: usize = <Sha512Config as Sha2BlockHasherConfig>::ROUNDS_PER_BLOCK;
    /// Number of words in a SHA384 hash
    const HASH_WORDS: usize = <Sha512Config as Sha2BlockHasherConfig>::HASH_WORDS;
    /// Number of vars needed to encode the row index with [Encoder]
    const ROW_VAR_CNT: usize = <Sha512Config as Sha2BlockHasherConfig>::ROW_VAR_CNT;

    fn get_invalid_carry_a(round_num: usize) -> &'static [u32] {
        &SHA384_INVALID_CARRY_A[round_num]
    }
    fn get_invalid_carry_e(round_num: usize) -> &'static [u32] {
        &SHA384_INVALID_CARRY_E[round_num]
    }
    fn get_k() -> &'static [u64] {
        &SHA384_K
    }
    fn get_h() -> &'static [u64] {
        &SHA384_H
    }
}

pub(crate) const SHA384_INVALID_CARRY_A: [[u32; Sha384Config::WORD_U16S];
    Sha384Config::ROUNDS_PER_ROW] = [
    [1571481603, 1428841901, 1050676523, 793575075],
    [1233315842, 1822329223, 112923808, 1874228927],
    [1245603842, 927240770, 1579759431, 70557227],
    [195532801, 594312107, 1429379950, 220407092],
];

pub(crate) const SHA384_INVALID_CARRY_E: [[u32; Sha384Config::WORD_U16S];
    Sha384Config::ROUNDS_PER_ROW] = [
    [1067980802, 1508061099, 1418826213, 1232569491],
    [1453086722, 1702524575, 152427899, 238512408],
    [1623674882, 701393097, 1002035664, 4776891],
    [1888911362, 184963225, 1151849224, 1034237098],
];

/// SHA384 constant K's
pub const SHA384_K: [u64; 80] = SHA512_K;

/// SHA384 initial hash values
pub const SHA384_H: [u64; 8] = [
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4,
];

// Needed to avoid compile errors in utils.rs
// not sure why this doesn't inf loop
pub trait RotateRight {
    fn rotate_right(self, n: u32) -> Self;
}
impl RotateRight for u32 {
    fn rotate_right(self, n: u32) -> Self {
        self.rotate_right(n)
    }
}
impl RotateRight for u64 {
    fn rotate_right(self, n: u32) -> Self {
        self.rotate_right(n)
    }
}
pub trait WrappingAdd {
    fn wrapping_add(self, n: Self) -> Self;
}
impl WrappingAdd for u32 {
    fn wrapping_add(self, n: u32) -> Self {
        self.wrapping_add(n)
    }
}
impl WrappingAdd for u64 {
    fn wrapping_add(self, n: u64) -> Self {
        self.wrapping_add(n)
    }
}
