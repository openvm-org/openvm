//! WARNING: the order of fields in the structs is important, do not change it

use openvm_circuit_primitives::{utils::not, AlignedBorrow};
use openvm_stark_backend::p3_field::FieldAlgebra;

use super::{
    SHA256_HASH_WORDS, SHA256_ROUNDS_PER_ROW, SHA256_ROW_VAR_CNT, SHA256_WORD_BITS,
    SHA256_WORD_U16S, SHA256_WORD_U8S,
};

/// In each SHA256 block:
/// - First 16 rows use Sha256RoundCols
/// - Final row uses Sha256DigestCols
///
/// Sha256RoundCols and Sha256DigestCols share the same first 3 fields:
/// - flags
/// - work_vars/hash (same type, different name)
/// - schedule_helper
///
/// This design allows for:
/// 1. Common constraints to work on either struct type by accessing these shared fields
/// 2. Specific constraints to use the appropriate struct, with flags helping to do conditional constraints
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256RoundCols<T> {
    pub flags: Sha256FlagsCols<T>,
    pub work_vars: Sha256WorkVarsCols<T>,
    pub schedule_helper: Sha256MessageHelperCols<T>,
    pub message_schedule: Sha256MessageScheduleCols<T>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256DigestCols<T> {
    pub flags: Sha256FlagsCols<T>,
    /// Will serve as previous hash values for the next block
    pub hash: Sha256WorkVarsCols<T>,
    pub schedule_helper: Sha256MessageHelperCols<T>,
    /// The actual final hash values of the given block
    /// Note: the above `hash` will be equal to `final_hash` unless we are on the last block
    pub final_hash: [[T; SHA256_WORD_U8S]; SHA256_HASH_WORDS],
    /// The final hash of the previous block
    /// Note: will be constrained using interactions with the chip itself
    pub prev_hash: [[T; SHA256_WORD_U16S]; SHA256_HASH_WORDS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256MessageScheduleCols<T> {
    /// The message schedule words as 32-bit integers
    pub w: [[T; SHA256_WORD_BITS]; SHA256_ROUNDS_PER_ROW],
    /// Will be message schedule carries for rows 4..16 and a buffer for rows 0..4 to be used freely by wrapper chips
    /// Note: carries are represented as 2 bit numbers
    pub carry_or_buffer: [[T; SHA256_WORD_U8S]; SHA256_ROUNDS_PER_ROW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256WorkVarsCols<T> {
    /// `a` and `e` after each iteration as 32-bits
    pub a: [[T; SHA256_WORD_BITS]; SHA256_ROUNDS_PER_ROW],
    pub e: [[T; SHA256_WORD_BITS]; SHA256_ROUNDS_PER_ROW],
    /// The carry's used for addition during each iteration when computing `a` and `e`
    pub carry_a: [[T; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW],
    pub carry_e: [[T; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW],
}

/// These are the columns that are used to help with the message schedule additions
/// Note: these need to be correctly assigned for every row even on padding rows
#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256MessageHelperCols<T> {
    /// The following are used to move data forward to constrain the message schedule additions
    /// The value of `w` from 3 rounds ago
    pub w_3: [[T; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW - 1],
    /// Here intermediate(i) =  w_i + sig_0(w_{i+1})
    /// Intermed_t represents the intermediate t rounds ago
    pub intermed_4: [[T; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW],
    pub intermed_8: [[T; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW],
    pub intermed_12: [[T; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct Sha256FlagsCols<T> {
    pub is_round_row: T,
    /// A flag that indicates if the current row is among the first 4 rows of a block
    pub is_first_4_rows: T,
    pub is_digest_row: T,
    pub is_last_block: T,
    /// We will encode the row index [0..17) using 5 cells
    pub row_idx: [T; SHA256_ROW_VAR_CNT],
    /// The global index of the current block
    pub global_block_idx: T,
    /// Will store the index of the current block in the current message starting from 0
    pub local_block_idx: T,
}

impl<O, T: Copy + core::ops::Add<Output = O>> Sha256FlagsCols<T> {
    pub fn is_not_padding_row(&self) -> O {
        self.is_round_row + self.is_digest_row
    }

    pub fn is_padding_row(&self) -> O
    where
        O: FieldAlgebra,
    {
        not(self.is_not_padding_row())
    }
}
