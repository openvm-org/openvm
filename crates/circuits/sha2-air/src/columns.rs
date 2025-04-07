//! WARNING: the order of fields in the structs is important, do not change it

use openvm_circuit_primitives::utils::not;
use openvm_sha_macros::ColsRef;
use openvm_stark_backend::p3_field::FieldAlgebra;

use crate::Sha2Config;

/// In each SHA block:
/// - First C::ROUND_ROWS rows use ShaRoundCols
/// - Final row uses ShaDigestCols
///
/// ShaRoundCols and ShaDigestCols share the same first 3 fields:
/// - flags
/// - work_vars/hash (same type, different name)
/// - schedule_helper
///
/// This design allows for:
/// 1. Common constraints to work on either struct type by accessing these shared fields
/// 2. Specific constraints to use the appropriate struct, with flags helping to do conditional constraints
#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct ShaRoundCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
> {
    pub flags: Sha2FlagsCols<T, ROW_VAR_CNT>,
    pub work_vars: ShaWorkVarsCols<T, WORD_BITS, ROUNDS_PER_ROW, WORD_U16S>,
    pub schedule_helper:
        Sha2MessageHelperCols<T, WORD_U16S, ROUNDS_PER_ROW, ROUNDS_PER_ROW_MINUS_ONE>,
    pub message_schedule: ShaMessageScheduleCols<T, WORD_BITS, ROUNDS_PER_ROW, WORD_U8S>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct ShaDigestCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const HASH_WORDS: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
> {
    pub flags: Sha2FlagsCols<T, ROW_VAR_CNT>,
    /// Will serve as previous hash values for the next block
    pub hash: ShaWorkVarsCols<T, WORD_BITS, ROUNDS_PER_ROW, WORD_U16S>,
    pub schedule_helper:
        Sha2MessageHelperCols<T, WORD_U16S, ROUNDS_PER_ROW, ROUNDS_PER_ROW_MINUS_ONE>,
    /// The actual final hash values of the given block
    /// Note: the above `hash` will be equal to `final_hash` unless we are on the last block
    pub final_hash: [[T; WORD_U8S]; HASH_WORDS],
    /// The final hash of the previous block
    /// Note: will be constrained using interactions with the chip itself
    pub prev_hash: [[T; WORD_U16S]; HASH_WORDS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct ShaMessageScheduleCols<
    T,
    const WORD_BITS: usize,
    const ROUNDS_PER_ROW: usize,
    const WORD_U8S: usize,
> {
    /// The message schedule words as 32-bit integers
    pub w: [[T; WORD_BITS]; ROUNDS_PER_ROW],
    /// Will be message schedule carries for rows 4..C::ROUND_ROWS and a buffer for rows 0..4 to be used freely by wrapper chips
    /// Note: carries are represented as 2 bit numbers
    pub carry_or_buffer: [[T; WORD_U8S]; ROUNDS_PER_ROW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct ShaWorkVarsCols<
    T,
    const WORD_BITS: usize,
    const ROUNDS_PER_ROW: usize,
    const WORD_U16S: usize,
> {
    /// `a` and `e` after each iteration as 32-bits
    pub a: [[T; WORD_BITS]; ROUNDS_PER_ROW],
    pub e: [[T; WORD_BITS]; ROUNDS_PER_ROW],
    /// The carry's used for addition during each iteration when computing `a` and `e`
    pub carry_a: [[T; WORD_U16S]; ROUNDS_PER_ROW],
    pub carry_e: [[T; WORD_U16S]; ROUNDS_PER_ROW],
}

/// These are the columns that are used to help with the message schedule additions
/// Note: these need to be correctly assigned for every row even on padding rows
#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct Sha2MessageHelperCols<
    T,
    const WORD_U16S: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
> {
    /// The following are used to move data forward to constrain the message schedule additions
    /// The value of `w` from 3 rounds ago
    pub w_3: [[T; WORD_U16S]; ROUNDS_PER_ROW_MINUS_ONE],
    /// Here intermediate(i) =  w_i + sig_0(w_{i+1})
    /// Intermed_t represents the intermediate t rounds ago
    pub intermed_4: [[T; WORD_U16S]; ROUNDS_PER_ROW],
    pub intermed_8: [[T; WORD_U16S]; ROUNDS_PER_ROW],
    pub intermed_12: [[T; WORD_U16S]; ROUNDS_PER_ROW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2Config)]
pub struct Sha2FlagsCols<T, const ROW_VAR_CNT: usize> {
    pub is_round_row: T,
    /// A flag that indicates if the current row is among the first 4 rows of a block (the message rows)
    pub is_first_4_rows: T,
    pub is_digest_row: T,
    pub is_last_block: T,
    /// We will encode the row index [0..17) using 5 cells
    pub row_idx: [T; ROW_VAR_CNT],
    /// The global index of the current block
    pub global_block_idx: T,
    /// Will store the index of the current block in the current message starting from 0
    pub local_block_idx: T,
}

impl<O, T: Copy + core::ops::Add<Output = O>, const ROW_VAR_CNT: usize>
    Sha2FlagsCols<T, ROW_VAR_CNT>
{
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

impl<O, T: Copy + core::ops::Add<Output = O>> Sha2FlagsColsRef<'_, T> {
    pub fn is_not_padding_row(&self) -> O {
        *self.is_round_row + *self.is_digest_row
    }

    pub fn is_padding_row(&self) -> O
    where
        O: FieldAlgebra,
    {
        not(self.is_not_padding_row())
    }
}
