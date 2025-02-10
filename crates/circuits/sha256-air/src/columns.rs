//! WARNING: the order of fields in the structs is important, do not change it

use openvm_circuit_primitives::utils::not;
use openvm_sha_macros::ColsRef;
use openvm_stark_backend::p3_field::FieldAlgebra;

use crate::ShaConfig;

/// In each SHA256 block:
/// - First 16 rows use Sha256RoundCols
/// - Final row uses Sha256DigestCols
///
/// Note that for soundness, we require that there is always a padding row after the last digest row in the trace.
/// Right now, this is true because the unpadded height is a multiple of 17, and thus not a power of 2.
///
/// Sha256RoundCols and Sha256DigestCols share the same first 3 fields:
/// - flags
/// - work_vars/hash (same type, different name)
/// - schedule_helper
///
/// This design allows for:
/// 1. Common constraints to work on either struct type by accessing these shared fields
/// 2. Specific constraints to use the appropriate struct, with flags helping to do conditional constraints
///
/// Note that the `Sha256WorkVarsCols` field it is used for different purposes in the two structs.
#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
pub struct ShaRoundCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
> {
    pub flags: ShaFlagsCols<T, ROW_VAR_CNT>,
    /// Stores the current state of the working variables
    pub work_vars: ShaWorkVarsCols<T, WORD_BITS, ROUNDS_PER_ROW, WORD_U16S>,
    pub schedule_helper:
        ShaMessageHelperCols<T, WORD_U16S, ROUNDS_PER_ROW, ROUNDS_PER_ROW_MINUS_ONE>,
    pub message_schedule: ShaMessageScheduleCols<T, WORD_BITS, ROUNDS_PER_ROW, WORD_U8S>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
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
    pub flags: ShaFlagsCols<T, ROW_VAR_CNT>,
    /// Will serve as previous hash values for the next block.
    ///     - on non-last blocks, this is the final hash of the current block
    ///     - on last blocks, this is the initial state constants, SHA256_H.
    /// The work variables constraints are applied on all rows, so `carry_a` and `carry_e`
    /// must be filled in with dummy values to ensure these constraints hold.
    pub hash: ShaWorkVarsCols<T, WORD_BITS, ROUNDS_PER_ROW, WORD_U16S>,
    pub schedule_helper:
        ShaMessageHelperCols<T, WORD_U16S, ROUNDS_PER_ROW, ROUNDS_PER_ROW_MINUS_ONE>,
    /// The actual final hash values of the given block
    /// Note: the above `hash` will be equal to `final_hash` unless we are on the last block
    pub final_hash: [[T; WORD_U8S]; HASH_WORDS],
    /// The final hash of the previous block
    /// Note: will be constrained using interactions with the chip itself
    pub prev_hash: [[T; WORD_U16S]; HASH_WORDS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
pub struct ShaMessageScheduleCols<
    T,
    const WORD_BITS: usize,
    const ROUNDS_PER_ROW: usize,
    const WORD_U8S: usize,
> {
    /// The message schedule words as C::WORD_BITS-bit integers
    /// The first 16 rows will be the message data
    pub w: [[T; WORD_BITS]; ROUNDS_PER_ROW],
    /// Will be message schedule carries for rows 4..16 and a buffer for rows 0..4 to be used freely by wrapper chips
    /// Note: carries are 2 bit numbers represented using 2 cells as individual bits
    pub carry_or_buffer: [[T; WORD_U8S]; ROUNDS_PER_ROW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
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
pub struct ShaMessageHelperCols<
    T,
    const WORD_U16S: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
> {
    /// The following are used to move data forward to constrain the message schedule additions
    /// The value of `w` (message schedule word) from 3 rounds ago
    /// In general, `w_i` means `w` from `i` rounds ago
    pub w_3: [[T; WORD_U16S]; ROUNDS_PER_ROW_MINUS_ONE],
    /// Here intermediate(i) =  w_i + sig_0(w_{i+1})
    /// Intermed_t represents the intermediate t rounds ago
    /// This is needed to constrain the message schedule, since we can only constrain on two rows at a time
    pub intermed_4: [[T; WORD_U16S]; ROUNDS_PER_ROW],
    pub intermed_8: [[T; WORD_U16S]; ROUNDS_PER_ROW],
    pub intermed_12: [[T; WORD_U16S]; ROUNDS_PER_ROW],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
pub struct ShaFlagsCols<T, const ROW_VAR_CNT: usize> {
    /// A flag that indicates if the current row is among the first C::ROUND_ROWS rows of a block.
    pub is_round_row: T,
    /// A flag that indicates if the current row is among the first 4 rows of a block.
    pub is_first_4_rows: T,
    /// A flag that indicates if the current row is the last (17th) row of a block.
    pub is_digest_row: T,
    // A flag that indicates if the current row is the last block of the message.
    // This flag is only used in digest rows.
    pub is_last_block: T,
    /// We will encode the row index [0..17) using 5 cells
    //#[length(ROW_VAR_CNT)]
    pub row_idx: [T; ROW_VAR_CNT],
    /// The index of the current block in the trace starting at 1.
    /// Set to 0 on padding rows.
    pub global_block_idx: T,
    /// The index of the current block in the current message starting at 0.
    /// Resets after every message.
    /// Set to 0 on padding rows.
    pub local_block_idx: T,
}

impl<O, T: Copy + core::ops::Add<Output = O>, const ROW_VAR_CNT: usize>
    ShaFlagsCols<T, ROW_VAR_CNT>
{
    // This refers to the padding rows that are added to the air to make the trace length a power of 2.
    // Not to be confused with the padding added to messages as part of the SHA hash function.
    pub fn is_not_padding_row(&self) -> O {
        self.is_round_row + self.is_digest_row
    }

    // This refers to the padding rows that are added to the air to make the trace length a power of 2.
    // Not to be confused with the padding added to messages as part of the SHA hash function.
    pub fn is_padding_row(&self) -> O
    where
        O: FieldAlgebra,
    {
        not(self.is_not_padding_row())
    }
}

// We need to implement this for the ColsRef type as well
impl<'a, O, T: Copy + core::ops::Add<Output = O>> ShaFlagsColsRef<'a, T> {
    pub fn is_not_padding_row(&self) -> O {
        *self.is_round_row + *self.is_digest_row
    }

    // This refers to the padding rows that are added to the air to make the trace length a power of 2.
    // Not to be confused with the padding added to messages as part of the SHA hash function.
    pub fn is_padding_row(&self) -> O
    where
        O: FieldAlgebra,
    {
        not(self.is_not_padding_row())
    }
}
