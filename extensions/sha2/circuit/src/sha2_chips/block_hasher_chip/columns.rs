use openvm_circuit_primitives_derive::ColsRef;
use openvm_sha2_air::{
    Sha2BlockHasherSubairConfig, Sha2DigestCols, Sha2DigestColsRef, Sha2DigestColsRefMut,
    Sha2RoundCols, Sha2RoundColsRef, Sha2RoundColsRefMut,
};

// offset in the columns struct where the inner column start
pub const INNER_OFFSET: usize = 1;

// Just adding request_id to both columns structs
#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2BlockHasherSubairConfig)]
pub struct Sha2BlockHasherRoundCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
> {
    pub request_id: T,
    pub inner: Sha2RoundCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
    >,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
#[config(Sha2BlockHasherSubairConfig)]
pub struct Sha2BlockHasherDigestCols<
    T,
    const WORD_BITS: usize,
    const WORD_U8S: usize,
    const WORD_U16S: usize,
    const HASH_WORDS: usize,
    const ROUNDS_PER_ROW: usize,
    const ROUNDS_PER_ROW_MINUS_ONE: usize,
    const ROW_VAR_CNT: usize,
> {
    pub request_id: T,
    pub inner: Sha2DigestCols<
        T,
        WORD_BITS,
        WORD_U8S,
        WORD_U16S,
        HASH_WORDS,
        ROUNDS_PER_ROW,
        ROUNDS_PER_ROW_MINUS_ONE,
        ROW_VAR_CNT,
    >,
}
