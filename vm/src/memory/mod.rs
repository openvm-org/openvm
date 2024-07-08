use p3_field::PrimeField64;
use std::array::from_fn;

pub mod offline_checker;
#[cfg(test)]
pub mod tests;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccess<const WORD_SIZE: usize, F> {
    pub timestamp: usize,
    pub op_type: OpType,
    pub address_space: F,
    pub address: F,
    pub data: [F; WORD_SIZE],
}

// panics if the word is not equal to decompose(elem) for some elem: F
pub fn compose<const WORD_SIZE: usize, F: PrimeField64>(word: [F; WORD_SIZE]) -> F {
    for &cell in word.iter().skip(1) {
        assert_eq!(cell, F::zero());
    }
    word[0]
}

pub fn decompose<const WORD_SIZE: usize, F: PrimeField64>(field_elem: F) -> [F; WORD_SIZE] {
    from_fn(|i| if i == 0 { field_elem } else { F::zero() })
}
