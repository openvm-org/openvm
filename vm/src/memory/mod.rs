use afs_primitives::offline_checker::OfflineCheckerOperation;
use p3_field::PrimeField64;
use std::array::from_fn;

pub mod expand;
pub mod offline_checker;
#[cfg(test)]
pub mod tests;
pub mod tree;

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

impl<const WORD_SIZE: usize, F: PrimeField64> OfflineCheckerOperation<F>
    for MemoryAccess<WORD_SIZE, F>
{
    fn get_timestamp(&self) -> usize {
        self.timestamp
    }

    fn get_idx(&self) -> Vec<F> {
        vec![self.address_space, self.address]
    }

    fn get_data(&self) -> Vec<F> {
        self.data.to_vec()
    }
    fn get_op_type(&self) -> u8 {
        self.op_type as u8
    }
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
