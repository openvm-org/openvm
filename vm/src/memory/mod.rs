use p3_field::PrimeField64;

pub mod audit;
// pub mod expand;
// pub mod expand_interface;
pub mod manager;
pub mod offline_checker;
#[cfg(test)]
pub mod tests;
pub mod tree;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

/// The full pointer to a location in memory consists of an address space and a pointer within
/// the address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryAddress<S, T> {
    pub address_space: S,
    pub pointer: T,
}

impl<S, T> MemoryAddress<S, T> {
    pub fn new(address_space: S, pointer: T) -> Self {
        Self {
            address_space,
            pointer,
        }
    }

    pub fn from<T1, T2>(a: MemoryAddress<T1, T2>) -> Self
    where
        T1: Into<S>,
        T2: Into<T>,
    {
        Self {
            address_space: a.address_space.into(),
            pointer: a.pointer.into(),
        }
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
    std::array::from_fn(|i| if i == 0 { field_elem } else { F::zero() })
}
