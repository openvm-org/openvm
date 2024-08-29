use std::{array, iter};

use derive_new::new;
use p3_air::AirBuilder;

#[derive(Clone, Debug, PartialEq, Eq, new)]
pub struct MemoryOperation<const N: usize, T> {
    pub addr_space: T,
    pub pointer: T,
    pub timestamp: T,
    pub data: [T; N],
    pub enabled: T,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryOperation<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            addr_space: slc[0].clone(),
            pointer: slc[1].clone(),
            timestamp: slc[2].clone(),
            enabled: slc[3].clone(),
            data: array::from_fn(|i| slc[4 + i].clone()),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryOperation<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        iter::once(self.addr_space)
            .chain(iter::once(self.pointer))
            .chain(iter::once(self.timestamp))
            .chain(iter::once(self.enabled))
            .chain(self.data)
            .collect()
    }

    pub fn width() -> usize {
        4 + WORD_SIZE
    }
}

impl<const WORD_SIZE: usize, T> MemoryOperation<WORD_SIZE, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> MemoryOperation<WORD_SIZE, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        MemoryOperation {
            addr_space: self.addr_space.into(),
            pointer: self.pointer.into(),
            timestamp: self.timestamp.into(),
            data: self.data.map(Into::into),
            enabled: self.enabled.into(),
        }
    }
}

impl<const N: usize, T: Default> Default for MemoryOperation<N, T> {
    fn default() -> Self {
        Self {
            addr_space: T::default(),
            pointer: T::default(),
            timestamp: T::default(),
            data: array::from_fn(|_| T::default()),
            enabled: T::default(),
        }
    }
}
