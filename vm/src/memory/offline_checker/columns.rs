//! Defines auxiliary columns for memory operations: `MemoryReadAuxCols`,
//! `MemoryReadWithImmediateAuxCols`, and `MemoryWriteAuxCols`.

use std::{array, borrow::Borrow, iter, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_primitives::assert_less_than::columns::AssertLessThanAuxCols;
use p3_field::AbstractField;

use crate::memory::offline_checker::bridge::AUX_LEN;

// repr(C) is needed to make sure that the compiler does not reorder the fields
// we assume the order of the fields when using borrow or borrow_mut
#[repr(C)]
/// Base structure for auxiliary memory columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow)]
pub(super) struct MemoryBaseAuxCols<T, const N: usize> {
    // TODO[zach]: Should be just prev_timestamp: T.
    /// The previous timestamps in which the cells were accessed.
    pub(super) prev_timestamps: [T; N],
    // TODO[zach]: Should be just clk_lt_aux: IsLessThanAuxCols<T>.
    /// The auxiliary columns to perform the less than check.
    pub(super) clk_lt_aux: [AssertLessThanAuxCols<T, AUX_LEN>; N],
}

impl<const N: usize, T: Clone> MemoryBaseAuxCols<T, N> {
    /// TODO[arayi]: Since we have AlignedBorrow, should remove all from_slice, from_iterator, and flatten in a future PR.
    pub fn from_slice(slc: &[T]) -> Self {
        let base_aux_cols: &MemoryBaseAuxCols<T, N> = slc.borrow();
        base_aux_cols.clone()
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let sm = iter.take(Self::width()).collect::<Vec<T>>();
        let base_aux_cols: &MemoryBaseAuxCols<T, N> = sm[..].borrow();
        base_aux_cols.clone()
    }
}

impl<const N: usize, T> MemoryBaseAuxCols<T, N> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.prev_timestamps)
            .chain(
                self.clk_lt_aux
                    .into_iter()
                    .flat_map(|x| x.lower_decomp),
            )
            .collect()
    }

    pub const fn width() -> usize {
        size_of::<MemoryBaseAuxCols<u8, N>>()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryWriteAuxCols<const N: usize, T> {
    pub(super) base: MemoryBaseAuxCols<T, N>,
    pub(super) prev_data: [T; N],
}

impl<const N: usize, T> MemoryWriteAuxCols<N, T> {
    pub fn new(
        prev_data: [T; N],
        prev_timestamps: [T; N],
        clk_lt_aux: [AssertLessThanAuxCols<T, AUX_LEN>; N],
    ) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamps,
                clk_lt_aux,
            },
            prev_data,
        }
    }
}

impl<const N: usize, T: Clone> MemoryWriteAuxCols<N, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let width = MemoryBaseAuxCols::<T, N>::width();
        Self {
            base: MemoryBaseAuxCols::from_slice(&slc[..width]),
            prev_data: array::from_fn(|i| slc[width + i].clone()),
        }
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_iterator(iter),
            prev_data: array::from_fn(|_| iter.next().unwrap()),
        }
    }
}

impl<const N: usize, T> MemoryWriteAuxCols<N, T> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.base.flatten())
            .chain(self.prev_data)
            .collect()
    }

    pub const fn width() -> usize {
        size_of::<MemoryWriteAuxCols<N, u8>>()
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryWriteAuxCols<N, F> {
    pub fn disabled() -> Self {
        let width = MemoryWriteAuxCols::<N, F>::width();
        MemoryWriteAuxCols::from_slice(&vec![F::zero(); width])
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryReadAuxCols<const N: usize, T> {
    pub(super) base: MemoryBaseAuxCols<T, N>,
}

impl<const N: usize, T> MemoryReadAuxCols<N, T> {
    pub fn new(
        prev_timestamps: [T; N],
        clk_lt_aux: [AssertLessThanAuxCols<T, AUX_LEN>; N],
    ) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamps,
                clk_lt_aux,
            },
        }
    }
}

impl<const N: usize, T: Clone> MemoryReadAuxCols<N, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_slice(slc),
        }
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_iterator(iter),
        }
    }
}

impl<const N: usize, T> MemoryReadAuxCols<N, T> {
    pub fn flatten(self) -> Vec<T> {
        self.base.flatten()
    }

    pub const fn width() -> usize {
        size_of::<MemoryReadAuxCols<N, u8>>()
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryReadAuxCols<N, F> {
    pub fn disabled() -> Self {
        let width = MemoryReadAuxCols::<N, F>::width();
        MemoryReadAuxCols::from_slice(&vec![F::zero(); width])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryReadOrImmediateAuxCols<T> {
    pub(super) base: MemoryBaseAuxCols<T, 1>,
    pub(super) is_immediate: T,
    pub(super) is_zero_aux: T,
}

impl<T> MemoryReadOrImmediateAuxCols<T> {
    pub fn new(
        prev_timestamp: T,
        is_immediate: T,
        is_zero_aux: T,
        clk_lt_aux: AssertLessThanAuxCols<T, AUX_LEN>,
    ) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamps: [prev_timestamp],
                clk_lt_aux: [clk_lt_aux],
            },
            is_immediate,
            is_zero_aux,
        }
    }
}

impl<T: Clone> MemoryReadOrImmediateAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let width = MemoryBaseAuxCols::<T, 1>::width();
        Self {
            base: MemoryBaseAuxCols::from_slice(&slc[..width]),
            is_immediate: slc[width].clone(),
            is_zero_aux: slc[width + 1].clone(),
        }
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_iterator(iter),
            is_immediate: iter.next().unwrap(),
            is_zero_aux: iter.next().unwrap(),
        }
    }
}

impl<T> MemoryReadOrImmediateAuxCols<T> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.base.flatten())
            .chain(iter::once(self.is_immediate))
            .chain(iter::once(self.is_zero_aux))
            .collect()
    }

    pub const fn width() -> usize {
        size_of::<MemoryReadOrImmediateAuxCols<u8>>()
    }
}

impl<F: AbstractField + Copy> MemoryReadOrImmediateAuxCols<F> {
    pub fn disabled() -> Self {
        let width = MemoryReadOrImmediateAuxCols::<F>::width();
        MemoryReadOrImmediateAuxCols::from_slice(&vec![F::zero(); width])
    }
}
