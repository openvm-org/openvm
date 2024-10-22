//! Defines auxiliary columns for memory operations: `MemoryReadAuxCols`,
//! `MemoryReadWithImmediateAuxCols`, and `MemoryWriteAuxCols`.

use std::{array, borrow::Borrow, iter};

use afs_derive::AlignedBorrow;
use afs_primitives::is_less_than::LessThanAuxCols;
use p3_field::{AbstractField, PrimeField32};

use crate::system::memory::offline_checker::bridge::AUX_LEN;

// repr(C) is needed to make sure that the compiler does not reorder the fields
// we assume the order of the fields when using borrow or borrow_mut
#[repr(C)]
/// Base structure for auxiliary memory columns.
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct MemoryBaseAuxCols<T> {
    /// The previous timestamps in which the cells were accessed.
    pub(super) prev_timestamp: T,
    /// The auxiliary columns to perform the less than check.
    pub(super) clk_lt_aux: LessThanAuxCols<T, AUX_LEN>,
}

impl<T: Clone> MemoryBaseAuxCols<T> {
    /// TODO[arayi]: Since we have AlignedBorrow, should remove all from_slice, from_iterator, and flatten in a future PR.
    pub fn from_slice(slc: &[T]) -> Self {
        let base_aux_cols: &MemoryBaseAuxCols<T> = slc.borrow();
        base_aux_cols.clone()
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let sm = iter.take(Self::width()).collect::<Vec<T>>();
        let base_aux_cols: &MemoryBaseAuxCols<T> = sm[..].borrow();
        base_aux_cols.clone()
    }
}

impl<T> MemoryBaseAuxCols<T> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(iter::once(self.prev_timestamp))
            .chain(self.clk_lt_aux.lower_decomp)
            .collect()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct MemoryWriteAuxCols<T, const N: usize> {
    pub(super) base: MemoryBaseAuxCols<T>,
    pub(super) prev_data: [T; N],
}

impl<const N: usize, T> MemoryWriteAuxCols<T, N> {
    pub fn new(prev_data: [T; N], prev_timestamp: T, lt_aux: LessThanAuxCols<T, AUX_LEN>) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp,
                clk_lt_aux: lt_aux,
            },
            prev_data,
        }
    }
}

impl<const N: usize, T: Clone> MemoryWriteAuxCols<T, N> {
    pub fn from_slice(slc: &[T]) -> Self {
        let width = MemoryBaseAuxCols::<T>::width();
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

    pub fn from_base(base: MemoryBaseAuxCols<T>, prev_data: [T; N]) -> Self {
        Self { base, prev_data }
    }

    pub fn get_base(self) -> MemoryBaseAuxCols<T> {
        self.base
    }
}

impl<const N: usize, T> MemoryWriteAuxCols<T, N> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.base.flatten())
            .chain(self.prev_data)
            .collect()
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryWriteAuxCols<F, N> {
    pub fn disabled() -> Self {
        let width = MemoryWriteAuxCols::<F, N>::width();
        MemoryWriteAuxCols::from_slice(&vec![F::zero(); width])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct MemoryReadAuxCols<T, const N: usize> {
    pub(super) base: MemoryBaseAuxCols<T>,
}

impl<const N: usize, F: PrimeField32> MemoryReadAuxCols<F, N> {
    pub fn new(prev_timestamp: u32, clk_lt_aux: LessThanAuxCols<F, AUX_LEN>) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp: F::from_canonical_u32(prev_timestamp),
                clk_lt_aux,
            },
        }
    }
}

impl<const N: usize, T: Clone> MemoryReadAuxCols<T, N> {
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

impl<const N: usize, T> MemoryReadAuxCols<T, N> {
    pub fn flatten(self) -> Vec<T> {
        self.base.flatten()
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryReadAuxCols<F, N> {
    pub fn disabled() -> Self {
        let width = MemoryReadAuxCols::<F, N>::width();
        MemoryReadAuxCols::from_slice(&vec![F::zero(); width])
    }
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct MemoryHeapReadAuxCols<T, const N: usize> {
    pub address: MemoryReadAuxCols<T, 1>,
    pub data: MemoryReadAuxCols<T, N>,
}

impl<const N: usize, T: Clone> MemoryHeapReadAuxCols<T, N> {
    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            address: MemoryReadAuxCols::from_iterator(iter),
            data: MemoryReadAuxCols::from_iterator(iter),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.address.flatten())
            .chain(self.data.flatten())
            .collect()
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryHeapReadAuxCols<F, N> {
    pub fn disabled() -> Self {
        let width = MemoryReadAuxCols::<F, 1>::width();
        let address = MemoryReadAuxCols::from_slice(&vec![F::zero(); width]);
        let width = MemoryReadAuxCols::<F, N>::width();
        let data = MemoryReadAuxCols::from_slice(&vec![F::zero(); width]);
        MemoryHeapReadAuxCols { address, data }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct MemoryHeapWriteAuxCols<T, const N: usize> {
    pub address: MemoryReadAuxCols<T, 1>,
    pub data: MemoryWriteAuxCols<T, N>,
}

impl<const N: usize, T: Clone> MemoryHeapWriteAuxCols<T, N> {
    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            address: MemoryReadAuxCols::from_iterator(iter),
            data: MemoryWriteAuxCols::from_iterator(iter),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.address.flatten())
            .chain(self.data.flatten())
            .collect()
    }

    pub const fn width() -> usize {
        MemoryReadAuxCols::<T, 1>::width() + MemoryWriteAuxCols::<T, N>::width()
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryHeapWriteAuxCols<F, N> {
    pub fn disabled() -> Self {
        let width = MemoryReadAuxCols::<F, 1>::width();
        let address = MemoryReadAuxCols::from_slice(&vec![F::zero(); width]);
        let width = MemoryWriteAuxCols::<F, N>::width();
        let data = MemoryWriteAuxCols::from_slice(&vec![F::zero(); width]);
        MemoryHeapWriteAuxCols { address, data }
    }
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct MemoryReadOrImmediateAuxCols<T> {
    pub(super) base: MemoryBaseAuxCols<T>,
    pub(super) is_immediate: T,
    pub(super) is_zero_aux: T,
}

impl<T> MemoryReadOrImmediateAuxCols<T> {
    pub fn new(
        prev_timestamp: T,
        is_immediate: T,
        is_zero_aux: T,
        clk_lt_aux: LessThanAuxCols<T, AUX_LEN>,
    ) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamp,
                clk_lt_aux,
            },
            is_immediate,
            is_zero_aux,
        }
    }
}

impl<T: Clone> MemoryReadOrImmediateAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let width = MemoryBaseAuxCols::<T>::width();
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
}

impl<F: AbstractField + Copy> MemoryReadOrImmediateAuxCols<F> {
    pub fn disabled() -> Self {
        let width = MemoryReadOrImmediateAuxCols::<F>::width();
        MemoryReadOrImmediateAuxCols::from_slice(&vec![F::zero(); width])
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn test_write_aux_cols_width() {
        type F = BabyBear;

        let disabled = MemoryWriteAuxCols::<F, 1>::disabled();
        assert_eq!(
            disabled.flatten().len(),
            MemoryWriteAuxCols::<F, 1>::width()
        );

        let disabled = MemoryWriteAuxCols::<F, 4>::disabled();
        assert_eq!(
            disabled.flatten().len(),
            MemoryWriteAuxCols::<F, 4>::width()
        );
    }

    #[test]
    fn test_read_aux_cols_width() {
        type F = BabyBear;

        let disabled = MemoryReadAuxCols::<F, 1>::disabled();
        assert_eq!(disabled.flatten().len(), MemoryReadAuxCols::<F, 1>::width());

        let disabled = MemoryReadAuxCols::<F, 4>::disabled();
        assert_eq!(disabled.flatten().len(), MemoryReadAuxCols::<F, 4>::width());
    }

    #[test]
    fn test_read_or_immediate_aux_cols_width() {
        type F = BabyBear;

        let disabled = MemoryReadOrImmediateAuxCols::<F>::disabled();
        assert_eq!(
            disabled.flatten().len(),
            MemoryReadOrImmediateAuxCols::<F>::width()
        );
    }
}
