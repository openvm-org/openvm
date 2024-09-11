//! Defines auxiliary columns for memory operations: `MemoryReadAuxCols`,
//! `MemoryReadWithImmediateAuxCols`, and `MemoryWriteAuxCols`.

use std::{array, iter};

use afs_primitives::is_less_than::{columns::IsLessThanAuxCols, IsLessThanAir};
use p3_field::AbstractField;

use super::bridge::MemoryOfflineChecker;

/// Base structure for auxiliary memory columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MemoryBaseAuxCols<T, const N: usize> {
    // TODO[zach]: Should be just prev_timestamp: T.
    /// The previous timestamps in which the cells were accessed.
    pub(super) prev_timestamps: [T; N],
    // TODO[jpw]: IsLessThan should be optimized to AssertLessThan
    // TODO[zach]: Should be just clk_lt_aux: IsLessThanAuxCols<T>.
    /// The auxiliary columns to perform the less than check.
    pub(super) clk_lt_aux: [IsLessThanAuxCols<T>; N],
}

impl<const N: usize, T: Clone> MemoryBaseAuxCols<T, N> {
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        Self {
            prev_timestamps: array::from_fn(|i| slc[i].clone()),
            clk_lt_aux: {
                let lt_width = IsLessThanAuxCols::<T>::width(&oc.timestamp_lt_air);
                let mut pos = N;
                array::from_fn(|_| {
                    pos += lt_width;
                    IsLessThanAuxCols::from_slice(&slc[pos - lt_width..pos])
                })
            },
        }
    }
}

impl<const N: usize, T> MemoryBaseAuxCols<T, N> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.prev_timestamps)
            .chain(self.clk_lt_aux.into_iter().flat_map(|x| x.flatten()))
            .collect()
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            prev_timestamps: array::from_fn(|_| iter.next().unwrap()),
            clk_lt_aux: array::from_fn(|_| IsLessThanAuxCols::from_iterator(iter, lt_air)),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        N + N * IsLessThanAuxCols::<T>::width(&oc.timestamp_lt_air)
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
        clk_lt_aux: [IsLessThanAuxCols<T>; N],
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
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        let width = MemoryBaseAuxCols::<T, N>::width(oc);
        Self {
            base: MemoryBaseAuxCols::from_slice(&slc[..width], oc),
            prev_data: array::from_fn(|i| slc[width + i].clone()),
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

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_iterator(iter, lt_air),
            prev_data: array::from_fn(|_| iter.next().unwrap()),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        MemoryBaseAuxCols::<T, N>::width(oc) + N
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryWriteAuxCols<N, F> {
    pub fn disabled(mem_oc: MemoryOfflineChecker) -> Self {
        let width = MemoryWriteAuxCols::<N, F>::width(&mem_oc);
        MemoryWriteAuxCols::from_slice(&vec![F::zero(); width], &mem_oc)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryReadAuxCols<const N: usize, T> {
    pub(super) base: MemoryBaseAuxCols<T, N>,
}

impl<const N: usize, T> MemoryReadAuxCols<N, T> {
    pub fn new(prev_timestamps: [T; N], clk_lt_aux: [IsLessThanAuxCols<T>; N]) -> Self {
        Self {
            base: MemoryBaseAuxCols {
                prev_timestamps,
                clk_lt_aux,
            },
        }
    }
}

impl<const N: usize, T: Clone> MemoryReadAuxCols<N, T> {
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_slice(slc, oc),
        }
    }
}

impl<const N: usize, T> MemoryReadAuxCols<N, T> {
    pub fn flatten(self) -> Vec<T> {
        self.base.flatten()
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_iterator(iter, lt_air),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        MemoryBaseAuxCols::<T, N>::width(oc)
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryReadAuxCols<N, F> {
    pub fn disabled(mem_oc: MemoryOfflineChecker) -> Self {
        let width = MemoryReadAuxCols::<N, F>::width(&mem_oc);
        MemoryReadAuxCols::from_slice(&vec![F::zero(); width], &mem_oc)
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
        clk_lt_aux: IsLessThanAuxCols<T>,
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
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        let width = MemoryBaseAuxCols::<T, 1>::width(oc);
        Self {
            base: MemoryBaseAuxCols::from_slice(&slc[..width], oc),
            is_immediate: slc[width].clone(),
            is_zero_aux: slc[width + 1].clone(),
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

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            base: MemoryBaseAuxCols::from_iterator(iter, lt_air),
            is_immediate: iter.next().unwrap(),
            is_zero_aux: iter.next().unwrap(),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        MemoryBaseAuxCols::<T, 1>::width(oc) + 2
    }
}

impl<F: AbstractField + Copy> MemoryReadOrImmediateAuxCols<F> {
    pub fn disabled(mem_oc: MemoryOfflineChecker) -> Self {
        let width = MemoryReadOrImmediateAuxCols::<F>::width(&mem_oc);
        MemoryReadOrImmediateAuxCols::from_slice(&vec![F::zero(); width], &mem_oc)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::memory::offline_checker::MemoryBus;

    #[test]
    fn test_write_aux_cols_width() {
        type F = BabyBear;

        let mem_oc = MemoryOfflineChecker::new(MemoryBus(1), 29, 16);

        let disabled = MemoryWriteAuxCols::<1, F>::disabled(mem_oc);
        assert_eq!(
            disabled.flatten().len(),
            MemoryWriteAuxCols::<1, F>::width(&mem_oc)
        );

        let disabled = MemoryWriteAuxCols::<4, F>::disabled(mem_oc);
        assert_eq!(
            disabled.flatten().len(),
            MemoryWriteAuxCols::<4, F>::width(&mem_oc)
        );
    }

    #[test]
    fn test_read_aux_cols_width() {
        type F = BabyBear;

        let mem_oc = MemoryOfflineChecker::new(MemoryBus(1), 29, 16);

        let disabled = MemoryReadAuxCols::<1, F>::disabled(mem_oc);
        assert_eq!(
            disabled.flatten().len(),
            MemoryReadAuxCols::<1, F>::width(&mem_oc)
        );

        let disabled = MemoryReadAuxCols::<4, F>::disabled(mem_oc);
        assert_eq!(
            disabled.flatten().len(),
            MemoryReadAuxCols::<4, F>::width(&mem_oc)
        );
    }

    #[test]
    fn test_read_or_immediate_aux_cols_width() {
        type F = BabyBear;

        let mem_oc = MemoryOfflineChecker::new(MemoryBus(1), 29, 16);

        let disabled = MemoryReadOrImmediateAuxCols::<F>::disabled(mem_oc);
        assert_eq!(
            disabled.flatten().len(),
            MemoryReadOrImmediateAuxCols::<F>::width(&mem_oc)
        );
    }
}
