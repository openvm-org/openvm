use std::{array, iter};

use afs_primitives::is_less_than::{columns::IsLessThanAuxCols, IsLessThanAir};
use p3_field::AbstractField;

use super::bridge::MemoryOfflineChecker;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BaseAuxCols<T, const N: usize> {
    // TODO[zach]: Should be just prev_timestamp: T.
    pub(super) prev_timestamps: [T; N],
    // TODO[zach]: Should be just clk_lt: T.
    pub(super) clk_lt: [T; N],
    // TODO[jpw]: IsLessThan should be optimized to AssertLessThan
    // TODO[zach]: Should be just clk_lt_aux: IsLessThanAuxCols<T>.
    pub(super) clk_lt_aux: [IsLessThanAuxCols<T>; N],
}

impl<const N: usize, T: Clone> BaseAuxCols<T, N> {
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        Self {
            prev_timestamps: array::from_fn(|i| slc[i].clone()),
            clk_lt: array::from_fn(|i| slc[N + i].clone()),
            clk_lt_aux: {
                let lt_width = IsLessThanAuxCols::<T>::width(&oc.timestamp_lt_air);
                let mut pos = 2 * N;
                array::from_fn(|_| {
                    pos += lt_width;
                    IsLessThanAuxCols::from_slice(&slc[pos - lt_width..pos])
                })
            },
        }
    }
}

impl<const N: usize, T> BaseAuxCols<T, N> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.prev_timestamps)
            .chain(self.clk_lt)
            .chain(self.clk_lt_aux.into_iter().flat_map(|x| x.flatten()))
            .collect()
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            prev_timestamps: array::from_fn(|_| iter.next().unwrap()),
            clk_lt: array::from_fn(|_| iter.next().unwrap()),
            clk_lt_aux: array::from_fn(|_| IsLessThanAuxCols::from_iterator(iter, lt_air)),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        N * (2 + IsLessThanAuxCols::<T>::width(&oc.timestamp_lt_air))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryWriteAuxCols<const N: usize, T> {
    pub(super) base: BaseAuxCols<T, N>,
    pub(super) prev_data: [T; N],
}

impl<const N: usize, T> MemoryWriteAuxCols<N, T> {
    pub fn new(
        prev_data: [T; N],
        prev_timestamps: [T; N],
        clk_lt: [T; N],
        clk_lt_aux: [IsLessThanAuxCols<T>; N],
    ) -> Self {
        Self {
            base: BaseAuxCols {
                prev_timestamps,
                clk_lt,
                clk_lt_aux,
            },
            prev_data,
        }
    }
}

impl<const N: usize, T: Clone> MemoryWriteAuxCols<N, T> {
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        let width = BaseAuxCols::<T, N>::width(oc);
        Self {
            base: BaseAuxCols::from_slice(&slc[..width], oc),
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
            base: BaseAuxCols::from_iterator(iter, lt_air),
            prev_data: array::from_fn(|_| iter.next().unwrap()),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        BaseAuxCols::<T, N>::width(oc) + N
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
    pub(super) base: BaseAuxCols<T, N>,
    pub(super) is_immediate: T,
    pub(super) is_zero_aux: T,
}

impl<const N: usize, T> MemoryReadAuxCols<N, T> {
    pub fn new(
        prev_timestamps: [T; N],
        is_immediate: T,
        is_zero_aux: T,
        clk_lt: [T; N],
        clk_lt_aux: [IsLessThanAuxCols<T>; N],
    ) -> Self {
        Self {
            base: BaseAuxCols {
                prev_timestamps,
                clk_lt,
                clk_lt_aux,
            },
            is_immediate,
            is_zero_aux,
        }
    }
}

impl<const N: usize, T: Clone> MemoryReadAuxCols<N, T> {
    pub fn from_slice(slc: &[T], oc: &MemoryOfflineChecker) -> Self {
        let width = BaseAuxCols::<T, N>::width(oc);
        Self {
            base: BaseAuxCols::from_slice(&slc[..width], oc),
            is_immediate: slc[width].clone(),
            is_zero_aux: slc[width + 1].clone(),
        }
    }
}

impl<const N: usize, T> MemoryReadAuxCols<N, T> {
    pub fn flatten(self) -> Vec<T> {
        iter::empty()
            .chain(self.base.flatten())
            .chain(iter::once(self.is_immediate))
            .chain(iter::once(self.is_zero_aux))
            .collect()
    }

    pub fn from_iterator<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            base: BaseAuxCols::from_iterator(iter, lt_air),
            is_immediate: iter.next().unwrap(),
            is_zero_aux: iter.next().unwrap(),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        BaseAuxCols::<T, N>::width(oc) + 2
    }
}

impl<const N: usize, F: AbstractField + Copy> MemoryReadAuxCols<N, F> {
    pub fn disabled(mem_oc: MemoryOfflineChecker) -> Self {
        let width = MemoryReadAuxCols::<N, F>::width(&mem_oc);
        MemoryReadAuxCols::from_slice(&vec![F::zero(); width], &mem_oc)
    }
}
