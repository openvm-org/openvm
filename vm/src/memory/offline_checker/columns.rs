use std::{array, iter};

use afs_primitives::is_less_than::{columns::IsLessThanAuxCols, IsLessThanAir};
use p3_field::Field;

use super::bridge::MemoryOfflineChecker;

// TODO: Remove extraneous old_cell from read cols.
pub type MemoryReadAuxCols<const WORD_SIZE: usize, T> = MemoryWriteAuxCols<WORD_SIZE, T>;

/// DEPRECATED: Use `MemoryReadAuxCols` or `MemoryWriteAuxCols`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryWriteAuxCols<const WORD_SIZE: usize, T> {
    // TODO[jpw]: Remove this; read does not need old_data
    pub(super) prev_data: [T; WORD_SIZE],
    // TODO[zach]: Should be just prev_timestamp: T.
    pub(super) prev_timestamps: [T; WORD_SIZE],
    pub(super) is_immediate: T,
    pub(super) is_zero_aux: T,
    // TODO[jpw]: IsLessThan should be optimized to AssertLessThan
    // TODO[zach]: Should be just clk_lt: T.
    pub(super) clk_lt: [T; WORD_SIZE],
    // TODO[zach]: Should be just clk_lt_aux: IsLessThanAuxCols<T>.
    pub(super) clk_lt_aux: [IsLessThanAuxCols<T>; WORD_SIZE],
}

impl<const WORD_SIZE: usize, T> MemoryWriteAuxCols<WORD_SIZE, T> {
    pub fn new(
        prev_data: [T; WORD_SIZE],
        prev_timestamps: [T; WORD_SIZE],
        is_immediate: T,
        is_zero_aux: T,
        clk_lt: [T; WORD_SIZE],
        clk_lt_aux: [IsLessThanAuxCols<T>; WORD_SIZE],
    ) -> Self {
        Self {
            prev_data,
            prev_timestamps,
            is_immediate,
            is_zero_aux,
            clk_lt,
            clk_lt_aux,
        }
    }
}

impl<const WORD_SIZE: usize, T: Clone> MemoryWriteAuxCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], oc: MemoryOfflineChecker) -> Self {
        let mut pos = 3 * WORD_SIZE + 2;
        Self {
            prev_data: array::from_fn(|i| slc[i].clone()),
            prev_timestamps: array::from_fn(|i| slc[WORD_SIZE + i].clone()),
            is_immediate: slc[2 * WORD_SIZE].clone(),
            is_zero_aux: slc[2 * WORD_SIZE + 1].clone(),
            clk_lt: array::from_fn(|i| slc[2 * WORD_SIZE + 2 + i].clone()),
            clk_lt_aux: array::from_fn(|_| {
                let width = IsLessThanAuxCols::<T>::width(&oc.timestamp_lt_air);
                pos += width;
                IsLessThanAuxCols::from_slice(&slc[pos - width..pos])
            }),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryWriteAuxCols<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        self.prev_data
            .into_iter()
            .chain(self.prev_timestamps)
            .chain(iter::once(self.is_immediate))
            .chain(iter::once(self.is_zero_aux))
            .chain(self.clk_lt)
            .chain(self.clk_lt_aux.into_iter().flat_map(|x| x.flatten()))
            .collect()
    }

    pub fn try_from_iter<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        Self {
            prev_data: array::from_fn(|_| iter.next().unwrap()),
            prev_timestamps: array::from_fn(|_| iter.next().unwrap()),
            is_immediate: iter.next().unwrap(),
            is_zero_aux: iter.next().unwrap(),
            clk_lt: array::from_fn(|_| iter.next().unwrap()),
            clk_lt_aux: array::from_fn(|_| IsLessThanAuxCols::try_from_iter(iter, lt_air)),
        }
    }

    pub fn width(oc: &MemoryOfflineChecker) -> usize {
        3 * WORD_SIZE + 2 + WORD_SIZE * IsLessThanAuxCols::<T>::width(&oc.timestamp_lt_air)
    }
}

impl<const WORD_SIZE: usize, F: Field> MemoryWriteAuxCols<WORD_SIZE, F> {
    pub fn disabled(mem_oc: MemoryOfflineChecker) -> Self {
        let width = MemoryReadAuxCols::<WORD_SIZE, F>::width(&mem_oc);
        MemoryWriteAuxCols::from_slice(&vec![F::zero(); width], mem_oc)
    }
}
