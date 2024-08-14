use std::{array::from_fn, iter};

use derive_new::new;

#[derive(Copy, Clone, Debug, PartialEq, Eq, new)]
pub struct AccessCell<const WORD_SIZE: usize, T> {
    pub data: [T; WORD_SIZE],
    pub clk: T,
}

impl<const WORD_SIZE: usize, T: Clone> AccessCell<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            data: from_fn(|i| slc[i].clone()),
            clk: slc[WORD_SIZE].clone(),
        }
    }
}

impl<const WORD_SIZE: usize, T> AccessCell<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        self.data.into_iter().chain(iter::once(self.clk)).collect()
    }

    pub fn width() -> usize {
        WORD_SIZE + 1
    }
}
