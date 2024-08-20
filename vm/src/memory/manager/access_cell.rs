use std::{array::from_fn, iter};

use derive_new::new;
use p3_air::AirBuilder;

// TODO[jpw]: delete this. Cell should refer to a single trace cell, Word refers to multiple cells
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

    pub fn try_from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            data: from_fn(|_| iter.next().unwrap()),
            clk: iter.next().unwrap(),
        }
    }

    pub fn width() -> usize {
        WORD_SIZE + 1
    }
}

impl<const WORD_SIZE: usize, T> AccessCell<WORD_SIZE, T> {
    pub fn into_expr<AB: AirBuilder>(self) -> AccessCell<WORD_SIZE, AB::Expr>
    where
        T: Into<AB::Expr>,
    {
        AccessCell::new(self.data.map(|x| x.into()), self.clk.into())
    }
}

impl<const WORD_SIZE: usize, T: Default> Default for AccessCell<WORD_SIZE, T> {
    fn default() -> Self {
        Self {
            data: from_fn(|_| T::default()),
            clk: T::default(),
        }
    }
}
