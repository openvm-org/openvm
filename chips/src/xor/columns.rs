use std::borrow::Borrow;

use afs_middleware_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

#[derive(Default, AlignedBorrow)]
pub struct XorHelperCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Clone> XorHelperCols<T> {
    pub fn from_placeholder(placeholder: &T) -> Self {
        Self {
            x: placeholder.clone(),
            y: placeholder.clone(),
            z: placeholder.clone(),
        }
    }
}

pub struct XorCols<const N: usize, T> {
    pub helper: XorHelperCols<T>,
    pub x_bits: Vec<T>,
    pub y_bits: Vec<T>,
    pub z_bits: Vec<T>,
}

impl<const N: usize, T: Clone> XorCols<N, T> {
    pub fn from_placeholder(placeholder: T) -> Self {
        Self {
            helper: XorHelperCols::from_placeholder(&placeholder),
            x_bits: vec![placeholder.clone(); N],
            y_bits: vec![placeholder.clone(); N],
            z_bits: vec![placeholder.clone(); N],
        }
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0].clone();
        let y = slc[1].clone();
        let z = slc[2].clone();

        let x_bits = slc[3..3 + N].to_vec();
        let y_bits = slc[3 + N..3 + 2 * N].to_vec();
        let z_bits = slc[3 + 2 * N..3 + 3 * N].to_vec();

        Self {
            helper: XorHelperCols { x, y, z },
            x_bits,
            y_bits,
            z_bits,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.push(self.helper.x.clone());
        flattened.push(self.helper.y.clone());
        flattened.push(self.helper.z.clone());

        flattened.extend(self.x_bits.iter().cloned());
        flattened.extend(self.y_bits.iter().cloned());
        flattened.extend(self.z_bits.iter().cloned());

        flattened
    }
}

impl<const N: usize, T> Borrow<XorCols<N, T>> for [T] {
    fn borrow(&self) -> &XorCols<N, T> {
        debug_assert_eq!(self.len(), 3 + 3 * N);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<XorCols<N, T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

pub const NUM_XOR_HELPER_COLS: usize = size_of::<XorHelperCols<u8>>();
pub const XOR_HELPER_COL_MAP: XorHelperCols<usize> = make_col_map();

const fn make_col_map() -> XorHelperCols<usize> {
    let indices_arr = indices_arr::<NUM_XOR_HELPER_COLS>();
    unsafe { transmute::<[usize; NUM_XOR_HELPER_COLS], XorHelperCols<usize>>(indices_arr) }
}
