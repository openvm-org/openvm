use afs_derive::AlignedBorrow;
use core::mem::{size_of, transmute};
use p3_util::indices_arr;

// #[derive(Default, AlignedBorrow)]
pub struct NBitXorCols<const N: usize, const M: usize, T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub x_limbs: Vec<T>,
    pub y_limbs: Vec<T>,
    pub z_limbs: Vec<T>,
}

impl<const N: usize, const M: usize, T: Clone> NBitXorCols<N, M, T> {
    pub fn from_placeholder(placeholder: T) -> Self {
        Self {
            x: placeholder.clone(),
            y: placeholder.clone(),
            z: placeholder.clone(),
            x_limbs: vec![placeholder.clone(); (N + M - 1) / M],
            y_limbs: vec![placeholder.clone(); (N + M - 1) / M],
            z_limbs: vec![placeholder.clone(); (N + M - 1) / M],
        }
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = (N + M - 1) / M;

        let x = slc[0].clone();
        let y = slc[1].clone();
        let z = slc[2].clone();
        let x_limbs = slc[3..3 + num_limbs].to_vec();
        let y_limbs = slc[3 + num_limbs..3 + 2 * num_limbs].to_vec();
        let z_limbs = slc[3 + 2 * num_limbs..3 + 3 * num_limbs].to_vec();

        Self {
            x,
            y,
            z,
            x_limbs,
            y_limbs,
            z_limbs,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.push(self.x.clone());
        flattened.push(self.y.clone());
        flattened.push(self.z.clone());

        flattened.extend(self.x_limbs.iter().cloned());
        flattened.extend(self.y_limbs.iter().cloned());
        flattened.extend(self.z_limbs.iter().cloned());

        flattened
    }

    pub fn get_width() -> usize {
        3 + 3 * (N + M - 1) / M
    }
}

// pub const NUM_NBIT_XOR_COLS: usize = size_of::<NBitXorCols<u8>>();
// pub const NBIT_XOR_COL_MAP: NBitXorCols<usize> = make_col_map();

// const fn make_col_map() -> NBitXorCols<usize> {
//     let indices_arr = indices_arr::<NUM_NBIT_XOR_COLS>();
//     unsafe { transmute::<[usize; NUM_NBIT_XOR_COLS], NBitXorCols<usize>>(indices_arr) }
// }
