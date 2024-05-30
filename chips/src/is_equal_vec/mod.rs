#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

// use p3_field::Field;

use crate::is_equal_vec::columns::IsEqualVecCols;
use crate::sub_chip::AirConfig;

#[derive(Default)]
pub struct IsEqualVecChip {
    pub x: Vec<Vec<u32>>,
    pub y: Vec<Vec<u32>>,
}

impl AirConfig for IsEqualVecChip {
    type Cols<T> = IsEqualVecCols<T>;
}

impl IsEqualVecChip {
    fn is_equal_vec(&self, x: Vec<u32>, y: Vec<u32>) -> u32 {
        if x == y {
            1
        } else {
            0
        }
    }

    pub fn request(&self, x: Vec<u32>, y: Vec<u32>) -> u32 {
        self.is_equal_vec(x, y)
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len()
    }

    pub fn vec_len(&self) -> usize {
        self.x[0].len()
    }
}
