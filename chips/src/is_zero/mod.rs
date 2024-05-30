#[cfg(test)]
pub mod tests;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

use crate::is_zero::columns::IsZeroCols;
use crate::sub_chip::AirConfig;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroChip {
    pub x: Vec<u32>,
}

impl AirConfig for IsZeroChip {
    type Cols<T> = IsZeroCols<T>;
}

impl IsZeroChip {
    pub fn new(x: Vec<u32>) -> Self {
        Self { x }
    }

    fn is_zero(&self, x: u32) -> u32 {
        if x == 0 {
            1
        } else {
            0
        }
    }

    pub fn request(&self, x: u32) -> u32 {
        self.is_zero(x)
    }
}
