#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroChip {
    pub x: Vec<u32>,
}

impl IsZeroChip {
    pub fn new(x: Vec<u32>) -> Self {
        Self { x }
    }

    pub fn request<F: Field>(&self, x: F) -> bool {
        x == F::zero()
    }
}
