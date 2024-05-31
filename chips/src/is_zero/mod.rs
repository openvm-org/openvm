#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroChip<F: Field> {
    pub x: Vec<F>,
}

impl<F: Field> IsZeroChip<F> {
    pub fn new(x: Vec<F>) -> Self {
        Self { x }
    }

    pub fn request(&self, x: F) -> bool {
        x == F::zero()
    }
}
