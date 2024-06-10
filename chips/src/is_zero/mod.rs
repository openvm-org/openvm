#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroChip {}

impl IsZeroChip {
    pub fn request<F: Field>(x: F) -> bool {
        x == F::zero()
    }
}
