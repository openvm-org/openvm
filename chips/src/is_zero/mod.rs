#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct IsZeroAir;

impl IsZeroAir {
    pub fn request<F: Field>(x: F) -> bool {
        x == F::zero()
    }
}
