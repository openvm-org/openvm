#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
pub struct IsEqualAir;

impl IsEqualAir {
    pub fn request<F: Field>(&self, x: F, y: F) -> bool {
        x == y
    }
}
