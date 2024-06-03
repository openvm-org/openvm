// #[cfg(test)]
// pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

use p3_field::Field;

#[derive(Default)]
/// A chip that checks if a number equals 0
pub struct DummyHashChip<const N: usize, const R: usize> {
    pub rate: usize,
    pub width: usize,
    pub digest_width: usize,
}

impl<const N: usize, const R: usize> DummyHashChip<N, R> {
    pub fn request<F: Field>(curr_state: [F; N], _to_absorb: [F; R]) -> [F; N] {
        curr_state
    }
}
