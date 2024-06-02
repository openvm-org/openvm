#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct IsEqualVecChip {
    vec_len: usize,
}

impl IsEqualVecChip {
    pub fn request<F: Clone + PartialEq>(&self, x: &[F], y: &[F]) -> bool {
        x == y
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len
    }
}
