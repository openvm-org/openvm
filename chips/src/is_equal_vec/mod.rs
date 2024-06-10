#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default)]
pub struct IsEqualVecAir {
    vec_len: usize,
}

impl IsEqualVecAir {
    pub fn request<F: Clone + PartialEq>(&self, x: &[F], y: &[F]) -> bool {
        x == y
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len
    }
}
