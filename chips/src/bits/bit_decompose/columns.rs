use crate::utils::Word32;

pub struct BitDecomposeCols<const N: usize, T> {
    pub x: Word32<T>,
    pub x_bits: Vec<T>,
}

impl<const N: usize, T: Clone> BitDecomposeCols<N, T> {
    pub fn from_slice(slice: &[T]) -> Self {
        let x = Word32([slice[0].clone(), slice[1].clone()]);
        let x_bits = slice[2..].to_vec();
        Self { x, x_bits }
    }
}
