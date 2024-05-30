#[derive(Default)]
pub struct IsEqualVecCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub prods: Vec<T>,
    pub invs: Vec<T>,
}

impl<T: Clone> IsEqualVecCols<T> {
    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        let x = slc[0..vec_len].to_vec();
        let y = slc[vec_len..2 * vec_len].to_vec();
        let prods = slc[2 * vec_len..3 * vec_len].to_vec();
        let invs = slc[3 * vec_len..4 * vec_len].to_vec();

        Self { x, y, prods, invs }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.x
            .iter()
            .chain(self.y.iter())
            .chain(self.prods.iter())
            .chain(self.invs.iter())
            .cloned()
            .collect()
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len()
    }

    pub fn vec_len(&self) -> usize {
        self.x.len()
    }
}
