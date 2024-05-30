#[derive(Default)]
pub struct IsEqualVecCols<T: Clone> {
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

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.extend(self.x.iter().cloned());
        flattened.extend(self.y.iter().cloned());
        flattened.extend(self.prods.iter().cloned());
        flattened.extend(self.invs.iter().cloned());

        flattened
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len()
    }

    pub fn vec_len(&self) -> usize {
        self.x.len()
    }
}
