#[derive(Default)]
pub struct IsEqualVecIoCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub is_equal: T,
}

impl<T: Clone> IsEqualVecIoCols<T> {
    pub fn flatten(&self, buf: &mut [T], start: usize) -> usize {
        buf[start..start + self.x.len()].clone_from_slice(&self.x);
        buf[start + self.x.len()..start + self.x.len() + self.y.len()].clone_from_slice(&self.y);
        buf[start + self.x.len() + self.y.len()] = self.is_equal.clone();
        self.x.len() + self.y.len() + 1
    }

    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        let x = slc[0..vec_len].to_vec();
        let y = slc[vec_len..2 * vec_len].to_vec();
        let is_equal = slc[2 * vec_len].clone();
        Self { x, y, is_equal }
    }

    pub fn get_width(vec_len: usize) -> usize {
        vec_len + vec_len + 1
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct IsEqualVecAuxCols<T> {
    /// prods[i] indicates whether x[i] == y[i] up to the i-th index
    pub prods: Vec<T>,
    pub invs: Vec<T>,
}

impl<T: Clone> IsEqualVecAuxCols<T> {
    pub fn new(prods: Vec<T>, invs: Vec<T>) -> Self {
        Self { prods, invs }
    }

    pub fn to_buf(self, buf: &mut Vec<T>) {
        buf.extend(self.prods);
        buf.extend(self.invs);
    }

    pub fn flatten(self, vec_len: usize) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::width(vec_len));
        self.to_buf(&mut buf);
        buf
    }

    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        let prods = slc[0..vec_len - 1].to_vec();
        let invs = slc[vec_len - 1..2 * vec_len - 1].to_vec();

        Self { prods, invs }
    }

    pub fn width(vec_len: usize) -> usize {
        vec_len + vec_len - 1
    }
}

#[derive(Default)]
pub struct IsEqualVecCols<T> {
    pub io: IsEqualVecIoCols<T>,
    pub aux: IsEqualVecAuxCols<T>,
}

impl<T: Clone> IsEqualVecCols<T> {
    pub fn new(x: Vec<T>, y: Vec<T>, is_equal: T, prods: Vec<T>, invs: Vec<T>) -> Self {
        Self {
            io: IsEqualVecIoCols { x, y, is_equal },
            aux: IsEqualVecAuxCols { prods, invs },
        }
    }

    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        let x = slc[0..vec_len].to_vec();
        let y = slc[vec_len..2 * vec_len].to_vec();
        let is_equal = slc[2 * vec_len].clone();
        let prods = slc[2 * vec_len + 1..3 * vec_len].to_vec();
        let invs = slc[3 * vec_len..4 * vec_len].to_vec();

        Self {
            io: IsEqualVecIoCols { x, y, is_equal },
            aux: IsEqualVecAuxCols { prods, invs },
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result: Vec<T> = self.io.x.iter().chain(self.io.y.iter()).cloned().collect();
        result.push(self.io.is_equal.clone());
        result.extend(self.aux.prods.clone());
        result.extend(self.aux.invs.clone());
        result
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len()
    }

    pub fn vec_len(&self) -> usize {
        self.io.x.len()
    }
}
