use super::IsEqualVecAir;

#[derive(Default)]
pub struct IsEqualVecIoCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub is_equal: T,
}

impl<T: Clone> IsEqualVecIoCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        let mut res: Vec<T> = self.x.iter().chain(self.y.iter()).cloned().collect();
        res.push(self.is_equal.clone());
        res
    }

    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        let x = slc[0..vec_len].to_vec();
        let y = slc[vec_len..2 * vec_len].to_vec();
        let is_equal = slc[2 * vec_len].clone();
        Self { x, y, is_equal }
    }
}

impl<T> IsEqualVecIoCols<T> {
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
    pub fn flatten(&self) -> Vec<T> {
        self.prods.iter().chain(self.invs.iter()).cloned().collect()
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

pub struct IsEqualVecIoColsMut<'a, T> {
    pub x: &'a mut [T],
    pub y: &'a mut [T],
    pub is_equal: &'a mut T,
}

impl<'a, T> IsEqualVecIoColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], is_equal: &IsEqualVecAir) -> Self {
        let (x, rest) = slc.split_at_mut(is_equal.vec_len);
        let (y, rest) = rest.split_at_mut(is_equal.vec_len);
        let (is_equal, _) = rest.split_first_mut().unwrap();

        Self { x, y, is_equal }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct IsEqualVecAuxColsMut<'a, T> {
    /// prods[i] indicates whether x[i] == y[i] up to the i-th index
    pub prods: &'a mut [T],
    pub invs: &'a mut [T],
}

impl<'a, T> IsEqualVecAuxColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], is_equal: &IsEqualVecAir) -> Self {
        let (prods, invs) = slc.split_at_mut(is_equal.vec_len - 1);

        Self { prods, invs }
    }
}

pub struct IsEqualVecColsMut<'a, T> {
    pub io: IsEqualVecIoColsMut<'a, T>,
    pub aux: IsEqualVecAuxColsMut<'a, T>,
}

impl<'a, T> IsEqualVecColsMut<'a, T> {
    pub fn from_slice(slc: &'a mut [T], is_equal: &IsEqualVecAir) -> Self {
        let (io, aux) = slc.split_at_mut(IsEqualVecIoCols::<T>::get_width(is_equal.vec_len));

        Self {
            io: IsEqualVecIoColsMut::from_slice(io, is_equal),
            aux: IsEqualVecAuxColsMut::from_slice(aux, is_equal),
        }
    }
}
