#[derive(Default)]
pub struct IsEqualVecIOCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub prod: T,
}

impl<T: Clone> IsEqualVecIOCols<T> {
    pub fn new(x: Vec<T>, y: Vec<T>, prod: T) -> Self {
        Self { x, y, prod }
    }

    // Note that the slice this function takes is of an unusual
    // slc should be a whole row of the trace
    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        Self {
            x: slc[0..vec_len].to_vec(),
            y: slc[vec_len..2 * vec_len].to_vec(),
            prod: slc[3 * vec_len - 1].clone(),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct IsEqualVecAuxCols<T> {
    pub prods: Vec<T>,
    pub invs: Vec<T>,
}

impl<T: Clone> IsEqualVecAuxCols<T> {
    pub fn new(prods: Vec<T>, invs: Vec<T>) -> Self {
        Self { prods, invs }
    }

    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        Self {
            prods: slc[0..vec_len].to_vec(),
            invs: slc[vec_len..2 * vec_len].to_vec(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.prods.iter().chain(self.invs.iter()).cloned().collect()
    }
}

#[derive(Default)]
pub struct IsEqualVecCols<T> {
    pub io: IsEqualVecIOCols<T>,
    pub aux: IsEqualVecAuxCols<T>,
}

impl<T: Clone> IsEqualVecCols<T> {
    pub fn new(x: Vec<T>, y: Vec<T>, prods: Vec<T>, invs: Vec<T>) -> Self {
        Self {
            io: IsEqualVecIOCols {
                x,
                y,
                prod: prods[prods.len() - 1].clone(),
            },
            aux: IsEqualVecAuxCols { prods, invs },
        }
    }

    pub fn from_slice(slc: &[T], vec_len: usize) -> Self {
        Self {
            io: IsEqualVecIOCols::from_slice(slc, vec_len),
            aux: IsEqualVecAuxCols::from_slice(&slc[2 * vec_len..], vec_len),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .x
            .iter()
            .chain(self.io.y.iter())
            .chain(self.aux.flatten().iter())
            .cloned()
            .collect()
    }

    pub fn get_width(&self) -> usize {
        4 * self.vec_len()
    }

    pub fn vec_len(&self) -> usize {
        self.io.x.len()
    }
}
