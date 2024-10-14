#[derive(Default)]
pub struct XorLimbsCols<const N: usize, const M: usize, T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub x_limbs: Vec<T>,
    pub y_limbs: Vec<T>,
    pub z_limbs: Vec<T>,
}

impl<const N: usize, const M: usize, T: Clone> XorLimbsCols<N, M, T> {
    fn num_limbs() -> usize {
        (N + M - 1) / M
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = Self::num_limbs();

        let x = slc[0].clone();
        let y = slc[1].clone();
        let z = slc[2].clone();
        let x_limbs = slc[3..3 + num_limbs].to_vec();
        let y_limbs = slc[3 + num_limbs..3 + 2 * num_limbs].to_vec();
        let z_limbs = slc[3 + 2 * num_limbs..3 + 3 * num_limbs].to_vec();

        Self {
            x,
            y,
            z,
            x_limbs,
            y_limbs,
            z_limbs,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.push(self.x.clone());
        flattened.push(self.y.clone());
        flattened.push(self.z.clone());

        flattened.extend(self.x_limbs.iter().cloned());
        flattened.extend(self.y_limbs.iter().cloned());
        flattened.extend(self.z_limbs.iter().cloned());

        flattened
    }

    pub fn get_width() -> usize {
        let num_limbs = Self::num_limbs();
        3 + 3 * num_limbs
    }

    pub fn cols_numbered(cols: &[usize]) -> XorLimbsCols<N, M, usize> {
        let num_limbs = Self::num_limbs();

        let x = cols[0];
        let y = cols[1];
        let z = cols[2];
        let x_limbs = cols[3..3 + num_limbs].to_vec();
        let y_limbs = cols[3 + num_limbs..3 + 2 * num_limbs].to_vec();
        let z_limbs = cols[3 + 2 * num_limbs..3 + 3 * num_limbs].to_vec();

        XorLimbsCols {
            x,
            y,
            z,
            x_limbs,
            y_limbs,
            z_limbs,
        }
    }
}
