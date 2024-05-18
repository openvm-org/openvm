use afs_derive::AlignedBorrow;

#[derive(Default, AlignedBorrow)]
pub struct XorIOCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub struct XorCols<const N: usize, T> {
    pub io: XorIOCols<T>,
    pub x_bits: Vec<T>,
    pub y_bits: Vec<T>,
    pub z_bits: Vec<T>,
}

impl<const N: usize, T: Clone> XorCols<N, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let x = slc[0].clone();
        let y = slc[1].clone();
        let z = slc[2].clone();

        let x_bits = slc[3..3 + N].to_vec();
        let y_bits = slc[3 + N..3 + 2 * N].to_vec();
        let z_bits = slc[3 + 2 * N..3 + 3 * N].to_vec();

        Self {
            io: XorIOCols { x, y, z },
            x_bits,
            y_bits,
            z_bits,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&[self.io.x.clone(), self.io.y.clone(), self.io.z.clone()]);

        flattened.extend_from_slice(&self.x_bits);
        flattened.extend_from_slice(&self.y_bits);
        flattened.extend_from_slice(&self.z_bits);

        flattened
    }

    pub fn get_width() -> usize {
        3 * N + 3
    }

    pub fn cols_to_receive(cols: &[usize]) -> XorIOCols<usize> {
        XorIOCols {
            x: cols[0],
            y: cols[1],
            z: cols[2],
        }
    }
}
