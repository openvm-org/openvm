use afs_derive::AlignedBorrow;

#[derive(Default, AlignedBorrow)]
pub struct XorIoCols<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// Bit decompositions
pub struct XorBitCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub z: Vec<T>,
}

pub struct XorCols<const N: usize, T> {
    pub io: XorIoCols<T>,
    pub bits: XorBitCols<T>,
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
            io: XorIoCols { x, y, z },
            bits: XorBitCols {
                x: x_bits,
                y: y_bits,
                z: z_bits,
            },
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&[self.io.x.clone(), self.io.y.clone(), self.io.z.clone()]);

        flattened.extend_from_slice(&self.bits.x);
        flattened.extend_from_slice(&self.bits.y);
        flattened.extend_from_slice(&self.bits.z);

        flattened
    }

    pub fn get_width() -> usize {
        3 * N + 3
    }

    pub fn cols_to_receive(cols: &[usize]) -> XorIoCols<usize> {
        XorIoCols {
            x: cols[0],
            y: cols[1],
            z: cols[2],
        }
    }
}
