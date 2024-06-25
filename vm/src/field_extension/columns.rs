use afs_derive::AlignedBorrow;

/// Columns for field extension add/sub chip.
///
/// IO columns for opcode, x, y, result.
#[derive(AlignedBorrow)]
pub struct FieldExtensionArithmeticCols<T> {
    pub io: FieldExtensionArithmeticIOCols<T>,
    pub aux: FieldExtensionArithmeticAuxCols<T>,
}

pub struct FieldExtensionArithmeticIOCols<T> {
    pub opcode: T,
    pub x: [T; 4],
    pub y: [T; 4],
    pub z: [T; 4],
}

pub struct FieldExtensionArithmeticAuxCols<T> {
    pub opcode_lo: T,
    pub opcode_hi: T,
    pub is_mul: T,
    pub is_inv: T,
    pub sum_or_diff: [T; 4],
    pub product: [T; 4],
    pub inv_c: T,
    pub inv: [T; 4],
}

impl<T: Clone> FieldExtensionArithmeticCols<T> {
    pub const NUM_COLS: usize = 30;

    pub fn get_width() -> usize {
        FieldExtensionArithmeticIOCols::<T>::get_width()
            + FieldExtensionArithmeticAuxCols::<T>::get_width()
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let opcode = slice[0].clone();
        let x: [T; 4] = [
            slice[1].clone(),
            slice[2].clone(),
            slice[3].clone(),
            slice[4].clone(),
        ];
        let y: [T; 4] = [
            slice[5].clone(),
            slice[6].clone(),
            slice[7].clone(),
            slice[8].clone(),
        ];
        let z: [T; 4] = [
            slice[9].clone(),
            slice[10].clone(),
            slice[11].clone(),
            slice[12].clone(),
        ];

        let opcode_lo = slice[13].clone();
        let opcode_hi = slice[14].clone();
        let is_mul = slice[15].clone();
        let is_inv = slice[16].clone();
        let sum_or_diff: [T; 4] = [
            slice[17].clone(),
            slice[18].clone(),
            slice[19].clone(),
            slice[20].clone(),
        ];
        let product: [T; 4] = [
            slice[21].clone(),
            slice[22].clone(),
            slice[23].clone(),
            slice[24].clone(),
        ];
        let inv_c = slice[25].clone();
        let inv: [T; 4] = [
            slice[26].clone(),
            slice[27].clone(),
            slice[28].clone(),
            slice[29].clone(),
        ];

        Self {
            io: FieldExtensionArithmeticIOCols { opcode, x, y, z },
            aux: FieldExtensionArithmeticAuxCols {
                opcode_lo,
                opcode_hi,
                is_mul,
                is_inv,
                sum_or_diff,
                product,
                inv_c,
                inv,
            },
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }
}

impl<T: Clone> FieldExtensionArithmeticIOCols<T> {
    pub fn get_width() -> usize {
        13
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.opcode.clone()];
        result.extend_from_slice(&self.x);
        result.extend_from_slice(&self.y);
        result.extend_from_slice(&self.z);
        result
    }
}

impl<T: Clone> FieldExtensionArithmeticAuxCols<T> {
    pub fn get_width() -> usize {
        17
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.opcode_lo.clone(),
            self.opcode_hi.clone(),
            self.is_mul.clone(),
            self.is_inv.clone(),
            self.sum_or_diff[0].clone(),
            self.sum_or_diff[1].clone(),
            self.sum_or_diff[2].clone(),
            self.sum_or_diff[3].clone(),
            self.product[0].clone(),
            self.product[1].clone(),
            self.product[2].clone(),
            self.product[3].clone(),
            self.inv_c.clone(),
            self.inv[0].clone(),
            self.inv[1].clone(),
            self.inv[2].clone(),
            self.inv[3].clone(),
        ]
    }
}
