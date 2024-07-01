use afs_derive::AlignedBorrow;

use super::EXTENSION_DEGREE;

/// Columns for field extension add/sub chip.
///
/// IO columns for opcode, x, y, result.
#[derive(AlignedBorrow)]
pub struct FieldExtensionArithmeticCols<T> {
    pub io: FieldExtensionArithmeticIoCols<T>,
    pub aux: FieldExtensionArithmeticAuxCols<T>,
}

pub struct FieldExtensionArithmeticIoCols<T> {
    pub opcode: T,
    pub x: [T; EXTENSION_DEGREE],
    pub y: [T; EXTENSION_DEGREE],
    pub z: [T; EXTENSION_DEGREE],
}

pub struct FieldExtensionArithmeticAuxCols<T> {
    pub opcode_lo: T,
    pub opcode_hi: T,
    pub is_mul: T,
    pub is_inv: T,
    pub sum_or_diff: [T; EXTENSION_DEGREE],
    pub product: [T; EXTENSION_DEGREE],
    pub inv_c: T,
    pub inv: [T; EXTENSION_DEGREE],
}

impl<T: Clone + std::fmt::Debug> FieldExtensionArithmeticCols<T> {
    pub const NUM_COLS: usize = 6 * EXTENSION_DEGREE + 6;

    pub fn get_width() -> usize {
        FieldExtensionArithmeticIoCols::<T>::get_width()
            + FieldExtensionArithmeticAuxCols::<T>::get_width()
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let mut idx = 0;

        let opcode = slice[idx].clone();
        idx += 1;

        let x: [T; EXTENSION_DEGREE] = slice[idx..idx + EXTENSION_DEGREE]
            .to_vec()
            .try_into()
            .expect("Expected a vector of length 4");
        idx += EXTENSION_DEGREE;

        let y: [T; EXTENSION_DEGREE] = slice[idx..idx + EXTENSION_DEGREE]
            .to_vec()
            .try_into()
            .expect("Expected a vector of length 4");
        idx += EXTENSION_DEGREE;

        let z: [T; EXTENSION_DEGREE] = slice[idx..idx + EXTENSION_DEGREE]
            .to_vec()
            .try_into()
            .expect("Expected a vector of length 4");
        idx += EXTENSION_DEGREE;

        let opcode_lo = slice[idx].clone();
        idx += 1;

        let opcode_hi = slice[idx].clone();
        idx += 1;

        let is_mul = slice[idx].clone();
        idx += 1;

        let is_inv = slice[idx].clone();
        idx += 1;

        let sum_or_diff: [T; EXTENSION_DEGREE] = slice[idx..idx + EXTENSION_DEGREE]
            .to_vec()
            .try_into()
            .expect("Expected a vector of length 4");
        idx += EXTENSION_DEGREE;

        let product: [T; EXTENSION_DEGREE] = slice[idx..idx + EXTENSION_DEGREE]
            .to_vec()
            .try_into()
            .expect("Expected a vector of length 4");
        idx += EXTENSION_DEGREE;

        let inv_c = slice[idx].clone();
        idx += 1;

        let inv: [T; EXTENSION_DEGREE] = slice[idx..idx + EXTENSION_DEGREE]
            .to_vec()
            .try_into()
            .expect("Expected a vector of length 4");

        Self {
            io: FieldExtensionArithmeticIoCols { opcode, x, y, z },
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

impl<T: Clone> FieldExtensionArithmeticIoCols<T> {
    pub fn get_width() -> usize {
        3 * EXTENSION_DEGREE + 1
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
        3 * EXTENSION_DEGREE + 5
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.opcode_lo.clone(),
            self.opcode_hi.clone(),
            self.is_mul.clone(),
            self.is_inv.clone(),
        ];
        result.extend_from_slice(&self.sum_or_diff);
        result.extend_from_slice(&self.product);
        result.push(self.inv_c.clone());
        result.extend_from_slice(&self.inv);
        result
    }
}
