use afs_derive::AlignedBorrow;
use p3_field::Field;

use super::{FieldExtensionArithmeticAir, EXTENSION_DEGREE};

/// Columns for field extension chip.
///
/// IO columns for opcode, x, y, result.
#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldExtensionArithmeticCols<T> {
    pub io: FieldExtensionArithmeticIoCols<T>,
    pub aux: FieldExtensionArithmeticAuxCols<T>,
}

#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldExtensionArithmeticIoCols<T> {
    pub opcode: T,
    pub x: [T; EXTENSION_DEGREE],
    pub y: [T; EXTENSION_DEGREE],
    pub z: [T; EXTENSION_DEGREE],
}

#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldExtensionArithmeticAuxCols<T> {
    // the lower bit of the opcode - BASE_OP
    pub opcode_lo: T,
    // the upper bit of the opcode - BASE_OP
    pub opcode_hi: T,
    // whether the opcode is BBE4MUL
    pub is_mul: T,
    // whether the opcode is BBE4INV
    pub is_inv: T,
    // the sum x + y if opcode_lo is 0, or the difference x - y if opcode_lo is 1
    pub sum_or_diff: [T; EXTENSION_DEGREE],
    // the product of x and y
    pub product: [T; EXTENSION_DEGREE],
    // the field extension inverse of x
    pub inv: [T; EXTENSION_DEGREE],
}

impl<T: Clone> FieldExtensionArithmeticCols<T>
where
    T: Field,
{
    pub const NUM_COLS: usize = 6 * EXTENSION_DEGREE + 5;

    pub fn get_width() -> usize {
        FieldExtensionArithmeticIoCols::<T>::get_width()
            + FieldExtensionArithmeticAuxCols::<T>::get_width()
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }

    pub fn blank_row() -> Self {
        Self {
            io: FieldExtensionArithmeticIoCols {
                opcode: T::from_canonical_u8(FieldExtensionArithmeticAir::BASE_OP),
                x: [T::zero(); EXTENSION_DEGREE],
                y: [T::zero(); EXTENSION_DEGREE],
                z: [T::zero(); EXTENSION_DEGREE],
            },
            aux: FieldExtensionArithmeticAuxCols {
                opcode_lo: T::zero(),
                opcode_hi: T::zero(),
                is_mul: T::zero(),
                is_inv: T::zero(),
                sum_or_diff: [T::zero(); EXTENSION_DEGREE],
                product: [T::zero(); EXTENSION_DEGREE],
                inv: [T::zero(); EXTENSION_DEGREE],
            },
        }
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
        3 * EXTENSION_DEGREE + 4
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
        result.extend_from_slice(&self.inv);
        result
    }
}
