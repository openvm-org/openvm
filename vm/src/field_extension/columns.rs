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
    pub is_valid: T,
    // whether the y read occurs: is_valid * (1 - is_inv)
    pub valid_y_read: T,
    pub start_timestamp: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub d: T,
    pub e: T,
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

impl<T: Clone> FieldExtensionArithmeticCols<T> {
    pub fn get_width() -> usize {
        FieldExtensionArithmeticIoCols::<T>::get_width()
            + FieldExtensionArithmeticAuxCols::<T>::get_width()
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let io = FieldExtensionArithmeticIoCols::<T>::from_slice(
            &slice[..FieldExtensionArithmeticIoCols::<T>::get_width()],
        );
        let aux = FieldExtensionArithmeticAuxCols::<T>::from_slice(
            &slice[FieldExtensionArithmeticIoCols::<T>::get_width()..],
        );
        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }
}

impl<T: Clone> FieldExtensionArithmeticCols<T>
where
    T: Field,
{
    pub fn blank_row() -> Self {
        Self {
            io: FieldExtensionArithmeticIoCols {
                opcode: T::from_canonical_u8(FieldExtensionArithmeticAir::BASE_OP),
                x: [T::zero(); EXTENSION_DEGREE],
                y: [T::zero(); EXTENSION_DEGREE],
                z: [T::zero(); EXTENSION_DEGREE],
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: T::zero(),
                valid_y_read: T::zero(),
                start_timestamp: T::zero(),
                op_a: T::zero(),
                op_b: T::zero(),
                op_c: T::zero(),
                d: T::zero(),
                e: T::zero(),

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

    pub fn from_slice(slice: &[T]) -> Self {
        let opcode = slice[0].clone();

        let x = [
            slice[1].clone(),
            slice[2].clone(),
            slice[3].clone(),
            slice[4].clone(),
        ];
        let y = [
            slice[5].clone(),
            slice[6].clone(),
            slice[7].clone(),
            slice[8].clone(),
        ];
        let z = [
            slice[9].clone(),
            slice[10].clone(),
            slice[11].clone(),
            slice[12].clone(),
        ];

        FieldExtensionArithmeticIoCols { opcode, x, y, z }
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
        3 * EXTENSION_DEGREE + 12
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let is_valid = slice[0].clone();
        let valid_y_read = slice[1].clone();
        let start_timestamp = slice[2].clone();
        let op_a = slice[3].clone();
        let op_b = slice[4].clone();
        let op_c = slice[5].clone();
        let d = slice[6].clone();
        let e = slice[7].clone();

        let opcode_lo = slice[8].clone();
        let opcode_hi = slice[9].clone();
        let is_mul = slice[10].clone();
        let is_inv = slice[11].clone();
        let sum_or_diff = [
            slice[12].clone(),
            slice[13].clone(),
            slice[14].clone(),
            slice[15].clone(),
        ];
        let product = [
            slice[16].clone(),
            slice[17].clone(),
            slice[18].clone(),
            slice[19].clone(),
        ];
        let inv = [
            slice[20].clone(),
            slice[21].clone(),
            slice[22].clone(),
            slice[23].clone(),
        ];

        FieldExtensionArithmeticAuxCols {
            is_valid,
            valid_y_read,
            start_timestamp,
            op_a,
            op_b,
            op_c,
            d,
            e,
            opcode_lo,
            opcode_hi,
            is_mul,
            is_inv,
            sum_or_diff,
            product,
            inv,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.is_valid.clone(),
            self.valid_y_read.clone(),
            self.start_timestamp.clone(),
            self.op_a.clone(),
            self.op_b.clone(),
            self.op_c.clone(),
            self.d.clone(),
            self.e.clone(),
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
