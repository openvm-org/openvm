use afs_derive::AlignedBorrow;
use p3_field::Field;

use super::FieldArithmeticAir;
/// Columns for field arithmetic chip.
///
/// Five IO columns for rcv_count, opcode, x, y, result.
/// Eight aux columns for interpreting opcode, evaluating indicators, inverse, and explicit computations.
#[derive(Copy, Clone, Debug, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticCols<T> {
    pub io: FieldArithmeticIoCols<T>,
    pub aux: FieldArithmeticAuxCols<T>,
}

#[derive(Copy, Clone, Debug, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticIoCols<T> {
    /// Number of times to receive
    pub rcv_count: T,
    pub opcode: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Copy, Clone, Debug, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticAuxCols<T> {
    pub opcode_lo: T,
    pub opcode_hi: T,
    pub is_mul: T,
    pub is_div: T,
    pub sum_or_diff: T,
    pub product: T,
    pub quotient: T,
    pub divisor_inv: T,
}

impl<T> FieldArithmeticCols<T>
where
    T: Field,
{
    pub const NUM_COLS: usize = 13;
    pub const NUM_IO_COLS: usize = 5;
    pub const NUM_AUX_COLS: usize = 8;

    pub fn get_width() -> usize {
        FieldArithmeticIoCols::<T>::get_width() + FieldArithmeticAuxCols::<T>::get_width()
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }

    pub fn blank_row() -> Self {
        Self {
            io: FieldArithmeticIoCols::<T> {
                rcv_count: T::zero(),
                opcode: T::from_canonical_u8(FieldArithmeticAir::BASE_OP),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            },
            aux: FieldArithmeticAuxCols::<T> {
                opcode_lo: T::zero(),
                opcode_hi: T::zero(),
                is_mul: T::zero(),
                is_div: T::zero(),
                sum_or_diff: T::zero(),
                product: T::zero(),
                quotient: T::zero(),
                divisor_inv: T::zero(),
            },
        }
    }
}

impl<T: Field> FieldArithmeticIoCols<T> {
    pub fn get_width() -> usize {
        5
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.rcv_count, self.opcode, self.x, self.y, self.z]
    }
}

impl<T: Field> FieldArithmeticAuxCols<T> {
    pub fn get_width() -> usize {
        8
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.opcode_lo,
            self.opcode_hi,
            self.is_mul,
            self.is_div,
            self.sum_or_diff,
            self.product,
            self.quotient,
            self.divisor_inv,
        ]
    }
}
