use afs_derive::AlignedBorrow;
use p3_field::Field;

use crate::cpu::OpCode;

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
    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,
}

impl<T> FieldArithmeticCols<T>
where
    T: Field,
{
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
                opcode: T::from_canonical_u32(OpCode::FADD as u32),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
            },
            aux: FieldArithmeticAuxCols::<T> {
                is_add: T::one(),
                is_sub: T::zero(),
                is_mul: T::zero(),
                is_div: T::zero(),
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
        5
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.is_add,
            self.is_sub,
            self.is_mul,
            self.is_div,
            self.divisor_inv,
        ]
    }
}
