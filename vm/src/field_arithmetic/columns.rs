use afs_derive::AlignedBorrow;
use p3_field::Field;

use super::FieldArithmeticAir;
use crate::arch::columns::ExecutionState;

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
    pub z_address: T,
    pub x_address: T,
    pub y_address: T,
    pub xz_as: T,
    pub y_as: T,
    pub x: T,
    pub y: T,
    pub z: T,
    pub prev_state: ExecutionState<T>,
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
    pub y_is_immediate: T,
    pub is_zero_aux: T,
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
                opcode: T::from_canonical_u8(FieldArithmeticAir::BASE_OP),
                z_address: T::zero(),
                x_address: T::zero(),
                y_address: T::zero(),
                xz_as: T::zero(),
                y_as: T::zero(),
                x: T::zero(),
                y: T::zero(),
                z: T::zero(),
                prev_state: Default::default(),
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
                y_is_immediate: T::one(),
                is_zero_aux: T::zero(),
            },
        }
    }
}

impl<T: Field> FieldArithmeticIoCols<T> {
    pub fn get_width() -> usize {
        10 + ExecutionState::<T>::get_width()
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.rcv_count,
            self.opcode,
            self.z_address,
            self.x_address,
            self.y_address,
            self.xz_as,
            self.y_as,
            self.x,
            self.y,
            self.z,
        ];
        result.extend(self.prev_state.flatten());
        result
    }
}

impl<T: Field> FieldArithmeticAuxCols<T> {
    pub fn get_width() -> usize {
        10
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
            self.y_is_immediate,
            self.is_zero_aux,
        ]
    }
}
