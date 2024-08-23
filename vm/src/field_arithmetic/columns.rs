use std::borrow::Borrow;

use derive_new::new;
use p3_field::Field;

use afs_derive::AlignedBorrow;

use crate::{
    arch::columns::ExecutionState,
    field_arithmetic::FieldArithmeticAir,
    memory::{
        MemoryAddress,
        offline_checker::columns::{MemoryOfflineCheckerAuxCols, MemoryOfflineCheckerCols},
    },
};

// Danny -- probably makes sense to refactor so that the IO cols are the ones used in the ExecutionBus interaction

/// Columns for field arithmetic chip.
///
/// Five IO columns for rcv_count, opcode, x, y, result.
/// Eight aux columns for interpreting opcode, evaluating indicators, inverse, and explicit computations.
#[derive(Clone, Debug, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticCols<T> {
    pub io: FieldArithmeticIoCols<T>,
    pub aux: FieldArithmeticAuxCols<T>,
}

#[derive(Copy, Clone, Debug, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticIoCols<T> {
    pub opcode: T,
    pub from_state: ExecutionState<T>,
    pub operand1: Operand<T>,
    pub operand2: Operand<T>,
    pub result: Operand<T>,
}

#[derive(Clone, Debug, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticAuxCols<T> {
    pub is_valid: T,

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,

    pub mem_oc_aux_cols: [MemoryOfflineCheckerAuxCols<1, T>; FieldArithmeticAir::TIMESTAMP_DELTA],
}

impl<F> FieldArithmeticCols<F>
where
    F: Field,
{
    pub fn get_width(air: &FieldArithmeticAir) -> usize {
        FieldArithmeticIoCols::<F>::get_width() + FieldArithmeticAuxCols::<F>::get_width(air)
    }

    pub fn flatten(&self) -> Vec<F> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }
}

impl<F: Field> FieldArithmeticIoCols<F> {
    pub fn get_width() -> usize {
        1 + ExecutionState::<F>::get_width() + (3 * Operand::<F>::get_width())
    }

    pub fn flatten(&self) -> Vec<F> {
        let mut result = vec![self.opcode];
        result.extend(self.from_state.flatten());
        result.extend(self.operand1.flatten());
        result.extend(self.operand2.flatten());
        result.extend(self.result.flatten());
        result
    }
}

impl<T: Clone> FieldArithmeticAuxCols<T> {
    pub fn get_width(air: &FieldArithmeticAir) -> usize {
        6 + (3 * MemoryOfflineCheckerCols::<1, T>::width(&air.mem_oc))
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.is_valid.clone(),
            self.is_add.clone(),
            self.is_sub.clone(),
            self.is_mul.clone(),
            self.is_div.clone(),
            self.divisor_inv.clone(),
        ];
        for mem_oc_aux_cols_here in self.mem_oc_aux_cols.clone() {
            result.extend(mem_oc_aux_cols_here.flatten());
        }
        result
    }
}

#[derive(Clone, Copy, PartialEq, Debug, new)]
pub struct Operand<F> {
    pub address_space: F,
    pub address: F,
    pub value: F,
}

impl<T: Clone> Operand<T> {
    pub fn get_width() -> usize {
        3
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.address_space.clone(),
            self.address.clone(),
            self.value.clone(),
        ]
    }

    pub fn memory_address(&self) -> MemoryAddress<T, T> {
        MemoryAddress::new(self.address_space.clone(), self.address.clone())
    }
}
