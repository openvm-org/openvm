use std::mem::size_of;

use afs_derive::AlignedBorrow;
use derive_new::new;

use crate::{
    arch::columns::ExecutionState,
    field_arithmetic::FieldArithmeticAir,
    memory::{
        offline_checker::columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};

/// Columns for field arithmetic chip.
///
/// Five IO columns for rcv_count, opcode, x, y, result.
/// Eight aux columns for interpreting opcode, evaluating indicators, inverse, and explicit computations.
#[derive(Clone, Debug)]
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
    pub x: Operand<T>,
    pub y: Operand<T>,
    pub z: Operand<T>,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct FieldArithmeticAuxCols<T> {
    pub is_valid: T,

    pub is_add: T,
    pub is_sub: T,
    pub is_mul: T,
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is FDIV and zero otherwise.
    pub divisor_inv: T,

    pub read_x_aux_cols: MemoryReadAuxCols<1, T>,
    pub read_y_aux_cols: MemoryReadAuxCols<1, T>,
    pub write_z_aux_cols: MemoryWriteAuxCols<1, T>,
}

impl<T: Clone> FieldArithmeticCols<T> {
    pub fn get_width(air: &FieldArithmeticAir) -> usize {
        FieldArithmeticIoCols::<T>::get_width() + FieldArithmeticAuxCols::<T>::get_width(air)
    }

    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I, air: &FieldArithmeticAir) -> Self {
        Self {
            io: FieldArithmeticIoCols::from_iter(iter),
            aux: FieldArithmeticAuxCols::from_iter(iter, air),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }
}

impl<T: Clone> FieldArithmeticIoCols<T> {
    pub fn get_width() -> usize {
        size_of::<FieldArithmeticIoCols<u8>>()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            opcode: iter.next().unwrap(),
            from_state: ExecutionState::from_iter(iter),
            x: Operand::from_iter(iter),
            y: Operand::from_iter(iter),
            z: Operand::from_iter(iter),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.opcode.clone()];
        result.extend(self.from_state.clone().flatten());
        result.extend(self.x.flatten());
        result.extend(self.y.flatten());
        result.extend(self.z.flatten());
        result
    }
}

impl<T: Clone> FieldArithmeticAuxCols<T> {
    pub fn get_width(air: &FieldArithmeticAir) -> usize {
        6 + (2 * MemoryReadAuxCols::<1, T>::width(&air.mem_oc)
            + MemoryWriteAuxCols::<1, T>::width(&air.mem_oc))
    }

    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I, air: &FieldArithmeticAir) -> Self {
        let mut next = || iter.next().unwrap();
        Self {
            is_valid: next(),
            is_add: next(),
            is_sub: next(),
            is_mul: next(),
            is_div: next(),
            divisor_inv: next(),
            read_x_aux_cols: MemoryReadAuxCols::try_from_iter(iter, &air.mem_oc.timestamp_lt_air),
            read_y_aux_cols: MemoryReadAuxCols::try_from_iter(iter, &air.mem_oc.timestamp_lt_air),
            write_z_aux_cols: MemoryWriteAuxCols::try_from_iter(iter, &air.mem_oc.timestamp_lt_air),
        }
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
        result.extend(self.read_x_aux_cols.clone().flatten());
        result.extend(self.read_y_aux_cols.clone().flatten());
        result.extend(self.write_z_aux_cols.clone().flatten());
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

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let mut next = || iter.next().unwrap();
        Self {
            address_space: next(),
            address: next(),
            value: next(),
        }
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
