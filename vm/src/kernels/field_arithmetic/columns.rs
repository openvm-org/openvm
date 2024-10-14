use std::{iter, mem::size_of};

use afs_derive::AlignedBorrow;
use derive_new::new;

use crate::{
    arch::ExecutionState,
    system::memory::{
        offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
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

#[derive(Copy, Clone, Debug, Default, AlignedBorrow)]
#[repr(C)]
pub struct FieldArithmeticIoCols<T> {
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

    pub read_x_aux_cols: MemoryReadOrImmediateAuxCols<T>,
    pub read_y_aux_cols: MemoryReadOrImmediateAuxCols<T>,
    pub write_z_aux_cols: MemoryWriteAuxCols<T, 1>,
}

impl<T: Clone> FieldArithmeticCols<T> {
    pub const fn get_width() -> usize {
        FieldArithmeticIoCols::<T>::get_width() + FieldArithmeticAuxCols::<T>::get_width()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            io: FieldArithmeticIoCols::from_iter(iter),
            aux: FieldArithmeticAuxCols::from_iter(iter),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }
}

impl<T: Clone> FieldArithmeticIoCols<T> {
    pub const fn get_width() -> usize {
        size_of::<FieldArithmeticIoCols<u8>>()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        Self {
            from_state: ExecutionState::from_iter(iter),
            x: Operand::from_iter(iter),
            y: Operand::from_iter(iter),
            z: Operand::from_iter(iter),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::empty()
            .chain(self.from_state.clone().flatten())
            .chain(self.x.flatten())
            .chain(self.y.flatten())
            .chain(self.z.flatten())
            .collect()
    }
}

impl<T: Clone> FieldArithmeticAuxCols<T> {
    pub const fn get_width() -> usize {
        6 + (2 * MemoryReadOrImmediateAuxCols::<T>::width() + MemoryWriteAuxCols::<T, 1>::width())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let mut next = || iter.next().unwrap();
        Self {
            is_valid: next(),
            is_add: next(),
            is_sub: next(),
            is_mul: next(),
            is_div: next(),
            divisor_inv: next(),
            read_x_aux_cols: MemoryReadOrImmediateAuxCols::from_iterator(iter),
            read_y_aux_cols: MemoryReadOrImmediateAuxCols::from_iterator(iter),
            write_z_aux_cols: MemoryWriteAuxCols::from_iterator(iter),
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

#[derive(Clone, Copy, PartialEq, Debug, Default, new)]
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
