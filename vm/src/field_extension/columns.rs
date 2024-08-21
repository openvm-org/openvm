use std::array;

use afs_derive::AlignedBorrow;
use afs_primitives::is_less_than::IsLessThanAir;
use p3_field::Field;

use super::{FieldExtensionArithmeticAir, EXTENSION_DEGREE};
use crate::memory::offline_checker::{
    bridge::MemoryOfflineChecker, columns::MemoryOfflineCheckerAuxCols,
};

/// Columns for field extension chip.
///
/// IO columns for opcode, x, y, result.
#[repr(C)]
pub struct FieldExtensionArithmeticCols<const WORD_SIZE: usize, T> {
    pub io: FieldExtensionArithmeticIoCols<T>,
    pub aux: FieldExtensionArithmeticAuxCols<WORD_SIZE, T>,
}

#[derive(AlignedBorrow)]
#[repr(C)]
pub struct FieldExtensionArithmeticIoCols<T> {
    pub opcode: T,
    pub clk: T,
    pub x: [T; EXTENSION_DEGREE],
    pub y: [T; EXTENSION_DEGREE],
    pub z: [T; EXTENSION_DEGREE],
}

#[repr(C)]
pub struct FieldExtensionArithmeticAuxCols<const WORD_SIZE: usize, T> {
    pub is_valid: T,
    // whether the y read occurs: is_valid * (1 - is_inv)
    pub valid_y_read: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub d: T,
    pub e: T,
    // whether the opcode is FE4ADD
    pub is_add: T,
    // whether the opcode is FE4SUB
    pub is_sub: T,
    // whether the opcode is BBE4MUL
    pub is_mul: T,
    // whether the opcode is BBE4INV
    pub is_inv: T,
    // the field extension inverse of x
    pub inv: [T; EXTENSION_DEGREE],
    // There should be 3 * EXTENSION_DEGREE memory accesses (WORD_SIZE=1)
    pub mem_oc_aux_cols: [MemoryOfflineCheckerAuxCols<WORD_SIZE, T>; 12],
}

impl<const WORD_SIZE: usize, T: Clone> FieldExtensionArithmeticCols<WORD_SIZE, T> {
    pub fn get_width(air: &FieldExtensionArithmeticAir<WORD_SIZE>) -> usize {
        FieldExtensionArithmeticIoCols::<T>::get_width()
            + FieldExtensionArithmeticAuxCols::<WORD_SIZE, T>::get_width(&air.mem_oc)
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }

    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        let mut next = || iter.next().unwrap();

        Self {
            io: FieldExtensionArithmeticIoCols {
                opcode: next(),
                clk: next(),
                x: array::from_fn(|_| next()),
                y: array::from_fn(|_| next()),
                z: array::from_fn(|_| next()),
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: next(),
                valid_y_read: next(),
                op_a: next(),
                op_b: next(),
                op_c: next(),
                d: next(),
                e: next(),
                is_add: next(),
                is_sub: next(),
                is_mul: next(),
                is_inv: next(),
                inv: array::from_fn(|_| next()),
                mem_oc_aux_cols: array::from_fn(|_| {
                    MemoryOfflineCheckerAuxCols::try_from_iter(iter, lt_air)
                }),
            },
        }
    }
}

impl<const WORD_SIZE: usize, T: Clone> FieldExtensionArithmeticCols<WORD_SIZE, T> where T: Field {}

impl<T: Clone> FieldExtensionArithmeticIoCols<T> {
    pub fn get_width() -> usize {
        3 * EXTENSION_DEGREE + 2
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.opcode.clone()];

        result.push(self.clk.clone());
        result.extend_from_slice(&self.x);
        result.extend_from_slice(&self.y);
        result.extend_from_slice(&self.z);
        result
    }
}

impl<const WORD_SIZE: usize, T: Clone> FieldExtensionArithmeticAuxCols<WORD_SIZE, T> {
    pub fn get_width(oc: &MemoryOfflineChecker) -> usize {
        EXTENSION_DEGREE + 11 + 12 * MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::width(oc)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.is_valid.clone(),
            self.valid_y_read.clone(),
            self.op_a.clone(),
            self.op_b.clone(),
            self.op_c.clone(),
            self.d.clone(),
            self.e.clone(),
            self.is_add.clone(),
            self.is_sub.clone(),
            self.is_mul.clone(),
            self.is_inv.clone(),
        ];
        result.extend_from_slice(&self.inv);
        for mem_oc_aux_cols in self.mem_oc_aux_cols.iter().cloned() {
            result.extend(mem_oc_aux_cols.flatten());
        }
        result
    }
}
