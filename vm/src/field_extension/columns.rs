use std::{array, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_primitives::is_less_than::IsLessThanAir;

use crate::{
    field_extension::{air::FieldExtensionArithmeticAir, chip::EXT_DEG},
    memory::offline_checker::{
        bridge::MemoryOfflineChecker,
        columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
    },
};

/// Columns for field extension chip.
///
/// IO columns for opcode, x, y, result.
#[repr(C)]
pub struct FieldExtensionArithmeticCols<T> {
    pub io: FieldExtensionArithmeticIoCols<T>,
    pub aux: FieldExtensionArithmeticAuxCols<T>,
}

#[derive(Copy, Clone, Debug, Default, AlignedBorrow)]
#[repr(C)]
pub struct FieldExtensionArithmeticIoCols<T> {
    pub pc: T,
    pub timestamp: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub d: T,
    pub e: T,
    pub x: [T; EXT_DEG],
    pub y: [T; EXT_DEG],
    pub z: [T; EXT_DEG],
}

#[repr(C)]
pub struct FieldExtensionArithmeticAuxCols<T> {
    /// Whether the row corresponds an actual event (vs a dummy row for padding).
    pub is_valid: T,
    // whether the opcode is FE4ADD
    pub is_add: T,
    // whether the opcode is FE4SUB
    pub is_sub: T,
    // whether the opcode is BBE4MUL
    pub is_mul: T,
    // whether the opcode is BBE4DIV
    pub is_div: T,
    /// `divisor_inv` is y.inverse() when opcode is BBE4DIV and zero otherwise.
    pub divisor_inv: [T; EXT_DEG],
    /// The aux columns for the x reads.
    pub read_x_aux_cols: MemoryReadAuxCols<EXT_DEG, T>,
    /// The aux columns for the y reads.
    pub read_y_aux_cols: MemoryReadAuxCols<EXT_DEG, T>,
    /// The aux columns for the z writes.
    pub write_aux_cols: MemoryWriteAuxCols<EXT_DEG, T>,
}

impl<T> FieldExtensionArithmeticCols<T> {
    pub fn get_width(air: &FieldExtensionArithmeticAir) -> usize {
        FieldExtensionArithmeticIoCols::<T>::get_width()
            + FieldExtensionArithmeticAuxCols::<T>::get_width(&air.mem_oc)
    }

    pub(crate) fn from_iter<I: Iterator<Item = T>>(iter: &mut I, lt_air: &IsLessThanAir) -> Self {
        let mut next = || iter.next().unwrap();

        Self {
            io: FieldExtensionArithmeticIoCols {
                pc: next(),
                timestamp: next(),
                op_a: next(),
                op_b: next(),
                op_c: next(),
                d: next(),
                e: next(),
                x: array::from_fn(|_| next()),
                y: array::from_fn(|_| next()),
                z: array::from_fn(|_| next()),
            },
            aux: FieldExtensionArithmeticAuxCols {
                is_valid: next(),
                is_add: next(),
                is_sub: next(),
                is_mul: next(),
                is_div: next(),
                divisor_inv: array::from_fn(|_| next()),
                read_x_aux_cols: MemoryReadAuxCols::try_from_iter(iter, lt_air),
                read_y_aux_cols: MemoryReadAuxCols::try_from_iter(iter, lt_air),
                write_aux_cols: MemoryWriteAuxCols::try_from_iter(iter, lt_air),
            },
        }
    }
}

impl<T: Clone> FieldExtensionArithmeticCols<T> {
    pub(crate) fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }
}

impl<T> FieldExtensionArithmeticIoCols<T> {
    pub fn get_width() -> usize {
        size_of::<FieldExtensionArithmeticIoCols<u8>>()
    }
}

impl<T: Clone> FieldExtensionArithmeticIoCols<T> {
    fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.pc.clone(),
            self.timestamp.clone(),
            self.op_a.clone(),
            self.op_b.clone(),
            self.op_c.clone(),
            self.d.clone(),
            self.e.clone(),
        ];
        result.extend_from_slice(&self.x);
        result.extend_from_slice(&self.y);
        result.extend_from_slice(&self.z);
        result
    }
}

impl<T> FieldExtensionArithmeticAuxCols<T> {
    pub fn get_width(oc: &MemoryOfflineChecker) -> usize {
        EXT_DEG
            + 5
            + 2 * MemoryReadAuxCols::<EXT_DEG, T>::width(oc)
            + MemoryWriteAuxCols::<EXT_DEG, T>::width(oc)
    }
}

impl<T: Clone> FieldExtensionArithmeticAuxCols<T> {
    fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.is_valid.clone(),
            self.is_add.clone(),
            self.is_sub.clone(),
            self.is_mul.clone(),
            self.is_div.clone(),
        ];
        result.extend_from_slice(&self.divisor_inv);
        result.extend(self.read_x_aux_cols.clone().flatten());
        result.extend(self.read_y_aux_cols.clone().flatten());
        result.extend(self.write_aux_cols.clone().flatten());
        result
    }
}
