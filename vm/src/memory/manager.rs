use std::{array::from_fn, iter};

use afs_stark_backend::interaction::InteractionBuilder;
use derive_new::new;

use crate::cpu::NEW_MEMORY_BUS;

#[derive(new)]
pub struct MemoryReadWriteOpCols<const WORD_SIZE: usize, T> {
    pub address_space: T,
    pub address: T,

    pub data_read: [T; WORD_SIZE],
    pub clk_read: T,
    pub data_write: [T; WORD_SIZE],
    pub clk_write: T,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryReadWriteOpCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            address_space: slc[0].clone(),
            address: slc[1].clone(),
            data_read: from_fn(|i| slc[2 + i].clone()),
            clk_read: slc[2 + WORD_SIZE].clone(),
            data_write: from_fn(|i| slc[3 + WORD_SIZE + i].clone()),
            clk_write: slc[3 + 2 * WORD_SIZE].clone(),
        }
    }
}

impl<const WORD_SIZE: usize, T> MemoryReadWriteOpCols<WORD_SIZE, T> {
    pub fn flatten(self) -> Vec<T> {
        vec![self.address_space, self.address]
            .into_iter()
            .chain(self.data_read)
            .chain(iter::once(self.clk_read))
            .chain(self.data_write)
            .chain(iter::once(self.clk_write))
            .collect()
    }

    pub fn width() -> usize {
        4 + 2 * WORD_SIZE
    }
}

pub fn eval_memory_interactions<const WORD_SIZE: usize, AB: InteractionBuilder>(
    builder: &mut AB,
    op_cols: MemoryReadWriteOpCols<WORD_SIZE, AB::Var>,
    mult: AB::Expr,
) {
    builder.push_send(
        NEW_MEMORY_BUS,
        iter::once(op_cols.address_space)
            .chain(iter::once(op_cols.address))
            .chain(op_cols.data_write)
            .chain(iter::once(op_cols.clk_write)),
        mult.clone(),
    );

    builder.push_receive(
        NEW_MEMORY_BUS,
        iter::once(op_cols.address_space)
            .chain(iter::once(op_cols.address))
            .chain(op_cols.data_read)
            .chain(iter::once(op_cols.clk_read)),
        mult,
    );
}

#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct MemoryAccessCols<const WORD_SIZE: usize, T> {
    pub enabled: T,

    pub address_space: T,
    // TODO: think if having those two fileds is necessary
    pub is_immediate: T,
    pub is_zero_aux: T,

    pub address: T,

    pub data_read: [T; WORD_SIZE],
    pub clk_read: T,
    pub data_write: [T; WORD_SIZE],
    pub clk_write: T,
}

impl<const WORD_SIZE: usize, T: Clone> MemoryAccessCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            enabled: slc[0].clone(),
            address_space: slc[1].clone(),
            is_immediate: slc[2].clone(),
            is_zero_aux: slc[3].clone(),
            address: slc[4].clone(),
            data_read: from_fn(|i| slc[5 + i].clone()),
            clk_read: slc[5 + WORD_SIZE].clone(),
            data_write: from_fn(|i| slc[6 + WORD_SIZE + i].clone()),
            clk_write: slc[6 + 2 * WORD_SIZE].clone(),
        }
    }
}
