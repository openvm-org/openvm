use afs_stark_backend::interaction::InteractionBuilder;

use super::air::NewMemoryOfflineChecker;
use crate::{
    cpu::NEW_MEMORY_BUS,
    memory::{manager::access_cell::AccessCell, MemoryAddress},
};

impl<const WORD_SIZE: usize> NewMemoryOfflineChecker<WORD_SIZE> {
    pub fn eval_memory_interactions<AB: InteractionBuilder>(
        builder: &mut AB,
        addr_space: AB::Expr,
        pointer: AB::Expr,
        old_cell: AccessCell<WORD_SIZE, AB::Expr>,
        new_cell: AccessCell<WORD_SIZE, AB::Expr>,
        mult: AB::Expr,
    ) {
        NEW_MEMORY_BUS
            .read(
                old_cell.clk,
                MemoryAddress::new(addr_space.clone(), pointer.clone()),
                old_cell.data,
            )
            .push(builder, mult.clone());
        NEW_MEMORY_BUS
            .write(
                new_cell.clk,
                MemoryAddress::new(addr_space, pointer),
                new_cell.data,
            )
            .push(builder, mult);
    }
}
