use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;

use super::air::NewMemoryOfflineChecker;
use crate::{cpu::NEW_MEMORY_BUS, memory::manager::access_cell::AccessCell};

impl<const WORD_SIZE: usize> NewMemoryOfflineChecker<WORD_SIZE> {
    pub fn eval_memory_interactions<AB: InteractionBuilder>(
        builder: &mut AB,
        addr_space: AB::Expr,
        pointer: AB::Expr,
        old_cell: AccessCell<WORD_SIZE, AB::Expr>,
        new_cell: AccessCell<WORD_SIZE, AB::Expr>,
        mult: AB::Expr,
    ) {
        builder.push_receive(
            NEW_MEMORY_BUS,
            iter::once(addr_space.clone())
                .chain(iter::once(pointer.clone()))
                .chain(old_cell.data)
                .chain(iter::once(old_cell.clk)),
            mult.clone(),
        );

        builder.push_send(
            NEW_MEMORY_BUS,
            iter::once(addr_space)
                .chain(iter::once(pointer))
                .chain(new_cell.data)
                .chain(iter::once(new_cell.clk)),
            mult,
        );
    }
}
