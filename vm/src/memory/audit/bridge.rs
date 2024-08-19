use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::MemoryAuditAir, columns::AuditCols};
use crate::{cpu::NEW_MEMORY_BUS, memory::MemoryAddress};

impl<const WORD_SIZE: usize> MemoryAuditAir<WORD_SIZE> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: AuditCols<WORD_SIZE, AB::Var>,
    ) {
        let mult = AB::Expr::one() - local.is_extra;
        // Write the initial memory values at initial timestamps
        NEW_MEMORY_BUS
            .write(
                MemoryAddress::new(local.addr_space, local.pointer),
                local.initial_cell.data,
                local.initial_cell.clk,
            )
            .push(builder, mult.clone());

        // Read the final memory values at last timestamps when written to
        NEW_MEMORY_BUS
            .read(
                MemoryAddress::new(local.addr_space, local.pointer),
                local.final_cell.data,
                local.final_cell.clk,
            )
            .push(builder, mult);
    }
}
