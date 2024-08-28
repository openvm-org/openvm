use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::MemoryAuditAir, columns::AuditCols};
use crate::memory::MemoryAddress;

impl MemoryAuditAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: AuditCols<AB::Var>,
    ) {
        let mult = AB::Expr::one() - local.is_extra;
        // Write the initial memory values at initial timestamps
        self.memory_bus
            .write(
                MemoryAddress::new(local.addr_space, local.pointer),
                [local.initial_data],
                AB::Expr::zero(),
            )
            .eval(builder, mult.clone());

        // Read the final memory values at last timestamps when written to
        self.memory_bus
            .read(
                MemoryAddress::new(local.addr_space, local.pointer),
                local.final_cell.data,
                local.final_cell.clk,
            )
            .eval(builder, mult);
    }
}
