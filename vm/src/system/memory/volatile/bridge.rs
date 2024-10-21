use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{air::VolatileBoundaryAir, columns::VolatileBoundaryCols};
use crate::system::memory::MemoryAddress;

impl VolatileBoundaryAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: VolatileBoundaryCols<AB::Var>,
    ) {
        let mult = AB::Expr::one() - local.is_extra;
        // Write the initial memory values at initial timestamps
        self.memory_bus
            .send(
                MemoryAddress::new(local.addr_space, local.address),
                vec![local.initial_data],
                AB::Expr::zero(),
            )
            .eval(builder, mult.clone());

        // Read the final memory values at last timestamps when written to
        self.memory_bus
            .receive(
                MemoryAddress::new(local.addr_space, local.address),
                vec![local.final_data],
                local.final_timestamp,
            )
            .eval(builder, mult);
    }
}
