use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::{CastFAir, FINAL_LIMB_SIZE, LIMB_SIZE},
    columns::{CastFAuxCols, CastFIoCols},
};
use crate::{arch::instructions::CastfOpcode, system::memory::MemoryAddress};

impl CastFAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &CastFIoCols<AB::Var>,
        aux: &CastFAuxCols<AB::Var>,
    ) {
        let timestamp: AB::Var = io.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let intermed_val =
            io.x.iter()
                .enumerate()
                .fold(AB::Expr::zero(), |acc, (i, &limb)| {
                    acc + limb * AB::Expr::from_canonical_u32(1 << (i * LIMB_SIZE))
                });

        self.memory_bridge
            .read(
                MemoryAddress::new(io.e, io.op_b),
                [intermed_val],
                timestamp_pp(),
                &aux.read_y_aux_cols,
            )
            .eval(builder, aux.is_valid);

        self.memory_bridge
            .write(
                MemoryAddress::new(io.d, io.op_a),
                io.x,
                timestamp_pp(),
                &aux.write_x_aux_cols,
            )
            .eval(builder, aux.is_valid);

        for i in 0..4 {
            self.bus
                .range_check(
                    io.x[i],
                    match i {
                        0..=2 => LIMB_SIZE,
                        3 => FINAL_LIMB_SIZE,
                        _ => unreachable!(),
                    },
                )
                .eval(builder, aux.is_valid);
        }

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(CastfOpcode::CASTF as usize + self.offset),
                [
                    io.op_a.into(),
                    io.op_b.into(),
                    AB::Expr::zero(),
                    io.d.into(),
                    io.e.into(),
                ],
                io.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, aux.is_valid);
    }
}
