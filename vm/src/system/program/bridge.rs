use std::iter;

use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::ProgramAir;

#[derive(Debug, Clone, Copy)]
pub struct ProgramBus(pub usize);

impl ProgramBus {
    pub fn send_instruction<AB: InteractionBuilder, E: Into<AB::Expr>>(
        &self,
        builder: &mut AB,
        pc: impl Into<AB::Expr>,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = E>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        builder.push_send(
            self.0,
            [pc.into(), opcode.into()].into_iter().chain(
                operands
                    .into_iter()
                    .map(Into::into)
                    .chain(iter::repeat(AB::Expr::zero()))
                    .take(7),
            ),
            multiplicity,
        );
    }
}

impl ProgramAir {
    pub fn eval_interactions<F: Field, AB: PartitionedAirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
    ) {
        let common_trace = builder.common_main();
        let cached_trace = &builder.cached_mains()[0];

        let exec_freq = common_trace.row_slice(0)[0];
        let exec_cols = cached_trace.row_slice(0).to_vec();

        builder.push_receive(self.bus.0, exec_cols, exec_freq);
    }
}
