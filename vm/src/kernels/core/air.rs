use std::borrow::Borrow;

use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{CoreAuxCols, CoreCols, CoreIoCols};
use crate::{arch::ExecutionBridge, system::memory::offline_checker::MemoryBridge};

/// Air for the Core. Carries no state and does not own execution.
#[derive(Clone, Debug)]
pub struct CoreAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,

    pub(super) offset: usize,
}

impl<F: Field> PartitionedBaseAir<F> for CoreAir {}
impl<F: Field> BaseAir<F> for CoreAir {
    fn width(&self) -> usize {
        CoreCols::<F>::get_width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for CoreAir {}

impl<AB: AirBuilderWithPublicValues + InteractionBuilder> Air<AB> for CoreAir {
    // TODO: continuation verification checks program counters match up [INT-1732]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = CoreCols::from_slice(local);

        let CoreCols { io, aux } = &local_cols;

        let &CoreIoCols { opcode, .. } = io;

        let &CoreAuxCols {
            is_valid,
            ref operation_flags,
        } = aux;

        builder.assert_bool(is_valid);

        // set correct operation flag
        let mut match_opcode = AB::Expr::zero();
        for (&opcode, &flag) in operation_flags.iter() {
            builder.assert_bool(flag);
            match_opcode += flag * AB::F::from_canonical_usize(opcode as usize);
        }
        builder.when(is_valid).assert_eq(opcode, match_opcode);

        // Turn on all interactions
        self.eval_interactions(builder, io, aux);
    }
}
