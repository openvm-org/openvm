use std::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::batch_constraint::bus::{
    BatchConstraintConductorBus, BatchConstraintConductorMessage, BatchConstraintInnerMessageType,
    ExpressionClaimBus,
};

#[derive(AlignedBorrow, Copy, Clone, Debug)]
#[repr(C)]
pub struct ExpressionClaimCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub num_present: T,
    pub is_interaction: T,
    pub idx: T,
    pub mu: [T; D_EF],
    pub value: [T; D_EF],
}

pub struct ExpressionClaimAir {
    pub claim_bus: ExpressionClaimBus,
    pub mu_bus: BatchConstraintConductorBus,
}

impl<F> BaseAirWithPublicValues<F> for ExpressionClaimAir {}
impl<F> PartitionedBaseAir<F> for ExpressionClaimAir {}

impl<F> BaseAir<F> for ExpressionClaimAir {
    fn width(&self) -> usize {
        ExpressionClaimCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpressionClaimAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &ExpressionClaimCols<AB::Var> = (*local).borrow();
        let _next: &ExpressionClaimCols<AB::Var> = (*next).borrow();

        // self.claim_bus.receive(
        //     builder,
        //     local.proof_idx,
        //     ExpressionClaimMessage {
        //         is_interaction: local.is_interaction,
        //         idx: local.idx,
        //         value: local.value,
        //     },
        //     local.is_valid,
        // );

        self.mu_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Mu.to_field(),
                idx: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.is_first * local.is_valid,
        );
    }
}
