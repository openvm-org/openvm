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

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, SumcheckClaimBus, SumcheckClaimMessage,
    },
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, TranscriptBus,
        TranscriptBusMessage,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct MultilinearSumcheckCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_idx: T,

    pub tidx: T,
    pub round: T,
    pub r: [T; D_EF],
    pub prefix_invfact: T,
    pub suffix_invfact: T,
    pub forw_invfact: T,
    pub back_invfact: T,
    pub i: T,
    pub eval_at_i: [T; D_EF],
    pub cur_sum: [T; D_EF],
    pub i_is_zero: T,
}

pub struct MultilinearSumcheckAir {
    pub claim_bus: SumcheckClaimBus,
    pub transcript_bus: TranscriptBus,
    pub randomness_bus: ConstraintSumcheckRandomnessBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,
}

impl<F> BaseAirWithPublicValues<F> for MultilinearSumcheckAir {}
impl<F> PartitionedBaseAir<F> for MultilinearSumcheckAir {}

impl<F> BaseAir<F> for MultilinearSumcheckAir {
    fn width(&self) -> usize {
        MultilinearSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for MultilinearSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MultilinearSumcheckCols<AB::Var> = (*local).borrow();
        let _next: &MultilinearSumcheckCols<AB::Var> = (*next).borrow();

        // ...

        for i in 0..D_EF {
            self.claim_bus.receive(
                builder,
                local.proof_idx,
                SumcheckClaimMessage {
                    round: local.round.into(),
                    value: local.eval_at_i.map(|x| x.into()),
                },
                local.is_first.into(),
            );
            self.claim_bus.send(
                builder,
                local.proof_idx,
                SumcheckClaimMessage {
                    round: local.round.into() + AB::Expr::ONE,
                    value: local.cur_sum.map(|x| x.into()),
                },
                local.is_first.into(),
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                    value: local.r[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.i_is_zero,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                    value: local.eval_at_i[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid - local.i_is_zero,
            );
        }
        self.randomness_bus.send(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round + AB::Expr::ONE,
                challenge: local.r.map(|x| x.into()),
            },
            local.i_is_zero,
        );
        self.batch_constraint_conductor_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: local.round + AB::Expr::ONE,
                value: local.r.map(|x| x.into()),
            },
            local.i_is_zero,
        );
    }
}
