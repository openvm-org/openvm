use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::izip;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poly_common::interpolate_quadratic_at_012,
    poseidon2::sponge::FiatShamirTranscript, proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, EqBitsLookupBus,
        EqKernelLookupBus, StackingModuleTidxBus, StackingModuleTidxMessage,
        StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, SumcheckClaimsBus,
        SumcheckClaimsMessage, TranscriptBus, TranscriptBusMessage,
    },
    system::{BusInventory, Preflight},
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SumcheckRoundsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_last: F,
    pub is_first: F,

    pub round: F,

    pub tidx: F,

    pub s_eval_at_0: [F; D_EF],
    pub s_eval_at_1: [F; D_EF],
    pub s_eval_at_2: [F; D_EF],

    pub s_eval_at_u: [F; D_EF],

    pub u_round: [F; D_EF],
    pub r_round: [F; D_EF],
    pub has_r: F,
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct SumcheckRoundsTraceGenerator;

impl SumcheckRoundsTraceGenerator {
    pub fn generate_trace<TS: FiatShamirTranscript>(
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> RowMajorMatrix<F> {
        let sumcheck_rounds = &proof.stacking_proof.sumcheck_round_polys;

        let width = SumcheckRoundsCols::<usize>::width();
        let num_rows = sumcheck_rounds.len().next_power_of_two();

        let mut trace = vec![F::ZERO; num_rows * width];

        let u = &preflight.stacking.sumcheck_rnd[1..];
        let r = &preflight.batch_constraint_sumcheck_randomness()[1..];

        let initial_tidx = preflight.stacking.intermediate_tidx[1];

        let mut s_eval_at_u = preflight.stacking.univariate_poly_rand_eval;

        for (round, (sumcheck_round, chunk, u_round)) in
            izip!(sumcheck_rounds.iter(), trace.chunks_mut(width), u.iter(),).enumerate()
        {
            let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();
            let s_eval_at_0 = s_eval_at_u - sumcheck_round[0];
            s_eval_at_u = interpolate_quadratic_at_012(
                &[s_eval_at_0, sumcheck_round[0], sumcheck_round[1]],
                *u_round,
            );
            cols.is_valid = F::ONE;
            cols.round = F::from_canonical_usize(round + 1);
            cols.s_eval_at_0 = from_fn(|i| s_eval_at_0.as_base_slice()[i]);
            cols.s_eval_at_1 = from_fn(|i| sumcheck_round[0].as_base_slice()[i]);
            cols.s_eval_at_2 = from_fn(|i| sumcheck_round[1].as_base_slice()[i]);
            // cols.s_eval_at_u = from_fn(|i| s_eval_at_u.as_base_slice()[i]);
            cols.u_round = from_fn(|i| u_round.as_base_slice()[i]);
            cols.tidx = F::from_canonical_usize(initial_tidx + (3 * D_EF * round));
            cols.is_last = F::from_bool(round + 1 == sumcheck_rounds.len());
            cols.is_first = F::from_bool(round == 0);

            if round < r.len() {
                cols.r_round = r[round].challenge;
                cols.has_r = F::ONE;
            }
        }

        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct SumcheckRoundsAir {
    // External buses
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,
}

impl SumcheckRoundsAir {
    pub fn new(buses: &BusInventory) -> Self {
        Self {
            constraint_randomness_bus: buses.constraint_randomness_bus,
            stacking_randomness_bus: buses.stacking_randomness_bus,
            transcript_bus: buses.transcript_bus,
            stacking_tidx_bus: buses.stacking_tidx_bus,
            sumcheck_claims_bus: buses.sumcheck_claims_bus,
            eq_kernel_lookup_bus: buses.eq_kernel_lookup_bus,
            eq_bits_lookup_bus: buses.eq_bits_lookup_bus,
        }
    }
}

impl BaseAirWithPublicValues<F> for SumcheckRoundsAir {}
impl PartitionedBaseAir<F> for SumcheckRoundsAir {}

impl<F> BaseAir<F> for SumcheckRoundsAir {
    fn width(&self) -> usize {
        SumcheckRoundsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for SumcheckRoundsAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &SumcheckRoundsCols<AB::Var> = (*local).borrow();
        let _next: &SumcheckRoundsCols<AB::Var> = (*next).borrow();

        self.constraint_randomness_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round,
                challenge: local.r_round,
            },
            local.has_r,
        );

        self.stacking_randomness_bus.send(
            builder,
            local.proof_idx,
            StackingSumcheckRandomnessMessage {
                idx: local.round,
                challenge: local.u_round,
            },
            local.is_valid,
        );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.s_eval_at_1[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i + D_EF) + local.tidx,
                    value: local.s_eval_at_2[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i + 2 * D_EF) + local.tidx,
                    value: local.u_round[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_valid,
            );
        }

        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ONE,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: AB::Expr::from_canonical_usize(3 * D_EF) + local.tidx,
            },
            local.is_last,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ONE,
                value: from_fn(|i| local.s_eval_at_0[i] + local.s_eval_at_1[i]),
            },
            local.is_first,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.s_eval_at_u.map(Into::into),
            },
            local.is_last,
        );
    }
}
