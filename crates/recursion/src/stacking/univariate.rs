use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::{Itertools, izip};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, EF, F, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus,
        StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, TranscriptBus,
        TranscriptBusMessage,
    },
    stacking::bus::{
        EqBitsLookupBus, EqKernelLookupBus, StackingModuleTidxBus, StackingModuleTidxMessage,
        SumcheckClaimsBus, SumcheckClaimsMessage,
    },
    system::Preflight,
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UnivariateRoundCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    pub tidx: F,

    pub coeff: [F; D_EF],
    pub u_0: [F; D_EF],
    pub u_0_pow: [F; D_EF],

    pub r_0: [F; D_EF],

    pub s_0_sum_over_d: [F; D_EF],

    pub poly_rand_eval: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct UnivariateRoundTraceGenerator;

impl UnivariateRoundTraceGenerator {
    pub fn generate_trace(proof: &Proof, preflight: &Preflight) -> RowMajorMatrix<F> {
        let coeffs = &proof.stacking_proof.univariate_round_coeffs;

        let width = UnivariateRoundCols::<usize>::width();
        let num_rows = coeffs.len().next_power_of_two();

        let mut trace = vec![F::ZERO; num_rows * width];

        let u_0 = preflight.stacking.sumcheck_rnd[0];
        let u_0_pows = u_0.powers().take(coeffs.len()).collect_vec();

        let initial_tidx = preflight.stacking.intermediate_tidx[0];

        let mut poly_rand_eval = EF::ZERO;

        for (i, (coeff, chunk, u_0_pow)) in
            izip!(coeffs.iter(), trace.chunks_mut(width), u_0_pows.iter()).enumerate()
        {
            let cols: &mut UnivariateRoundCols<F> = chunk.borrow_mut();
            cols.is_valid = F::ONE;
            cols.coeff = from_fn(|i| coeff.as_base_slice()[i]);
            cols.u_0 = from_fn(|i| u_0.as_base_slice()[i]);
            cols.u_0_pow = from_fn(|i| u_0_pow.as_base_slice()[i]);
            cols.r_0 = from_fn(|i| preflight.batch_constraint.sumcheck_rnd[0].as_base_slice()[i]);
            cols.tidx = F::from_canonical_usize(initial_tidx + (D_EF * i));
            cols.is_last = F::from_bool(i + 1 == coeffs.len());
            cols.is_first = F::from_bool(i == 0);

            poly_rand_eval += *coeff * *u_0_pow;
            cols.poly_rand_eval = from_fn(|i| poly_rand_eval.as_base_slice()[i]);
        }

        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct UnivariateRoundAir {
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

impl BaseAirWithPublicValues<F> for UnivariateRoundAir {}
impl PartitionedBaseAir<F> for UnivariateRoundAir {}

impl<F> BaseAir<F> for UnivariateRoundAir {
    fn width(&self) -> usize {
        UnivariateRoundCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UnivariateRoundAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &UnivariateRoundCols<AB::Var> = (*local).borrow();
        let _next: &UnivariateRoundCols<AB::Var> = (*next).borrow();

        self.constraint_randomness_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: AB::Expr::ZERO,
                challenge: local.r_0.map(Into::into),
            },
            local.is_last,
        );

        self.stacking_randomness_bus.send(
            builder,
            local.proof_idx,
            StackingSumcheckRandomnessMessage {
                idx: AB::Expr::ZERO,
                challenge: local.u_0.map(Into::into),
            },
            local.is_last,
        );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.coeff[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i + D_EF) + local.tidx,
                    value: local.u_0[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_last,
            );
        }

        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ZERO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ONE,
                tidx: AB::Expr::from_canonical_usize(2 * D_EF) + local.tidx,
            },
            local.is_last,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ZERO,
                value: local.s_0_sum_over_d.map(Into::into),
            },
            local.is_last,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ONE,
                value: local.poly_rand_eval.map(Into::into),
            },
            local.is_last,
        );
    }
}
