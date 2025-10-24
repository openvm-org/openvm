use std::borrow::{Borrow, BorrowMut};

use itertools::{Itertools, izip};
use openvm_circuit_primitives::utils::{and, not};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        StackingIndexMessage, StackingIndicesBus, TranscriptBus, TranscriptBusMessage,
        WhirModuleBus, WhirModuleMessage,
    },
    stacking::{
        bus::{
            ClaimCoefficientsBus, ClaimCoefficientsMessage, StackingModuleTidxBus,
            StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
        },
        utils::{compute_coefficients, get_stacked_slice_data},
    },
    system::Preflight,
    utils::{assert_eq_array, assert_one_ext, ext_field_add, ext_field_multiply},
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct StackingClaimsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Correspond to stacking_claim
    pub commit_idx: F,
    pub stacked_col_idx: F,

    // Sampled transcript values
    pub tidx: F,
    pub mu: [F; D_EF],
    pub mu_pow: [F; D_EF],

    // Stacking claim and batched coefficient computed in OpeningClaimsCols
    pub stacking_claim: [F; D_EF],
    pub claim_coefficient: [F; D_EF],

    // Sum of each stacking_claim * claim_coefficient
    pub final_s_eval: [F; D_EF],

    // RLC of stacking claims using mu
    pub whir_claim: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct StackingClaimsTraceGenerator;

impl StackingClaimsTraceGenerator {
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> RowMajorMatrix<F> {
        let claims = proof
            .stacking_proof
            .stacking_openings
            .iter()
            .enumerate()
            .flat_map(|(commit_idx, openings)| {
                openings
                    .iter()
                    .enumerate()
                    .map(move |(stacked_col_idx, opening)| (commit_idx, stacked_col_idx, opening))
            })
            .collect_vec();
        let stacked_slices = get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

        let coeffs = compute_coefficients(
            proof,
            &stacked_slices,
            &preflight.stacking.sumcheck_rnd,
            &preflight.batch_constraint.sumcheck_rnd,
            &preflight.stacking.lambda,
            vk.inner.params.l_skip,
            vk.inner.params.n_stack,
        )
        .0
        .into_iter()
        .flatten()
        .collect_vec();

        let width = StackingClaimsCols::<usize>::width();
        let num_valid = claims.len();
        let num_rows = num_valid.next_power_of_two();

        let mut trace = vec![F::ZERO; num_rows * width];

        let initial_tidx = preflight.stacking.intermediate_tidx[2];

        let mu = preflight.stacking.stacking_batching_challenge;
        let mu_pows = mu.powers().take(claims.len()).collect_vec();

        let mut final_s_eval = EF::ZERO;
        let mut whir_claim = EF::ZERO;

        for (idx, (&(commit_idx, stacked_col_idx, &claim), coeff, chunk)) in
            izip!(&claims, coeffs, trace.chunks_mut(width)).enumerate()
        {
            let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();
            // TODO[stephen]: handle proof idx
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(idx == 0);
            cols.is_last = F::from_bool(idx + 1 == num_valid);

            cols.commit_idx = F::from_canonical_usize(commit_idx);
            cols.stacked_col_idx = F::from_canonical_usize(stacked_col_idx);

            cols.tidx = F::from_canonical_usize(initial_tidx + (D_EF * idx));
            cols.mu.copy_from_slice(mu.as_base_slice());
            cols.mu_pow.copy_from_slice(mu_pows[idx].as_base_slice());

            cols.stacking_claim.copy_from_slice(claim.as_base_slice());
            cols.claim_coefficient
                .copy_from_slice(coeff.as_base_slice());
            final_s_eval += claim * coeff;
            cols.final_s_eval
                .copy_from_slice(final_s_eval.as_base_slice());

            whir_claim += mu_pows[idx] * claim;
            cols.whir_claim.copy_from_slice(whir_claim.as_base_slice());
        }

        if num_valid < num_rows {
            let last_row: &mut StackingClaimsCols<F> = trace[(num_rows - 1) * width..].borrow_mut();
            last_row.is_last = F::ONE;
        }

        RowMajorMatrix::new(trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct StackingClaimsAir {
    // External buses
    pub stacking_indices_bus: StackingIndicesBus,
    pub whir_module_bus: WhirModuleBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
}

impl BaseAirWithPublicValues<F> for StackingClaimsAir {}
impl PartitionedBaseAir<F> for StackingClaimsAir {}

impl<F> BaseAir<F> for StackingClaimsAir {
    fn width(&self) -> usize {
        StackingClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for StackingClaimsAir
where
    AB::F: PrimeField32,
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &StackingClaimsCols<AB::Var> = (*local).borrow();
        let next: &StackingClaimsCols<AB::Var> = (*next).borrow();

        // TODO[stephen]: constrain proof_idx, is_valid, is_first, is_last

        /*
         * Constrain that commit_idx and stacked_col_idx increment correctly.
         */
        builder.when(local.is_first).assert_zero(local.commit_idx);
        builder
            .when(local.is_first)
            .assert_zero(local.stacked_col_idx);

        builder.when(not(local.is_last)).assert_zero(
            (next.commit_idx - local.commit_idx - AB::Expr::ONE)
                * (next.stacked_col_idx - local.stacked_col_idx - AB::Expr::ONE)
                - not(local.is_valid),
        );

        self.stacking_indices_bus.send(
            builder,
            local.proof_idx,
            StackingIndexMessage {
                commit_idx: local.commit_idx,
                col_idx: local.stacked_col_idx,
            },
            local.is_valid,
        );

        /*
         * Compute the running sum of stacking_claim * claim_coefficient values and then
         * constrain the final result to be equal to s_{n_stack}(u_{n_stack}), which is
         * sent from SumcheckRoundsAir.
         */
        self.claim_coefficients_bus.receive(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.claim_coefficient,
            },
            local.is_valid,
        );

        assert_eq_array(
            &mut builder.when(local.is_first),
            ext_field_multiply(local.stacking_claim, local.claim_coefficient),
            local.final_s_eval,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                ext_field_multiply(next.stacking_claim, next.claim_coefficient),
                local.final_s_eval,
            ),
            next.final_s_eval,
        );

        self.sumcheck_claims_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::TWO,
                value: local.final_s_eval.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Constrain transcript operations and send the final tidx to the WHIR module.
         */
        self.stacking_tidx_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::TWO,
                tidx: local.tidx.into(),
            },
            local.is_first,
        );

        builder
            .when(not(local.is_last) * local.is_valid)
            .assert_eq(local.tidx + AB::F::from_canonical_usize(D_EF), next.tidx);

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i) + local.tidx,
                    value: local.stacking_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(i + D_EF) + local.tidx,
                    value: local.mu[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        /*
         * Compute the RLC of the stacking claims and send it to the WHIR module.
         */
        assert_one_ext(&mut builder.when(local.is_first), local.mu_pow);
        assert_eq_array(&mut builder.when(not(local.is_last)), local.mu, next.mu);
        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(local.mu, local.mu_pow),
            next.mu_pow,
        );

        assert_eq_array(
            &mut builder.when(local.is_first),
            local.stacking_claim,
            local.whir_claim,
        );

        assert_eq_array(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                ext_field_multiply(next.stacking_claim, next.mu_pow),
                local.whir_claim,
            ),
            next.whir_claim,
        );

        self.whir_module_bus.send(
            builder,
            local.proof_idx,
            WhirModuleMessage {
                tidx: AB::Expr::from_canonical_usize(2 * D_EF) + local.tidx,
                mu: local.mu.map(Into::into),
                claim: local.whir_claim.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );
    }
}
