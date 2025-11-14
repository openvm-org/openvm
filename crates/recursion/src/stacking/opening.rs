use std::borrow::{Borrow, BorrowMut};

use itertools::izip;
use openvm_circuit_primitives::{
    SubAir,
    utils::{and, assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField32, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        ColumnClaimsBus, ColumnClaimsMessage, LiftedHeightsBus, LiftedHeightsBusMessage,
        StackingModuleBus, StackingModuleMessage, TranscriptBus, TranscriptBusMessage,
    },
    stacking::{
        bus::{
            ClaimCoefficientsBus, ClaimCoefficientsMessage, EqBitsLookupBus, EqBitsLookupMessage,
            EqKernelLookupBus, EqKernelLookupMessage, StackingModuleTidxBus,
            StackingModuleTidxMessage, SumcheckClaimsBus, SumcheckClaimsMessage,
        },
        utils::{
            ColumnOpeningPair, compute_coefficients, get_stacked_slice_data, sorted_column_claims,
        },
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{assert_one_ext, ext_field_add, ext_field_multiply},
};

///////////////////////////////////////////////////////////////////////////
/// COLUMNS
///////////////////////////////////////////////////////////////////////////
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct OpeningClaimsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,
    pub is_last: F,

    // Received from batch constraints module
    pub sort_idx: F,
    pub part_idx: F,
    pub col_idx: F,
    pub col_claim: [F; D_EF],
    pub rot_claim: [F; D_EF],

    // Used to constrain the order of received messages
    pub is_main: F,
    pub is_transition_main: F,

    // From proof shape (n, n_lift + l_skip, 2^{n_lift + l_skip}, 2^{- (n_lift + l_skip)})
    pub hypercube_dim: F,
    pub log_lifted_height: F,
    pub lifted_height: F,
    pub lifted_height_inv: F,

    // Sampled transcript values
    pub tidx: F,
    pub lambda: [F; D_EF],
    pub lambda_pow: [F; D_EF],

    // Location in stacked matrices
    pub commit_idx: F,
    pub stacked_col_idx: F,
    pub row_idx: F,
    pub is_last_for_claim: F,

    // Lookups to compute claim coefficient
    pub eq_in: [F; D_EF],
    pub k_rot_in: [F; D_EF],
    pub eq_bits: [F; D_EF],

    // Intermediate values to compute claim coefficient
    pub lambda_pow_eq_bits: [F; D_EF],

    // Stacking claim coefficient to be sent
    pub stacking_claim_coefficient: [F; D_EF],

    // RLC of column claims * coefficient using lambda
    pub s_0: [F; D_EF],
}

///////////////////////////////////////////////////////////////////////////
/// TRACE GENERATOR
///////////////////////////////////////////////////////////////////////////
pub struct OpeningClaimsTraceGenerator;

impl OpeningClaimsTraceGenerator {
    #[tracing::instrument(name = "generate_trace(OpeningClaimsAir)", skip_all)]
    pub fn generate_trace(
        vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> RowMajorMatrix<F> {
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = OpeningClaimsCols::<usize>::width();

        if proofs.is_empty() {
            return RowMajorMatrix::new(vec![F::ZERO; width], width);
        }

        let mut combined_trace = Vec::<F>::new();
        let mut total_rows = 0usize;

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights).enumerate() {
            let claims = sorted_column_claims(proof);
            let stacked_slices =
                get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

            let (_, per_slice) = compute_coefficients(
                proof,
                &stacked_slices,
                &preflight.stacking.sumcheck_rnd,
                &preflight.batch_constraint.sumcheck_rnd,
                &preflight.stacking.lambda,
                vk.inner.params.l_skip,
                vk.inner.params.n_stack,
            );

            let num_rows = claims.len();
            let proof_idx_value = F::from_canonical_usize(proof_idx);

            let mut trace = vec![F::ZERO; num_rows * width];

            let mut lambda_pows = preflight.stacking.lambda.square().powers().take(num_rows);
            let mut stacking_claim_coefficient = EF::ZERO;
            let mut s_0 = EF::ZERO;

            let last_main_idx = claims
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(i, claim)| {
                    if claim.part_idx != 0 {
                        Some(i - 1)
                    } else {
                        None
                    }
                })
                .unwrap_or(num_rows - 1);

            for (row_idx, (claim, slice, (eq_in, k_rot_in, eq_bits), chunk)) in
                izip!(claims, stacked_slices, per_slice, trace.chunks_mut(width)).enumerate()
            {
                let ColumnOpeningPair {
                    sort_idx,
                    part_idx,
                    col_idx,
                    col_claim,
                    rot_claim,
                } = claim;
                let cols: &mut OpeningClaimsCols<F> = chunk.borrow_mut();
                cols.proof_idx = proof_idx_value;
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row_idx == 0);
                cols.is_last = F::from_bool(row_idx + 1 == num_rows);

                cols.sort_idx = F::from_canonical_usize(sort_idx);
                cols.part_idx = F::from_canonical_usize(part_idx);
                cols.col_idx = F::from_canonical_usize(col_idx);
                cols.col_claim.copy_from_slice(col_claim.as_base_slice());
                cols.rot_claim.copy_from_slice(rot_claim.as_base_slice());

                cols.is_main = F::from_bool(part_idx == 0);
                cols.is_transition_main =
                    F::from_bool(row_idx + 1 != num_rows && row_idx != last_main_idx);

                let n_lift = slice.n.max(0) as usize;
                cols.hypercube_dim = if slice.n.is_positive() {
                    F::from_canonical_usize(n_lift)
                } else {
                    -F::from_canonical_usize(slice.n.unsigned_abs())
                };
                cols.log_lifted_height = F::from_canonical_usize(n_lift + vk.inner.params.l_skip);
                cols.lifted_height =
                    F::from_canonical_usize(1 << (n_lift + vk.inner.params.l_skip));
                cols.lifted_height_inv = cols.lifted_height.inverse();

                cols.tidx = F::from_canonical_usize(
                    preflight.batch_constraint.tidx_before_column_openings + 2 * row_idx * D_EF,
                );
                cols.lambda
                    .copy_from_slice(preflight.stacking.lambda.as_base_slice());

                let lambda_pow = lambda_pows.next().unwrap();
                cols.lambda_pow.copy_from_slice(lambda_pow.as_base_slice());

                cols.commit_idx = F::from_canonical_usize(slice.commit_idx);
                cols.stacked_col_idx = F::from_canonical_usize(slice.col_idx);
                cols.row_idx = F::from_canonical_usize(slice.row_idx);
                cols.is_last_for_claim = F::from_bool(slice.is_last_for_claim);

                cols.eq_in.copy_from_slice(eq_in.as_base_slice());
                cols.k_rot_in.copy_from_slice(k_rot_in.as_base_slice());
                cols.eq_bits.copy_from_slice(eq_bits.as_base_slice());

                let lambda_pow_eq_bits = lambda_pow * eq_bits;
                cols.lambda_pow_eq_bits
                    .copy_from_slice(lambda_pow_eq_bits.as_base_slice());

                stacking_claim_coefficient +=
                    lambda_pow_eq_bits * (eq_in + preflight.stacking.lambda * k_rot_in);
                cols.stacking_claim_coefficient
                    .copy_from_slice(stacking_claim_coefficient.as_base_slice());
                if slice.is_last_for_claim {
                    stacking_claim_coefficient = EF::ZERO;
                }

                s_0 += lambda_pow * (col_claim + preflight.stacking.lambda * rot_claim);
                cols.s_0.copy_from_slice(s_0.as_base_slice());
            }

            combined_trace.extend(trace);
            total_rows += num_rows;
        }

        let padded_rows = total_rows.next_power_of_two();
        if padded_rows > total_rows {
            let padding_start = combined_trace.len();
            combined_trace.resize(padded_rows * width, F::ZERO);

            let padding_proof_idx = F::from_canonical_usize(proofs.len());
            let mut chunks = combined_trace[padding_start..].chunks_mut(width);
            let num_padded_rows = padded_rows - total_rows;
            for i in 0..num_padded_rows {
                let chunk = chunks.next().unwrap();
                let cols: &mut OpeningClaimsCols<F> = chunk.borrow_mut();
                cols.proof_idx = padding_proof_idx;
                if i + 1 == num_padded_rows {
                    cols.is_last = F::ONE;
                }
            }
        }

        RowMajorMatrix::new(combined_trace, width)
    }
}

///////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////
pub struct OpeningClaimsAir {
    // External buses
    pub lifted_heights_bus: LiftedHeightsBus,
    pub stacking_module_bus: StackingModuleBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub transcript_bus: TranscriptBus,

    // Internal buses
    pub stacking_tidx_bus: StackingModuleTidxBus,
    pub claim_coefficients_bus: ClaimCoefficientsBus,
    pub sumcheck_claims_bus: SumcheckClaimsBus,
    pub eq_kernel_lookup_bus: EqKernelLookupBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,

    // Other fields
    pub n_stack: usize,
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for OpeningClaimsAir {}
impl PartitionedBaseAir<F> for OpeningClaimsAir {}

impl<F> BaseAir<F> for OpeningClaimsAir {
    fn width(&self) -> usize {
        OpeningClaimsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for OpeningClaimsAir
where
    AB::F: PrimeField32,
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &OpeningClaimsCols<AB::Var> = (*local).borrow();
        let next: &OpeningClaimsCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1, 0> {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_zero((local.proof_idx + AB::F::ONE - next.proof_idx) * next.proof_idx);
        builder
            .when(and(not(local.is_valid), local.is_last))
            .assert_zero(next.proof_idx);

        /*
         * Constrain the sortedness of each ColumnClaimsMessage. Main claims (i.e part_idx = 0)
         * should be first and sorted by sort_idx and then col_idx. The remaining claims should
         * be sorted by sort_idx, then part_idx, and finally col_idx.
         */
        builder.assert_bool(local.is_main);
        builder.when(not(local.is_valid)).assert_zero(local.is_main);
        builder.when(local.is_main).assert_zero(local.part_idx);
        builder.when(local.is_main).assert_zero(local.commit_idx);

        builder.assert_bool(local.is_transition_main);
        builder
            .when(local.is_transition_main)
            .assert_eq(local.is_main, next.is_main);
        builder
            .when(local.is_transition_main)
            .assert_one(and(local.is_valid, next.is_valid));
        builder
            .when(local.is_transition_main)
            .assert_zero(local.is_last);
        builder
            .when(and(not(local.is_main), next.is_main))
            .assert_one(local.is_last);

        let mut when_both_main = builder.when(and(local.is_main, local.is_transition_main));
        when_both_main.assert_bool(next.sort_idx - local.sort_idx);
        when_both_main
            .when_ne(local.sort_idx, next.sort_idx)
            .assert_zero(next.col_idx);
        when_both_main
            .when_ne(local.sort_idx + AB::F::ONE, next.sort_idx)
            .assert_one(next.col_idx - local.col_idx);

        let mut when_last_main = builder.when(and(local.is_main, not(next.is_main)));
        when_last_main.assert_zero((next.part_idx - AB::F::ONE) * not(local.is_last));
        when_last_main.assert_zero(next.col_idx * not(local.is_last));

        builder
            .when(and(local.is_valid, not(local.is_last)))
            .assert_bool(next.commit_idx - local.commit_idx);
        builder
            .when(and(local.is_transition_main, not(local.is_main)))
            .when_ne(local.commit_idx + AB::F::ONE, next.commit_idx)
            .assert_one(next.col_idx - local.col_idx);

        /*
         * Compute col_claim[0] + lambda * rot_claim + ... (i.e. RLC of column/rotation claims)
         * and sent it to UnivariateRoundAir, which will constrain that the RLC is equal to the
         * sum of the proof's s_0 polynomial evaluated at each z in D.
         */
        self.column_claims_bus.send(
            builder,
            local.proof_idx,
            ColumnClaimsMessage {
                sort_idx: local.sort_idx.into(),
                part_idx: local.part_idx.into(),
                col_idx: local.col_idx.into(),
                claim: local.col_claim.map(Into::into),
                is_rot: AB::Expr::ZERO,
            },
            local.is_valid,
        );
        self.column_claims_bus.send(
            builder,
            local.proof_idx,
            ColumnClaimsMessage {
                sort_idx: local.sort_idx.into(),
                part_idx: local.part_idx.into(),
                col_idx: local.col_idx.into(),
                claim: local.rot_claim.map(Into::into),
                is_rot: AB::Expr::ONE,
            },
            local.is_valid,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            local.lambda,
            next.lambda,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_multiply(
                ext_field_multiply(local.lambda, local.lambda),
                local.lambda_pow,
            ),
            next.lambda_pow,
        );

        assert_one_ext(&mut builder.when(local.is_first), local.lambda_pow);

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_add::<AB::Expr>(
                local.col_claim,
                ext_field_multiply(local.lambda, local.rot_claim),
            ),
            local.s_0,
        );

        assert_array_eq(
            &mut builder.when(not(local.is_last)),
            ext_field_add(
                local.s_0,
                ext_field_add::<AB::Expr>(
                    ext_field_multiply(next.lambda_pow, next.col_claim),
                    ext_field_multiply(
                        ext_field_multiply(next.lambda_pow, next.lambda),
                        next.rot_claim,
                    ),
                ),
            ),
            next.s_0,
        );

        self.sumcheck_claims_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimsMessage {
                module_idx: AB::Expr::ZERO,
                value: local.s_0.map(Into::into),
            },
            and(local.is_last, local.is_valid),
        );

        /*
         * Compute coefficients for stacking commits and send them to StackingClaimsAir. We do
         * this by mapping each (sort_idx, part_idx, col_idx) to their (commit_idx, col_idx,
         * row_idx) tuple. We can compute this tuple by recording the start and end point of
         * each trace slice in the commit matrix. Note that we assume that column claims are
         * properly ordered to achieve this.
         */
        builder.when(local.is_first).assert_zero(local.commit_idx);
        builder
            .when(local.is_first)
            .assert_zero(local.stacked_col_idx);
        builder.when(local.is_first).assert_zero(local.row_idx);

        builder
            .when(not(local.is_last))
            .assert_bool(next.commit_idx - local.commit_idx);
        builder
            .when(next.commit_idx - local.commit_idx)
            .assert_zero(next.stacked_col_idx);

        builder.assert_bool(local.is_last_for_claim);
        builder
            .when(next.commit_idx - local.commit_idx)
            .assert_one(local.is_last_for_claim);
        builder
            .when(next.stacked_col_idx - local.stacked_col_idx)
            .assert_one(local.is_last_for_claim);
        builder
            .when(and(local.is_valid, local.is_last))
            .assert_one(local.is_last_for_claim);

        builder
            .when(local.is_last_for_claim)
            .assert_zero(next.row_idx);
        builder
            .when(not::<AB::Expr>(local.is_last_for_claim))
            .assert_eq(local.row_idx + local.lifted_height, next.row_idx);

        builder
            .when(and::<AB::Expr>(
                and(local.is_last_for_claim, not(local.is_last)),
                not::<AB::Expr>(next.commit_idx - local.commit_idx),
            ))
            .assert_eq(
                local.row_idx + local.lifted_height,
                AB::F::from_canonical_usize(1 << (self.n_stack + self.l_skip)),
            );

        assert_array_eq(
            builder,
            ext_field_multiply(local.lambda_pow, local.eq_bits),
            local.lambda_pow_eq_bits,
        );

        assert_array_eq(
            &mut builder.when(local.is_first),
            ext_field_multiply(
                local.lambda_pow_eq_bits,
                ext_field_add::<AB::Expr>(
                    local.eq_in,
                    ext_field_multiply(local.lambda, local.k_rot_in),
                ),
            ),
            local.stacking_claim_coefficient,
        );

        assert_array_eq(
            &mut builder.when(local.is_last_for_claim),
            ext_field_multiply(
                next.lambda_pow_eq_bits,
                ext_field_add::<AB::Expr>(
                    next.eq_in,
                    ext_field_multiply(next.lambda, next.k_rot_in),
                ),
            ),
            next.stacking_claim_coefficient,
        );

        assert_array_eq(
            &mut builder.when(not::<AB::Expr>(local.is_last_for_claim)),
            ext_field_add(
                local.stacking_claim_coefficient,
                ext_field_multiply(
                    next.lambda_pow_eq_bits,
                    ext_field_add::<AB::Expr>(
                        next.eq_in,
                        ext_field_multiply(next.lambda, next.k_rot_in),
                    ),
                ),
            ),
            next.stacking_claim_coefficient,
        );

        self.claim_coefficients_bus.send(
            builder,
            local.proof_idx,
            ClaimCoefficientsMessage {
                commit_idx: local.commit_idx,
                stacked_col_idx: local.stacked_col_idx,
                coefficient: local.stacking_claim_coefficient,
            },
            and(local.is_valid, local.is_last_for_claim),
        );

        /*
         * Constrain correctness of lookup values via interactions. Heights are received
         * from ProofShapeAir, while eq(u, r), k_rot(u, r), and eq_>(u, b) values are
         * computed and provided via lookup by other stacking module AIRs.
         */
        self.lifted_heights_bus.receive(
            builder,
            local.proof_idx,
            LiftedHeightsBusMessage {
                sort_idx: local.sort_idx,
                part_idx: local.part_idx,
                commit_idx: local.commit_idx,
                hypercube_dim: local.hypercube_dim,
                lifted_height: local.lifted_height,
                log_lifted_height: local.log_lifted_height,
            },
            local.is_valid,
        );

        self.eq_kernel_lookup_bus.receive(
            builder,
            local.proof_idx,
            EqKernelLookupMessage {
                n: local.hypercube_dim,
                eq_in: local.eq_in,
                k_rot_in: local.k_rot_in,
            },
            local.is_valid,
        );

        builder
            .when(local.is_valid)
            .assert_one(local.lifted_height * local.lifted_height_inv);

        self.eq_bits_lookup_bus.receive(
            builder,
            local.proof_idx,
            EqBitsLookupMessage {
                b_value: local.row_idx
                    * local.lifted_height_inv
                    * AB::F::from_canonical_usize(1 << self.l_skip),
                num_bits: AB::Expr::from_canonical_usize(self.n_stack + self.l_skip)
                    - local.log_lifted_height,
                eval: local.eq_bits.map(Into::into),
            },
            local.is_valid,
        );

        /*
         * Constrain transcript operations and send the final tidx to UnivariateRoundAir.
         */
        self.stacking_module_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx.into(),
            },
            and(local.is_first, local.is_valid),
        );

        builder
            .when(and(local.is_valid, not(local.is_last)))
            .assert_eq(
                local.tidx + AB::Expr::from_canonical_usize(2 * D_EF),
                next.tidx,
            );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                    value: local.col_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(D_EF + i),
                    value: local.rot_claim[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );

            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: AB::Expr::from_canonical_usize(2 * D_EF + i) + local.tidx,
                    value: local.lambda[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                and(local.is_last, local.is_valid),
            );
        }

        self.stacking_tidx_bus.send(
            builder,
            local.proof_idx,
            StackingModuleTidxMessage {
                module_idx: AB::Expr::ZERO,
                tidx: local.tidx + AB::Expr::from_canonical_usize(3 * D_EF),
            },
            and(local.is_last, local.is_valid),
        );
    }
}
