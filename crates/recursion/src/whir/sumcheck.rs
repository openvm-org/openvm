use core::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poly_common::Squarable, proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, WhirOpeningPointBus, WhirOpeningPointMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{eq_1, ext_field_multiply, interpolate_quadratic},
    whir::bus::{
        WhirAlphaBus, WhirAlphaMessage, WhirEqAlphaUBus, WhirEqAlphaUMessage, WhirSumcheckBus,
        WhirSumcheckBusMessage,
    },
};

/// The columns for `SumcheckAir`.
///
/// Each row in `SumcheckAir` constrains a round of sumcheck. Rows are grouped
/// into groups of size `k_whir`.
#[repr(C)]
#[derive(AlignedBorrow)]
struct SumcheckCols<T> {
    is_enabled: T,
    proof_idx: T,
    whir_round: T,
    /// A counter that goes from 0 to k_whir - 1.
    subidx: T,
    is_first_in_proof: T,
    is_first_in_round: T,
    is_same_proof: T,
    is_same_round: T,
    /// The transcript index at the beginning of the current sumcheck round.
    tidx: T,
    ev1: [T; D_EF],
    ev2: [T; D_EF],
    alpha: [T; D_EF],
    u: [T; D_EF],
    /// The claim at the beginning of the sumcheck round.
    pre_claim: [T; D_EF],
    /// The claim at the end of the group of sumcheck rounds.
    post_group_claim: [T; D_EF],
    /// The value `eq_i(alpha, u)` on the ith sumcheck row (within a proof).
    eq_partial: [T; D_EF],
    /// The number of times the challenge `alpha` is looked up by other AIRs. This is
    /// unconstrained.
    alpha_lookup_count: T,
}

pub struct SumcheckAir {
    pub sumcheck_bus: WhirSumcheckBus,
    pub alpha_bus: WhirAlphaBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub transcript_bus: TranscriptBus,
    pub k: usize,
}

impl BaseAirWithPublicValues<F> for SumcheckAir {}
impl PartitionedBaseAir<F> for SumcheckAir {}

impl<F> BaseAir<F> for SumcheckAir {
    fn width(&self) -> usize {
        SumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for SumcheckAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &SumcheckCols<AB::Var> = (*local).borrow();
        let next: &SumcheckCols<AB::Var> = (*next).borrow();

        let proof_idx = local.proof_idx;
        let is_enabled = local.is_enabled;
        builder.assert_bool(is_enabled);

        NestedForLoopSubAir::<3, 2>.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_enabled.into(),
                        counter: [
                            local.proof_idx.into(),
                            local.whir_round.into(),
                            local.subidx.into(),
                        ],
                        is_first: [
                            local.is_first_in_proof.into(),
                            local.is_first_in_round.into(),
                            AB::Expr::ONE,
                        ],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.is_enabled.into(),
                        counter: [
                            next.proof_idx.into(),
                            next.whir_round.into(),
                            next.subidx.into(),
                        ],
                        is_first: [
                            next.is_first_in_proof.into(),
                            next.is_first_in_round.into(),
                            AB::Expr::ONE,
                        ],
                    },
                ),
                NestedForLoopAuxCols {
                    is_transition: [local.is_same_proof, local.is_same_round],
                }
                .map_into(),
            ),
        );

        builder
            .when(local.is_enabled - local.is_same_round)
            .assert_eq(local.subidx, AB::Expr::from_canonical_usize(self.k - 1));

        let sumcheck_idx = local.whir_round * AB::Expr::from_canonical_usize(self.k) + local.subidx;
        self.sumcheck_bus.receive(
            builder,
            proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx.into(),
                sumcheck_idx: sumcheck_idx.clone(),
                pre_claim: local.pre_claim.map(Into::into),
                post_claim: local.post_group_claim.map(Into::into),
            },
            local.is_first_in_round,
        );

        let post_claim = interpolate_quadratic(local.pre_claim, local.ev1, local.ev2, local.alpha);
        assert_array_eq(
            &mut builder.when(local.is_enabled - local.is_same_round),
            post_claim.clone(),
            local.post_group_claim,
        );

        let mut when_sumcheck_transition = builder.when(local.is_same_round);
        when_sumcheck_transition.assert_zero(next.is_first_in_round);
        assert_array_eq(&mut when_sumcheck_transition, post_claim, next.pre_claim);
        assert_array_eq(
            &mut when_sumcheck_transition,
            next.post_group_claim,
            local.post_group_claim,
        );
        when_sumcheck_transition.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_canonical_usize(3 * D_EF),
        );
        assert_array_eq(
            &mut when_sumcheck_transition,
            next.eq_partial,
            ext_field_multiply::<AB::Expr>(local.eq_partial, eq_1::<AB::Expr>(next.alpha, next.u)),
        );

        when_sumcheck_transition.assert_eq(next.subidx, local.subidx + AB::Expr::ONE);
        builder
            .when(local.is_enabled - local.is_same_round)
            .assert_zero(next.subidx);
        // builder
        //     .when(next.is_enabled)
        //     .when(AB::Expr::ONE - next.is_transition_in_group)
        //     .assert_one(next.is_first_in_group);

        assert_array_eq(
            &mut builder.when(local.is_first_in_proof),
            local.eq_partial,
            eq_1::<AB::Expr>(local.alpha, local.u),
        );

        self.transcript_bus
            .observe_ext(builder, proof_idx, local.tidx, local.ev1, is_enabled);
        self.transcript_bus.observe_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(D_EF),
            local.ev2,
            is_enabled,
        );
        self.transcript_bus.sample_ext(
            builder,
            proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(2 * D_EF),
            local.alpha,
            is_enabled,
        );

        self.alpha_bus.add_key_with_lookups(
            builder,
            proof_idx,
            WhirAlphaMessage {
                idx: sumcheck_idx.clone(),
                challenge: local.alpha.map(Into::into),
            },
            local.alpha_lookup_count,
        );
        self.eq_alpha_u_bus.send(
            builder,
            proof_idx,
            WhirEqAlphaUMessage {
                value: local.eq_partial,
            },
            local.is_enabled - local.is_same_proof,
        );
        self.whir_opening_point_bus.receive(
            builder,
            proof_idx,
            WhirOpeningPointMessage {
                idx: sumcheck_idx,
                value: local.u.map(Into::into),
            },
            is_enabled,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(proofs.len(), preflights.len());

    let params = vk.inner.params;
    let k_whir = params.k_whir;
    let num_whir_rounds = params.num_whir_rounds();

    let whir_opening_point_per_proof = preflights
        .iter()
        .map(|preflight| {
            let sumcheck_rnd = &preflight.stacking.sumcheck_rnd;
            sumcheck_rnd[0]
                .exp_powers_of_2()
                .take(params.l_skip)
                .chain(sumcheck_rnd.iter().skip(1).copied())
                .collect_vec()
        })
        .collect_vec();

    let mut alpha_lookup_counts = vec![0usize; params.num_whir_sumcheck_rounds()];
    let q = params.num_whir_queries;
    for r in 0..num_whir_rounds {
        let base = r * (q + 1);
        for j in 0..k_whir {
            alpha_lookup_counts[r * k_whir + j] = base + q * (1 << (k_whir - 1 - j));
        }
    }

    let rows_per_proof = params.num_whir_sumcheck_rounds();
    let total_valid_rows = rows_per_proof * proofs.len();

    let width = SumcheckCols::<F>::width();
    let height = total_valid_rows.next_power_of_two();
    let mut trace = F::zero_vec(width * height);

    trace
        .par_chunks_exact_mut(width)
        .take(total_valid_rows)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let proof_idx = row_idx / rows_per_proof;
            let i = row_idx % rows_per_proof;

            let whir_round = i / k_whir;
            let j = i % k_whir;

            let proof = &proofs[proof_idx];
            let whir = &preflights[proof_idx].whir;

            let num_rounds = whir.pow_samples.len();
            debug_assert_eq!(whir.alphas.len(), num_rounds * k_whir);

            let is_first_in_group = j == 0;
            let last_group_row_idx = (whir_round + 1) * k_whir - 1;
            let tidx = whir.tidx_per_round[whir_round] + 3 * D_EF * j;

            let cols: &mut SumcheckCols<F> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_canonical_usize(proof_idx);
            cols.is_first_in_proof = F::from_bool(i == 0);
            cols.is_same_proof = F::from_bool(i + 1 < rows_per_proof);
            cols.is_first_in_round = F::from_bool(is_first_in_group);
            cols.is_same_round = F::from_bool(j + 1 < k_whir);
            cols.whir_round = F::from_canonical_usize(whir_round);
            cols.subidx = F::from_canonical_usize(j);
            cols.tidx = F::from_canonical_usize(tidx);
            let sumcheck_polys = &proof.whir_proof.whir_sumcheck_polys[i];
            cols.ev1.copy_from_slice(sumcheck_polys[0].as_base_slice());
            cols.ev2.copy_from_slice(sumcheck_polys[1].as_base_slice());
            cols.eq_partial
                .copy_from_slice(whir.eq_partials[i].as_base_slice());
            cols.alpha.copy_from_slice(whir.alphas[i].as_base_slice());
            cols.u
                .copy_from_slice(whir_opening_point_per_proof[proof_idx][i].as_base_slice());
            cols.pre_claim.copy_from_slice(
                if is_first_in_group {
                    whir.initial_claim_per_round[whir_round]
                } else {
                    whir.post_sumcheck_claims[i - 1]
                }
                .as_base_slice(),
            );
            cols.post_group_claim
                .copy_from_slice(whir.post_sumcheck_claims[last_group_row_idx].as_base_slice());
            cols.alpha_lookup_count =
                F::from_canonical_usize(alpha_lookup_counts[whir_round * k_whir + j]);
        });

    trace
        .chunks_exact_mut(width)
        .skip(total_valid_rows)
        .for_each(|row| {
            let cols: &mut SumcheckCols<F> = row.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len());
        });

    RowMajorMatrix::new(trace, width)
}
