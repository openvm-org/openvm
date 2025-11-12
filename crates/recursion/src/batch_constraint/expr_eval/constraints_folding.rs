use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit_primitives::{
    SubAir,
    utils::{assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        BatchConstraintBlobCpu,
        bus::{
            ConstraintsFoldingBus, ConstraintsFoldingMessage, EqNOuterBus, EqNOuterMessage,
            ExpressionClaimBus, ExpressionClaimMessage,
        },
    },
    bus::TranscriptBus,
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{MultiProofVecVec, ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
struct ConstraintsFoldingCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    air_idx: T,
    sort_idx: T,
    constraint_idx: T,
    n_lift: T,

    lambda_tidx: T,
    lambda: [T; D_EF],

    value: [T; D_EF],
    cur_sum: [T; D_EF],
    eq_n: [T; D_EF],

    is_first_in_air: T,
    loop_aux: NestedForLoopAuxCols<T, 1>,
}

pub struct ConstraintsFoldingAir {
    pub transcript_bus: TranscriptBus,
    pub constraint_bus: ConstraintsFoldingBus,
    pub expression_claim_bus: ExpressionClaimBus,
    pub eq_n_outer_bus: EqNOuterBus,
}

impl<F> BaseAirWithPublicValues<F> for ConstraintsFoldingAir {}
impl<F> PartitionedBaseAir<F> for ConstraintsFoldingAir {}

impl<F> BaseAir<F> for ConstraintsFoldingAir {
    fn width(&self) -> usize {
        ConstraintsFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ConstraintsFoldingAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &ConstraintsFoldingCols<AB::Var> = (*local).borrow();
        let next: &ConstraintsFoldingCols<AB::Var> = (*next).borrow();

        type LoopSubAir = NestedForLoopSubAir<2, 1>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx, local.sort_idx],
                        is_first: [local.is_first, local.is_first_in_air],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx, next.sort_idx],
                        is_first: [next.is_first, next.is_first_in_air],
                    }
                    .map_into(),
                ),
                local.loop_aux.map_into(),
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);

        builder.assert_bool(local.is_first_in_air);

        // =========================== indices consistency ===============================
        // When we are within one air, constraint_idx increases by 0/1
        builder
            .when(not(next.is_first_in_air))
            .assert_bool(next.constraint_idx - local.constraint_idx);
        // First constraint_idx within an air is zero
        builder
            .when(local.is_first_in_air)
            .assert_zero(local.constraint_idx);
        builder
            .when(not(next.is_first_in_air))
            .assert_eq(local.n_lift, next.n_lift);
        // TODO receive n_lift by sort_idx or air_idx

        // ======================== lambda and cur sum consistency ============================
        assert_array_eq(
            &mut builder.when(not(next.is_first)),
            local.lambda,
            next.lambda,
        );
        assert_array_eq(
            &mut builder.when(not(next.is_first_in_air)),
            local.cur_sum,
            ext_field_add(
                local.value,
                ext_field_multiply::<AB::Expr>(local.lambda, next.cur_sum),
            ),
        );
        assert_array_eq(
            &mut builder.when(not(next.is_first_in_air)),
            local.eq_n,
            next.eq_n,
        );
        // numerator and the last element of the message are just the corresponding values
        assert_array_eq(
            &mut builder.when(next.is_first_in_air),
            local.cur_sum,
            local.value,
        );

        self.constraint_bus.receive(
            builder,
            local.proof_idx,
            ConstraintsFoldingMessage {
                air_idx: local.air_idx.into(),
                constraint_idx: local.constraint_idx - AB::Expr::ONE,
                value: local.value.map(Into::into),
            },
            local.is_valid * (AB::Expr::ONE - local.is_first_in_air),
        );
        let folded_sum: [AB::Expr; D_EF] = ext_field_add(
            ext_field_multiply_scalar::<AB::Expr>(
                next.cur_sum,
                AB::Expr::ONE - next.is_first_in_air,
            ),
            ext_field_multiply_scalar::<AB::Expr>(local.cur_sum, next.is_first_in_air),
        );
        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ZERO,
                idx: local.sort_idx.into(),
                value: ext_field_multiply(folded_sum, local.eq_n),
            },
            local.is_first_in_air * local.is_valid,
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.lambda_tidx,
            local.lambda,
            local.is_valid * local.is_first,
        );

        self.eq_n_outer_bus.receive(
            builder,
            local.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ZERO,
                n: local.n_lift.into(),
                value: local.eq_n.map(Into::into),
            },
            local.is_first_in_air * local.is_valid,
        );
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::batch_constraint) struct ConstraintsFoldingRecord {
    sort_idx: usize,
    air_idx: usize,
    constraint_idx: usize,
    node_idx: usize,
    is_first_in_air: bool,
    value: EF,
}

pub(in crate::batch_constraint) struct ConstraintsFoldingBlob {
    pub(in crate::batch_constraint) records: MultiProofVecVec<ConstraintsFoldingRecord>,
    // (n, value), n is before lift, can be negative
    pub(in crate::batch_constraint) folded_claims: MultiProofVecVec<(isize, EF)>,
}

pub(in crate::batch_constraint) fn generate_constraints_folding_blob(
    vk: &MultiStarkVerifyingKeyV2,
    bc_blob: &BatchConstraintBlobCpu,
    preflights: &[Preflight],
) -> ConstraintsFoldingBlob {
    let constraints = vk
        .inner
        .per_air
        .iter()
        .map(|vk| vk.symbolic_constraints.constraints.constraint_idx.clone())
        .collect_vec();

    let mut records = MultiProofVecVec::new();
    let mut folded = MultiProofVecVec::new();
    for (preflight, node_claims) in preflights.iter().zip(bc_blob.expr_evals.iter()) {
        let lambda_tidx = preflight.batch_constraint.lambda_tidx;
        let lambda =
            EF::from_base_slice(&preflight.transcript.values()[lambda_tidx..lambda_tidx + D_EF]);

        let vdata = &preflight.proof_shape.sorted_trace_vdata;
        for (sort_idx, (air_idx, v)) in vdata.iter().enumerate() {
            let constrs = &constraints[*air_idx];
            records.push(ConstraintsFoldingRecord {
                // dummy to avoid handling case with no constraints
                sort_idx,
                air_idx: *air_idx,
                constraint_idx: 0,
                node_idx: 0,
                is_first_in_air: true,
                value: EF::ZERO,
            });
            let mut folded_claim = EF::ZERO;
            let mut lambda_pow = EF::ONE;
            for (constraint_idx, &constr) in constrs.iter().enumerate() {
                let value = node_claims[*air_idx][constr];
                folded_claim += lambda_pow * value;
                lambda_pow *= lambda;
                records.push(ConstraintsFoldingRecord {
                    sort_idx,
                    air_idx: *air_idx,
                    constraint_idx: constraint_idx + 1,
                    node_idx: constr,
                    is_first_in_air: false,
                    value,
                });
            }
            let n_lift = v.log_height.saturating_sub(vk.inner.params.l_skip);
            let n = v.log_height as isize - vk.inner.params.l_skip as isize;
            folded.push((
                n,
                folded_claim * preflight.batch_constraint.eq_ns_frontloaded[n_lift],
            ));
        }
        records.end_proof();
        folded.end_proof();
    }
    ConstraintsFoldingBlob {
        records,
        folded_claims: folded,
    }
}

#[tracing::instrument(name = "generate_trace(ConstraintsFoldingAir)", skip_all)]
pub(in crate::batch_constraint) fn generate_constraints_folding_trace(
    blob: &ConstraintsFoldingBlob,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = ConstraintsFoldingCols::<F>::width();

    let total_height = blob.records.len();
    debug_assert!(total_height > 0);
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let lambda_tidx = preflight.batch_constraint.lambda_tidx;
        let lambda_slice = &preflight.transcript.values()[lambda_tidx..lambda_tidx + D_EF];
        let records = &blob.records[pidx];

        trace[cur_height * width..(cur_height + records.len()) * width]
            .par_chunks_exact_mut(width)
            .zip(records.par_iter())
            .for_each(|(chunk, record)| {
                let cols: &mut ConstraintsFoldingCols<_> = chunk.borrow_mut();
                let n_lift = preflight.proof_shape.sorted_trace_vdata[record.sort_idx]
                    .1
                    .log_height
                    .saturating_sub(preflight.proof_shape.l_skip);

                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.air_idx = F::from_canonical_usize(record.air_idx);
                cols.sort_idx = F::from_canonical_usize(record.sort_idx);
                cols.constraint_idx = F::from_canonical_usize(record.constraint_idx);
                cols.n_lift = F::from_canonical_usize(n_lift);
                cols.lambda_tidx = F::from_canonical_usize(lambda_tidx);
                cols.lambda.copy_from_slice(lambda_slice);
                cols.value.copy_from_slice(record.value.as_base_slice());
                cols.eq_n.copy_from_slice(
                    preflight.batch_constraint.eq_ns_frontloaded[n_lift].as_base_slice(),
                );
                cols.is_first_in_air = F::from_bool(record.is_first_in_air);
                cols.loop_aux.is_transition[0] = F::ONE;
            });

        // Setting `cur_sum`
        let mut cur_sum = EF::ZERO;
        let lambda = EF::from_base_slice(lambda_slice);
        trace[cur_height * width..(cur_height + records.len()) * width]
            .chunks_exact_mut(width)
            .rev()
            .for_each(|chunk| {
                let cols: &mut ConstraintsFoldingCols<_> = chunk.borrow_mut();
                cur_sum = cur_sum * lambda + EF::from_base_slice(&cols.value);
                cols.cur_sum.copy_from_slice(cur_sum.as_base_slice());
                if cols.is_first_in_air == F::ONE {
                    cur_sum = EF::ZERO;
                }
            });

        {
            let cols: &mut ConstraintsFoldingCols<_> =
                trace[cur_height * width..(cur_height + 1) * width].borrow_mut();
            cols.is_first = F::ONE;
        }
        cur_height += records.len();
        {
            let cols: &mut ConstraintsFoldingCols<_> =
                trace[(cur_height - 1) * width..cur_height * width].borrow_mut();
            cols.is_last = F::ONE;
            cols.loop_aux.is_transition[0] = F::ZERO;
        }
    }
    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut ConstraintsFoldingCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
            cols.is_first_in_air = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
