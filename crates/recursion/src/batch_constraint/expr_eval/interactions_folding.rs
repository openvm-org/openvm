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
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::{MultiStarkVerifyingKey0V2, MultiStarkVerifyingKeyV2},
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        BatchConstraintBlobCpu,
        bus::{
            Eq3bBus, Eq3bMessage, ExpressionClaimBus, ExpressionClaimMessage,
            InteractionsFoldingBus, InteractionsFoldingMessage,
        },
        eq_airs::Eq3bBlob,
    },
    bus::{AirShapeBus, AirShapeBusMessage, AirShapeProperty, TranscriptBus},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        MultiProofVecVec, MultiVecWithBounds, assert_zeros, ext_field_add, ext_field_multiply,
    },
};

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
struct InteractionsFoldingCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    beta_tidx: T,

    air_idx: T,
    sort_idx: T,
    interaction_idx: T,
    node_idx: T,

    has_interactions: T,

    is_first_in_air: T,
    /// It's true for the num row, which doesn't need to be beta folded.
    is_first_in_message: T, // aka "is_mult"
    // the second in message is the first denom, and it's cur_sum is the folded denom
    is_second_in_message: T,
    is_bus_index: T,

    loop_aux: NestedForLoopAuxCols<T, 2>,

    idx_in_message: T,
    value: [T; D_EF],
    /// Current sum for doing beta folding. This is the value for one interaction.
    /// When local.is_first_in_message, next.cur_sum should be the folded denom.
    /// (because local row is for the num row, which doesn't need to be beta folded)
    /// It doesn't multiply with eq_3b yet.
    cur_sum: [T; D_EF],
    beta: [T; D_EF],
    eq_3b: [T; D_EF],

    /// The summed num and denom for all interactions.
    /// It's summed over all the interactions in the AIR: cur_sum * eq_3b when is_first_in_message
    final_acc_num: [T; D_EF],
    final_acc_denom: [T; D_EF],
}

pub struct InteractionsFoldingAir {
    pub interaction_bus: InteractionsFoldingBus,
    pub air_shape_bus: AirShapeBus,
    pub transcript_bus: TranscriptBus,
    pub expression_claim_bus: ExpressionClaimBus,
    pub eq_3b_bus: Eq3bBus,
}

impl<F> BaseAirWithPublicValues<F> for InteractionsFoldingAir {}
impl<F> PartitionedBaseAir<F> for InteractionsFoldingAir {}

impl<F> BaseAir<F> for InteractionsFoldingAir {
    fn width(&self) -> usize {
        InteractionsFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for InteractionsFoldingAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &InteractionsFoldingCols<AB::Var> = (*local).borrow();
        let next: &InteractionsFoldingCols<AB::Var> = (*next).borrow();

        type LoopSubAir = NestedForLoopSubAir<3, 2>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx, local.sort_idx, local.interaction_idx],
                        is_first: [
                            local.is_first,
                            local.is_first_in_air,
                            local.is_first_in_message,
                        ],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx, next.sort_idx, next.interaction_idx],
                        is_first: [
                            next.is_first,
                            next.is_first_in_air,
                            next.is_first_in_message,
                        ],
                    }
                    .map_into(),
                ),
                local.loop_aux.map_into(),
            ),
        );

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);

        builder.assert_bool(local.has_interactions);
        builder.assert_bool(local.is_first_in_air);
        builder.assert_bool(local.is_first_in_message);
        builder
            .when(local.has_interactions)
            .assert_one(local.is_valid);

        // =========================== indices consistency ===============================
        // When we are within one proof, sort_idx increases by 0/1
        builder
            .when(not(next.is_first))
            .assert_bool(next.sort_idx - local.sort_idx);
        // When we are within one AIR, interaction_idx increases by 0/1 as well
        let within_one_air = not(next.is_first) * (AB::Expr::ONE - next.sort_idx + local.sort_idx);
        builder
            .when(within_one_air.clone())
            .assert_bool(next.interaction_idx - local.interaction_idx);
        // First AIR within a proof is zero, and first interaction within an AIR is also zero
        builder.when(local.is_first).assert_zero(local.sort_idx);
        builder
            .when(not::<AB::Expr>(within_one_air))
            .assert_zero(next.interaction_idx);

        // // =========================== general consistency ================================
        // The row describes an AIR without interactions iff it's first and last in the message,
        // unless the row is invalid
        builder.when(local.is_valid).assert_eq(
            local.is_first_in_message * next.is_first_in_message,
            not(local.has_interactions),
        );
        // If we have interactions, then the row is valid
        builder
            .when(local.has_interactions)
            .assert_one(local.is_valid);
        // If we don't have interactions and the row is valid, then it's first and last _within AIR_
        builder
            .when(not(local.has_interactions))
            .when(local.is_valid)
            .assert_one(local.is_first_in_air);
        builder
            .when(not(local.has_interactions))
            .when(local.is_valid)
            .assert_one(next.is_first_in_air);
        // // If it's last in the interaction and the row is valid, then its value is just bus_idx +
        // 1 assert_array_eq(
        //     &mut builder.when(next.is_first_in_message).when(local.is_valid),
        //     local.value,
        //     base_to_ext::<AB::Expr>(local.node_idx + AB::Expr::ONE),
        // );
        // TODO: receive something from the symbolic expr air to check that it's indeed the bus
        // index TODO: otherwise receive the value by node_idx

        // final_acc_num only changes when it's first in message
        assert_array_eq(
            &mut builder
                .when(not(local.is_first_in_message) * local.is_valid * not(next.is_first_in_air)),
            local.final_acc_num,
            next.final_acc_num,
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_message * local.has_interactions),
            local.final_acc_num,
            ext_field_add(
                next.final_acc_num,
                ext_field_multiply(local.cur_sum, local.eq_3b),
            ),
        );
        assert_zeros(
            &mut builder
                .when(local.is_first_in_message * (local.is_valid - local.has_interactions)),
            local.final_acc_num,
        );
        // final_acc_denom only changes when it's second in message
        assert_array_eq(
            &mut builder.when(
                (not(local.is_second_in_message) + not(local.has_interactions))
                    * local.is_valid
                    * not(next.is_first_in_air),
            ),
            local.final_acc_denom,
            next.final_acc_denom,
        );
        assert_array_eq(
            &mut builder.when(local.is_second_in_message * local.is_valid),
            local.final_acc_denom,
            ext_field_add(
                next.final_acc_denom,
                ext_field_multiply(local.cur_sum, local.eq_3b),
            ),
        );
        // Constraint is_second_in_message
        builder.assert_bool(local.is_second_in_message);
        builder
            .when(local.is_first_in_message * local.has_interactions)
            .assert_one(next.is_second_in_message);
        builder
            .when(next.is_second_in_message)
            .assert_one(local.is_first_in_message);

        // ======================== beta and cur sum consistency ============================
        assert_array_eq(&mut builder.when(not(next.is_first)), local.beta, next.beta);
        assert_array_eq(
            &mut builder.when(not(next.is_first_in_message) * not(local.is_first_in_message)),
            local.cur_sum,
            ext_field_add(
                local.value,
                ext_field_multiply::<AB::Expr>(local.beta, next.cur_sum),
            ),
        );
        // numerator and the last element of the message are just the corresponding values
        assert_array_eq(
            &mut builder.when(next.is_first_in_message + local.is_first_in_message),
            local.cur_sum,
            local.value,
        );

        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ONE,
                idx: local.sort_idx * AB::Expr::TWO,
                // value: ext_field_multiply(local.cur_sum, local.eq_3b),
                value: local.final_acc_num.map(Into::into),
            },
            local.is_first_in_air * local.is_valid,
        );
        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ONE,
                idx: local.sort_idx * AB::Expr::TWO + AB::Expr::ONE,
                // value: ext_field_multiply(next.cur_sum, next.eq_3b),
                value: local.final_acc_denom.map(Into::into),
            },
            local.is_first_in_air * local.is_valid,
        );
        self.interaction_bus.receive(
            builder,
            local.proof_idx,
            InteractionsFoldingMessage {
                air_idx: local.air_idx.into(),
                interaction_idx: local.interaction_idx.into(),
                is_mult: AB::Expr::ZERO,
                idx_in_message: local.idx_in_message.into(),
                value: local.value.map(Into::into),
            },
            local.has_interactions
                * (AB::Expr::ONE - local.is_first_in_message)
                * (AB::Expr::ONE - local.is_bus_index),
        );
        self.interaction_bus.receive(
            builder,
            local.proof_idx,
            InteractionsFoldingMessage {
                air_idx: local.air_idx.into(),
                interaction_idx: local.interaction_idx.into(),
                is_mult: AB::Expr::ONE,
                idx_in_message: AB::Expr::ZERO,
                value: local.value.map(Into::into),
            },
            local.is_first_in_message * local.has_interactions,
        );

        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.beta_tidx,
            local.beta,
            local.is_valid * local.is_first,
        );

        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::NumInteractions.to_field(),
                value: (local.interaction_idx + AB::Expr::ONE) * local.has_interactions,
            },
            next.is_first_in_air * local.is_valid,
        );

        self.eq_3b_bus.receive(
            builder,
            local.proof_idx,
            Eq3bMessage {
                sort_idx: local.sort_idx,
                interaction_idx: local.interaction_idx,
                eq_3b: local.eq_3b,
            },
            local.has_interactions * local.is_first_in_message,
        );
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::batch_constraint) struct InteractionsFoldingRecord {
    value: EF,
    air_idx: usize,
    sort_idx: usize,
    interaction_idx: usize,
    node_idx: usize,
    idx_in_message: usize,
    has_interactions: bool,
    is_first_in_air: bool,
    is_last_in_air: bool,
    is_mult: bool,
    is_bus_index: bool,
}

pub(in crate::batch_constraint) struct InteractionsFoldingBlob {
    pub(in crate::batch_constraint) records: MultiProofVecVec<InteractionsFoldingRecord>,
    // (n, value), n is before lift, can be negative
    pub(in crate::batch_constraint) folded_claims: MultiProofVecVec<(isize, EF)>,
}

pub(in crate::batch_constraint) fn generate_interactions_folding_blob(
    vk: &MultiStarkVerifyingKey0V2,
    expr_evals: &MultiVecWithBounds<EF, 2>,
    eq_3b_blob: &Eq3bBlob,
    preflights: &[Preflight],
) -> InteractionsFoldingBlob {
    let l_skip = vk.params.l_skip;
    let interactions = vk
        .per_air
        .iter()
        .map(|vk| vk.symbolic_constraints.interactions.clone())
        .collect_vec();

    let mut records = MultiProofVecVec::new();
    let mut folded = MultiProofVecVec::new();
    for (pidx, preflight) in preflights.iter().enumerate() {
        let beta_tidx = preflight.proof_shape.post_tidx + 2 + D_EF;
        let beta = EF::from_base_slice(&preflight.transcript.values()[beta_tidx..beta_tidx + D_EF]);

        let eq_3bs = &eq_3b_blob.all_stacked_ids[pidx];
        let mut cur_eq3b_idx = 0;

        let vdata = &preflight.proof_shape.sorted_trace_vdata;
        for (sort_idx, (air_idx, vdata)) in vdata.iter().enumerate() {
            let n = vdata.log_height as isize - l_skip as isize;
            let inters = &interactions[*air_idx];
            let mut num_sum = EF::ZERO;
            let mut denom_sum = EF::ZERO;
            if inters.is_empty() {
                records.push(InteractionsFoldingRecord {
                    value: EF::ZERO,
                    air_idx: *air_idx,
                    sort_idx,
                    interaction_idx: 0,
                    node_idx: 0,
                    idx_in_message: 0,
                    has_interactions: false,
                    is_first_in_air: true,
                    is_last_in_air: true,
                    is_mult: false,
                    is_bus_index: false,
                });
                cur_eq3b_idx += 1;
            } else {
                // `cur_interactions_evals` in rust verifier are the list of evaluated node_claims
                // After multiplying with eq_3b and sum together we get the `num` and `denom` in rust verifier.
                for (interaction_idx, inter) in inters.iter().enumerate() {
                    let eq_3b = eq_3bs[cur_eq3b_idx].eq_mle(
                        &preflight.batch_constraint.xi,
                        vk.params.l_skip,
                        preflight.proof_shape.n_logup,
                    );
                    cur_eq3b_idx += 1;
                    records.push(InteractionsFoldingRecord {
                        value: expr_evals[[pidx, *air_idx]][inter.count],
                        air_idx: *air_idx,
                        sort_idx,
                        interaction_idx,
                        node_idx: inter.count,
                        idx_in_message: 0,
                        has_interactions: true,
                        is_first_in_air: interaction_idx == 0,
                        is_last_in_air: false,
                        is_mult: true, // for each interaction, only the first record with is_mult = true
                        is_bus_index: false,
                    });
                    num_sum += expr_evals[[pidx, *air_idx]][inter.count] * eq_3b;

                    let mut beta_pow = EF::ONE;
                    let mut cur_sum = EF::ZERO;
                    for (j, &node_idx) in inter.message.iter().enumerate() {
                        let value = expr_evals[[pidx, *air_idx]][node_idx];
                        cur_sum += beta_pow * value;
                        beta_pow *= beta;
                        records.push(InteractionsFoldingRecord {
                            value,
                            air_idx: *air_idx,
                            sort_idx,
                            interaction_idx,
                            node_idx,
                            idx_in_message: j,
                            has_interactions: true,
                            is_first_in_air: false,
                            is_last_in_air: false,
                            is_mult: false,
                            is_bus_index: false,
                        });
                    }

                    cur_sum += beta_pow * EF::from_canonical_u16(inter.bus_index + 1);
                    records.push(InteractionsFoldingRecord {
                        value: EF::from_canonical_u16(inter.bus_index + 1),
                        air_idx: *air_idx,
                        sort_idx,
                        interaction_idx,
                        node_idx: inter.bus_index as usize + 1,
                        idx_in_message: inter.message.len() + 1,
                        has_interactions: true,
                        is_first_in_air: false,
                        is_last_in_air: interaction_idx + 1 == inters.len(),
                        is_mult: false,
                        is_bus_index: true,
                    });
                    denom_sum += cur_sum * eq_3b;
                }
            }
            // Finally, this should be `interactions_evals`, minus norm_factor and eq_sharp_ns.
            folded.push((n, num_sum));
            folded.push((n, denom_sum));
        }
        folded.end_proof();
        records.end_proof();
    }
    InteractionsFoldingBlob {
        records,
        folded_claims: folded,
    }
}

#[tracing::instrument(name = "generate_trace", skip_all)]
pub(in crate::batch_constraint) fn generate_interactions_folding_trace(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &BatchConstraintBlobCpu,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let eq_3b_blob = &blob.eq_3b_blob;
    let if_blob = &blob.if_blob;

    let width = InteractionsFoldingCols::<F>::width();

    let total_height = if_blob.records.len();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let beta_tidx = preflight.proof_shape.post_tidx + 2 + D_EF;
        let beta_slice = &preflight.transcript.values()[beta_tidx..beta_tidx + D_EF];
        let records = &if_blob.records[pidx];
        let eq_3bs = &eq_3b_blob.all_stacked_ids[pidx];

        let mut is_first_in_message_indices = vec![];
        let mut cur_eq3b_idx = -1i32;
        let mut was_first_interaction_in_message = false;
        trace[cur_height * width..(cur_height + records.len()) * width]
            .chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                let record = &records[i];
                let air_idx = preflight.proof_shape.sorted_trace_vdata[record.sort_idx].0;
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.beta_tidx = F::from_canonical_usize(beta_tidx);
                cols.air_idx = F::from_canonical_usize(record.air_idx);
                cols.sort_idx = F::from_canonical_usize(record.sort_idx);
                cols.interaction_idx = F::from_canonical_usize(record.interaction_idx);
                cols.node_idx = F::from_canonical_usize(record.node_idx);
                cols.has_interactions = F::from_bool(record.has_interactions);
                cols.is_first_in_air = F::from_bool(record.is_first_in_air);
                cols.is_first_in_message = F::from_bool(record.is_mult || !record.has_interactions);
                cols.is_second_in_message = F::from_bool(was_first_interaction_in_message);
                was_first_interaction_in_message = record.is_mult;
                cols.is_bus_index = F::from_bool(record.is_bus_index);
                cols.idx_in_message = F::from_canonical_usize(record.idx_in_message);
                cols.loop_aux.is_transition[0] = F::ONE;
                cols.loop_aux.is_transition[1] = F::from_bool(!record.is_last_in_air);
                if !record.is_bus_index {
                    cols.value.copy_from_slice(
                        blob.expr_evals[[pidx, air_idx]][record.node_idx].as_base_slice(),
                    );
                } else {
                    cols.value[0] = cols.node_idx;
                }
                cols.beta.copy_from_slice(beta_slice);

                if !record.has_interactions || record.is_mult {
                    cur_eq3b_idx += 1;
                }
                if record.has_interactions {
                    cols.eq_3b.copy_from_slice(
                        eq_3bs[cur_eq3b_idx as usize]
                            .eq_mle(
                                &preflight.batch_constraint.xi,
                                vk.inner.params.l_skip,
                                preflight.proof_shape.n_logup,
                            )
                            .as_base_slice(),
                    );
                }

                if cols.is_first_in_message == F::ONE && record.has_interactions {
                    is_first_in_message_indices.push(i);
                }
            });

        // Setting `cur_sum` and final acc
        let mut cur_sum = EF::ZERO;
        let beta = EF::from_base_slice(beta_slice);
        let mut cur_acc_num = EF::ZERO;
        let mut cur_acc_denom = EF::ZERO;
        trace[cur_height * width..(cur_height + records.len()) * width]
            .chunks_exact_mut(width)
            .enumerate()
            .rev()
            .for_each(|(i, chunk)| {
                let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                // Handling cur_sum
                if cols.is_first_in_message == F::ONE {
                    cols.cur_sum.copy_from_slice(&cols.value);
                    cur_sum = EF::ZERO;
                } else {
                    cur_sum = cur_sum * beta + EF::from_base_slice(&cols.value);
                    cols.cur_sum.copy_from_slice(cur_sum.as_base_slice());
                }

                // Adding to the final acc
                if cols.is_first_in_message == F::ONE {
                    // Case 1: first in message, only accumulate the num
                    cur_acc_num +=
                        EF::from_base_slice(&cols.cur_sum) * EF::from_base_slice(&cols.eq_3b);
                    if cols.has_interactions == F::ZERO {
                        // AIR with no interactions doesn't have "second in message"
                        cur_acc_denom +=
                            EF::from_base_slice(&cols.cur_sum) * EF::from_base_slice(&cols.eq_3b);
                    }
                } else if is_first_in_message_indices.contains(&(i - 1)) {
                    // Case 2: second in message, accumulate the denom
                    cur_acc_denom +=
                        EF::from_base_slice(&cols.cur_sum) * EF::from_base_slice(&cols.eq_3b);
                }
                cols.final_acc_num
                    .copy_from_slice(cur_acc_num.as_base_slice());
                cols.final_acc_denom
                    .copy_from_slice(cur_acc_denom.as_base_slice());

                // Reset per AIR
                if cols.is_first_in_air == F::ONE {
                    cur_acc_num = EF::ZERO;
                    cur_acc_denom = EF::ZERO;
                }
            });

        // Setting is_first and is_last for this proof
        {
            let cols: &mut InteractionsFoldingCols<_> =
                trace[cur_height * width..(cur_height + 1) * width].borrow_mut();
            cols.is_first = F::ONE;
        }
        cur_height += records.len();
        {
            let cols: &mut InteractionsFoldingCols<_> =
                trace[(cur_height - 1) * width..cur_height * width].borrow_mut();
            cols.is_last = F::ONE;
            cols.loop_aux.is_transition[0] = F::ZERO;
        }
    }
    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut InteractionsFoldingCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
            cols.is_first_in_air = F::ONE;
            cols.is_first_in_message = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
