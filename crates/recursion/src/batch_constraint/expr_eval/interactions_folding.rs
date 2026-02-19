use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        bus::{
            Eq3bBus, Eq3bMessage, ExpressionClaimBus, ExpressionClaimMessage,
            InteractionsFoldingBus, InteractionsFoldingMessage,
        },
        eq_airs::Eq3bBlob,
        BatchConstraintBlobCpu,
    },
    bus::{AirShapeBus, AirShapeBusMessage, AirShapeProperty, TranscriptBus},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{
        assert_zeros, ext_field_add, ext_field_multiply, pow_tidx_count, MultiProofVecVec,
        MultiVecWithBounds,
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
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

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
        builder.assert_bool(local.is_bus_index);
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
        // If it's last in the interaction and the row is valid, then it's the bus index
        builder
            .when(next.is_first_in_message)
            .when(local.has_interactions)
            .assert_one(local.is_bus_index);

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
        self.interaction_bus.receive(
            builder,
            local.proof_idx,
            InteractionsFoldingMessage {
                air_idx: local.air_idx.into(),
                interaction_idx: local.interaction_idx.into(),
                is_mult: AB::Expr::ZERO,
                idx_in_message: AB::Expr::NEG_ONE,
                value: local.value.map(Into::into),
            },
            local.is_bus_index,
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

pub(crate) struct InteractionsFoldingBlob {
    pub(in crate::batch_constraint) records: MultiProofVecVec<InteractionsFoldingRecord>,
    // (n, value), n is before lift, can be negative
    pub(in crate::batch_constraint) folded_claims: MultiProofVecVec<(isize, EF)>,
}

impl InteractionsFoldingBlob {
    pub fn new(
        vk: &MultiStarkVerifyingKey0<BabyBearPoseidon2Config>,
        expr_evals: &MultiVecWithBounds<EF, 2>,
        eq_3b_blob: &Eq3bBlob,
        preflights: &[&Preflight],
    ) -> Self {
        let l_skip = vk.params.l_skip;
        let interactions = vk
            .per_air
            .iter()
            .map(|vk| vk.symbolic_constraints.interactions.clone())
            .collect_vec();

        let logup_pow_offset = pow_tidx_count(vk.params.logup.pow_bits);
        let mut records = MultiProofVecVec::new();
        let mut folded = MultiProofVecVec::new();
        for (pidx, preflight) in preflights.iter().enumerate() {
            let beta_tidx = preflight.proof_shape.post_tidx + logup_pow_offset + D_EF;
            let beta = EF::from_basis_coefficients_slice(
                &preflight.transcript.values()[beta_tidx..beta_tidx + D_EF],
            )
            .unwrap();

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
                    // `cur_interactions_evals` in rust verifier are the list of evaluated
                    // node_claims After multiplying with eq_3b and sum together we get the
                    // `num` and `denom` in rust verifier.
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
                            is_mult: true, /* for each interaction, only the first record with
                                            * is_mult = true */
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

                        cur_sum += beta_pow * EF::from_u16(inter.bus_index + 1);
                        records.push(InteractionsFoldingRecord {
                            value: EF::from_u16(inter.bus_index + 1),
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
        Self {
            records,
            folded_claims: folded,
        }
    }
}

pub struct InteractionsFoldingTraceGenerator;

impl RowMajorChip<F> for InteractionsFoldingTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a BatchConstraintBlobCpu,
        &'a [&'a Preflight],
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, blob, preflights) = ctx;
        let eq_3b_blob = &blob.common_blob.eq_3b_blob;
        let if_blob = blob.if_blob.as_ref().unwrap();

        let width = InteractionsFoldingCols::<F>::width();

        let total_height = if_blob.records.len();
        let padding_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padding_height * width];

        let logup_pow_offset = pow_tidx_count(vk.inner.params.logup.pow_bits);
        let mut cur_height = 0;
        for (pidx, preflight) in preflights.iter().enumerate() {
            let beta_tidx = preflight.proof_shape.post_tidx + logup_pow_offset + D_EF;
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
                    cols.proof_idx = F::from_usize(pidx);
                    cols.beta_tidx = F::from_usize(beta_tidx);
                    cols.air_idx = F::from_usize(record.air_idx);
                    cols.sort_idx = F::from_usize(record.sort_idx);
                    cols.interaction_idx = F::from_usize(record.interaction_idx);
                    cols.node_idx = F::from_usize(record.node_idx);
                    cols.has_interactions = F::from_bool(record.has_interactions);
                    cols.is_first_in_air = F::from_bool(record.is_first_in_air);
                    cols.is_first_in_message =
                        F::from_bool(record.is_mult || !record.has_interactions);
                    cols.is_second_in_message = F::from_bool(was_first_interaction_in_message);
                    was_first_interaction_in_message = record.is_mult;
                    cols.is_bus_index = F::from_bool(record.is_bus_index);
                    cols.idx_in_message = F::from_usize(record.idx_in_message);
                    cols.loop_aux.is_transition[0] = F::ONE;
                    cols.loop_aux.is_transition[1] = F::from_bool(!record.is_last_in_air);
                    if !record.is_bus_index {
                        cols.value.copy_from_slice(
                            blob.common_blob.expr_evals[[pidx, air_idx]][record.node_idx]
                                .as_basis_coefficients_slice(),
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
                                .as_basis_coefficients_slice(),
                        );
                    }

                    if cols.is_first_in_message == F::ONE && record.has_interactions {
                        is_first_in_message_indices.push(i);
                    }
                });

            // Setting `cur_sum` and final acc
            let mut cur_sum = EF::ZERO;
            let beta = EF::from_basis_coefficients_slice(beta_slice).unwrap();
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
                        cur_sum = cur_sum * beta
                            + EF::from_basis_coefficients_slice(&cols.value).unwrap();
                        cols.cur_sum
                            .copy_from_slice(cur_sum.as_basis_coefficients_slice());
                    }

                    // Adding to the final acc
                    if cols.is_first_in_message == F::ONE {
                        // Case 1: first in message, only accumulate the num
                        cur_acc_num += EF::from_basis_coefficients_slice(&cols.cur_sum).unwrap()
                            * EF::from_basis_coefficients_slice(&cols.eq_3b).unwrap();
                        if cols.has_interactions == F::ZERO {
                            // AIR with no interactions doesn't have "second in message"
                            cur_acc_denom += EF::from_basis_coefficients_slice(&cols.cur_sum)
                                .unwrap()
                                * EF::from_basis_coefficients_slice(&cols.eq_3b).unwrap();
                        }
                    } else if is_first_in_message_indices.contains(&(i - 1)) {
                        // Case 2: second in message, accumulate the denom
                        cur_acc_denom += EF::from_basis_coefficients_slice(&cols.cur_sum).unwrap()
                            * EF::from_basis_coefficients_slice(&cols.eq_3b).unwrap();
                    }
                    cols.final_acc_num
                        .copy_from_slice(cur_acc_num.as_basis_coefficients_slice());
                    cols.final_acc_denom
                        .copy_from_slice(cur_acc_denom.as_basis_coefficients_slice());

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
                cols.proof_idx = F::from_usize(preflights.len() + i);
                cols.is_first = F::ONE;
                cols.is_last = F::ONE;
                cols.is_first_in_air = F::ONE;
                cols.is_first_in_message = F::ONE;
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}

#[cfg(feature = "cuda")]
pub(in crate::batch_constraint) mod cuda {
    use openvm_circuit_primitives::cuda_abi::UInt2;
    use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        batch_constraint::cuda_abi::{
            interactions_folding_tracegen, interactions_folding_tracegen_temp_bytes, AffineFpExt,
            FpExtWithTidx, InteractionRecord,
        },
        cuda::{preflight::PreflightGpu, vk::VerifyingKeyGpu},
        tracegen::ModuleChip,
    };

    pub struct InteractionsFoldingBlobGpu {
        // Per proof, per AIR, per interaction, per index
        pub values: Vec<Vec<Vec<Vec<EF>>>>,
        // Per valid row
        pub node_idxs: Vec<u32>,
        // Per proof, per interaction
        pub interaction_records: Vec<Vec<InteractionRecord>>,
        // Per proof
        pub interactions_folding_per_proof: Vec<FpExtWithTidx>,
        // For compatibility with CPU tracegen
        pub folded_claims: MultiProofVecVec<(isize, EF)>,
    }

    impl InteractionsFoldingBlobGpu {
        pub fn new(
            vk: &VerifyingKeyGpu,
            expr_evals: &MultiVecWithBounds<EF, 2>,
            eq_3b_blob: &Eq3bBlob,
            preflights: &[PreflightGpu],
        ) -> Self {
            let l_skip = vk.system_params.l_skip;
            let interactions = vk
                .cpu
                .inner
                .per_air
                .iter()
                .map(|vk| vk.symbolic_constraints.interactions.clone())
                .collect_vec();

            let mut global_current_row = 0;

            let mut values = Vec::with_capacity(preflights.len());
            let mut node_idxs = vec![];
            let mut interaction_records = Vec::with_capacity(preflights.len());
            let mut interactions_folding_per_proof = Vec::with_capacity(preflights.len());
            let mut folded_claims = MultiProofVecVec::new();
            let logup_pow_offset = pow_tidx_count(vk.cpu.inner.params.logup.pow_bits);

            for (pidx, preflight) in preflights.iter().enumerate() {
                let beta_tidx = preflight.proof_shape.post_tidx + logup_pow_offset + D_EF;
                let beta = EF::from_basis_coefficients_slice(
                    &preflight.cpu.transcript.values()[beta_tidx..beta_tidx + D_EF],
                )
                .unwrap();

                let eq_3bs = &eq_3b_blob.all_stacked_ids[pidx];
                let mut cur_eq3b_idx = 0;

                let vdata = &preflight.cpu.proof_shape.sorted_trace_vdata;
                let mut proof_values = Vec::with_capacity(vdata.len());
                let mut proof_interaction_records = vec![];

                for (air_idx, vdata) in vdata {
                    let n = vdata.log_height as isize - l_skip as isize;
                    let inters = &interactions[*air_idx];

                    let mut num_sum = EF::ZERO;
                    let mut denom_sum = EF::ZERO;
                    let mut air_values = Vec::with_capacity(inters.len());

                    if inters.is_empty() {
                        // Note differs from what is written in CPU blob generation, but matches
                        // tracegen
                        air_values.push(vec![expr_evals[[pidx, *air_idx]][0]]);
                        node_idxs.push(0);
                        proof_interaction_records.push(InteractionRecord {
                            interaction_num_rows: 1,
                            global_start_row: global_current_row,
                            stacked_idx: 0,
                        });
                        global_current_row += 1;
                        cur_eq3b_idx += 1;
                    } else {
                        for inter in inters {
                            let stacked_idx_record = eq_3bs[cur_eq3b_idx];
                            let eq_3b = stacked_idx_record.eq_mle(
                                &preflight.cpu.batch_constraint.xi,
                                l_skip,
                                preflight.proof_shape.n_logup,
                            );
                            cur_eq3b_idx += 1;
                            num_sum += expr_evals[[pidx, *air_idx]][inter.count] * eq_3b;

                            let message_len = inter.message.len();
                            let interaction_num_rows = message_len as u32 + 2;
                            proof_interaction_records.push(InteractionRecord {
                                interaction_num_rows,
                                global_start_row: global_current_row,
                                stacked_idx: stacked_idx_record.stacked_idx,
                            });
                            global_current_row += interaction_num_rows;

                            let mut interaction_values = Vec::with_capacity(message_len + 2);
                            interaction_values.push(expr_evals[[pidx, *air_idx]][inter.count]);
                            node_idxs.push(inter.count as u32);

                            let mut beta_pow = EF::ONE;
                            let mut cur_sum = EF::ZERO;
                            for &node_idx in &inter.message {
                                let value = expr_evals[[pidx, *air_idx]][node_idx];
                                cur_sum += beta_pow * value;
                                beta_pow *= beta;
                                interaction_values.push(value);
                                node_idxs.push(node_idx as u32);
                            }

                            let bus_value = EF::from_u16(inter.bus_index + 1);
                            cur_sum += beta_pow * bus_value;
                            interaction_values.push(bus_value);
                            node_idxs.push(inter.bus_index as u32 + 1);
                            denom_sum += cur_sum * eq_3b;

                            air_values.push(interaction_values);
                        }
                    }

                    proof_values.push(air_values);
                    folded_claims.push((n, num_sum));
                    folded_claims.push((n, denom_sum));
                }

                values.push(proof_values);
                interaction_records.push(proof_interaction_records);
                interactions_folding_per_proof.push(FpExtWithTidx {
                    value: beta,
                    tidx: beta_tidx as u32,
                });
                folded_claims.end_proof();
            }

            Self {
                values,
                node_idxs,
                interaction_records,
                interactions_folding_per_proof,
                folded_claims,
            }
        }
    }

    impl ModuleChip<GpuBackend> for InteractionsFoldingTraceGenerator {
        type Ctx<'a> = (
            &'a VerifyingKeyGpu,
            &'a [PreflightGpu],
            &'a InteractionsFoldingBlobGpu,
        );

        #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            let (child_vk, preflights_gpu, blob) = ctx;

            let num_airs = preflights_gpu
                .iter()
                .map(|preflight| preflight.cpu.proof_shape.sorted_trace_vdata.len() as u32)
                .collect_vec();
            let n_logups = preflights_gpu
                .iter()
                .map(|p| p.proof_shape.n_logup as u32)
                .collect_vec();
            let mut num_valid_rows = 0u32;

            let mut row_bounds = Vec::with_capacity(preflights_gpu.len());
            let mut air_interaction_bounds = Vec::with_capacity(preflights_gpu.len());
            let mut interaction_row_bounds = Vec::with_capacity(preflights_gpu.len());

            let expected_num_valid_rows = blob.node_idxs.len();
            let mut idx_keys = Vec::with_capacity(expected_num_valid_rows);
            let mut flat_values = Vec::with_capacity(expected_num_valid_rows);

            for (proof_idx, proof_values) in blob.values.iter().enumerate() {
                let mut proof_num_rows = 0;
                let mut proof_num_interactions = 0;
                let mut proof_air_interaction_bounds =
                    Vec::with_capacity(num_airs[proof_idx] as usize);
                let mut proof_interaction_row_bounds = vec![];

                for air_values in proof_values {
                    for interaction_values in air_values {
                        let global_interaction_idx = proof_num_interactions;
                        proof_num_interactions += 1;
                        for v in interaction_values {
                            flat_values.push(*v);
                            idx_keys.push(UInt2 {
                                x: proof_idx as u32,
                                y: global_interaction_idx,
                            });
                        }
                        proof_num_rows += interaction_values.len() as u32;
                        proof_interaction_row_bounds.push(proof_num_rows);
                    }
                    proof_air_interaction_bounds.push(proof_num_interactions);
                }

                num_valid_rows += proof_num_rows;
                row_bounds.push(num_valid_rows);
                air_interaction_bounds.push(proof_air_interaction_bounds.to_device().unwrap());
                interaction_row_bounds.push(proof_interaction_row_bounds.to_device().unwrap());
            }

            assert_eq!(num_valid_rows as usize, expected_num_valid_rows);

            let records = blob
                .interaction_records
                .iter()
                .map(|records| records.to_device().unwrap())
                .collect_vec();
            let xis = preflights_gpu
                .iter()
                .map(|preflight| preflight.cpu.batch_constraint.xi.to_device().unwrap())
                .collect_vec();

            let height = if let Some(height) = required_height {
                if height < num_valid_rows as usize {
                    return None;
                }
                height
            } else {
                (num_valid_rows as usize).next_power_of_two()
            };
            let width = InteractionsFoldingCols::<F>::width();
            let d_trace = DeviceMatrix::<F>::with_capacity(height, width);

            let d_idx_keys = idx_keys.to_device().unwrap();
            let d_values = flat_values.to_device().unwrap();
            let d_node_idxs = blob.node_idxs.to_device().unwrap();
            let d_cur_sum_evals = DeviceBuffer::<AffineFpExt>::with_capacity(d_values.len());

            let d_air_interaction_bounds = air_interaction_bounds
                .iter()
                .map(|b| b.as_ptr())
                .collect_vec();
            let d_interaction_row_bounds = interaction_row_bounds
                .iter()
                .map(|b| b.as_ptr())
                .collect_vec();
            let d_sorted_trace_vdata = preflights_gpu
                .iter()
                .map(|preflight| preflight.proof_shape.sorted_trace_heights.as_ptr())
                .collect_vec();
            let d_records = records.iter().map(|b| b.as_ptr()).collect_vec();
            let d_xis = xis.iter().map(|b| b.as_ptr()).collect_vec();

            let d_per_proof = blob.interactions_folding_per_proof.to_device().unwrap();

            unsafe {
                let temp_bytes = interactions_folding_tracegen_temp_bytes(
                    d_trace.buffer(),
                    height,
                    &d_idx_keys,
                    &d_cur_sum_evals,
                    num_valid_rows,
                )
                .unwrap();
                let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                interactions_folding_tracegen(
                    d_trace.buffer(),
                    height,
                    width,
                    &d_idx_keys,
                    &d_cur_sum_evals,
                    &d_values,
                    &d_node_idxs,
                    &row_bounds,
                    d_air_interaction_bounds,
                    d_interaction_row_bounds,
                    d_sorted_trace_vdata,
                    d_records,
                    d_xis,
                    &d_per_proof,
                    &num_airs,
                    &n_logups,
                    preflights_gpu.len() as u32,
                    num_valid_rows,
                    child_vk.system_params.l_skip as u32,
                    &d_temp_buffer,
                    temp_bytes,
                )
                .unwrap();
            }
            Some(AirProvingContext::simple_no_pis(d_trace))
        }
    }
}
