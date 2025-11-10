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
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, Eq3bBus, Eq3bMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        MultiProofVecVec, base_to_ext, ext_field_add, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_one_minus,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Eq3bColumns<T> {
    is_valid: T,
    is_first: T,
    proof_idx: T,

    sort_idx: T,
    interaction_idx: T,

    n_lift: T,
    two_to_the_n_lift: T,
    n: T,
    hypercube_volume: T, // 2^n
    n_at_least_n_lift: T,

    is_first_in_air: T,
    is_first_in_interaction: T,

    idx: T,         // stacked_idx >> l_skip, restored bit by bit
    running_idx: T, // the current stacked_idx >> l_skip
    nth_bit: T,     // TODO: can we derive it from local.idx, next.idx and hypercube volume?

    loop_aux: NestedForLoopAuxCols<T, 2>,

    xi: [T; D_EF],
    eq: [T; D_EF],
}

pub struct Eq3bAir {
    pub eq_3b_bus: Eq3bBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for Eq3bAir {}
impl<F> PartitionedBaseAir<F> for Eq3bAir {}

impl<F> BaseAir<F> for Eq3bAir {
    fn width(&self) -> usize {
        Eq3bColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for Eq3bAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &Eq3bColumns<AB::Var> = (*local).borrow();
        let next: &Eq3bColumns<AB::Var> = (*next).borrow();

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
                            local.is_first_in_interaction,
                        ],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx, next.sort_idx, next.interaction_idx],
                        is_first: [
                            next.is_first,
                            next.is_first_in_air,
                            next.is_first_in_interaction,
                        ],
                    }
                    .map_into(),
                ),
                local.loop_aux.map_into(),
            ),
        );

        builder.assert_bool(local.n_at_least_n_lift);
        builder.assert_bool(local.nth_bit);

        let within_one_air = not(next.is_first_in_air);
        let within_one_interaction = not(next.is_first_in_interaction);

        // =============================== n consistency ==================================
        builder
            .when(local.is_first_in_interaction)
            .assert_zero(local.n);
        builder
            .when(local.is_first_in_interaction)
            .when(local.is_valid)
            .assert_one(local.hypercube_volume);
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.n_lift, local.n_lift);
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.two_to_the_n_lift, local.two_to_the_n_lift);
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.n, local.n + AB::Expr::ONE);
        builder.when(within_one_interaction.clone()).assert_eq(
            next.hypercube_volume,
            local.hypercube_volume * AB::Expr::TWO,
        );
        // n_at_least_n_lift is nondecreasing
        builder
            .when(within_one_interaction.clone())
            .when(local.n_at_least_n_lift)
            .assert_one(next.n_at_least_n_lift);
        // it's always 1 in the end
        builder
            .when(next.is_first_in_interaction)
            .when(local.is_valid)
            .assert_one(local.n_at_least_n_lift);

        // Either there is a moment where it switches from 0 to 1, then it's when n = n_lift
        builder
            .when(not(local.n_at_least_n_lift))
            .when(next.n_at_least_n_lift)
            .assert_eq(next.n, next.n_lift);
        builder
            .when(not(local.n_at_least_n_lift))
            .when(next.n_at_least_n_lift)
            .assert_eq(next.hypercube_volume, next.two_to_the_n_lift);
        // Or it's 1 from the beginning, in which case n_lift = 0
        builder
            .when(local.is_first_in_interaction)
            .when(local.n_at_least_n_lift)
            .assert_zero(local.n_lift);
        builder
            .when(local.is_first_in_interaction)
            .when(local.n_at_least_n_lift)
            .assert_one(local.two_to_the_n_lift);

        builder.when(within_one_air).assert_eq(
            next.running_idx,
            local.running_idx + next.is_first_in_interaction * local.two_to_the_n_lift,
        );

        // =========================== Xi and product consistency =============================
        // Boundary conditions
        assert_array_eq(
            &mut builder.when(local.is_valid * local.is_first),
            local.eq,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        builder
            .when(local.is_first_in_interaction)
            .assert_zero(local.idx);
        builder.when(local.is_first).assert_zero(local.running_idx);
        builder
            .when(LoopSubAir::local_is_last(
                next.is_valid,
                next.is_first_in_interaction,
            ))
            .assert_eq(local.idx, local.running_idx);

        // If n is less than n_lift, assert that eq doesn't change
        assert_array_eq(
            &mut builder
                .when(local.is_valid)
                .when(not(local.n_at_least_n_lift)),
            local.eq,
            next.eq,
        );
        // Within transition, idx increases by nth_bit * hypercube_volume
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.idx, local.idx + local.nth_bit * local.hypercube_volume);
        // It can't increase if n < n_lift
        builder
            .when(not(local.n_at_least_n_lift))
            .assert_zero(local.nth_bit);
        // When transition, eq multiplies correspondingly
        assert_array_eq(
            &mut builder.when(within_one_interaction.clone()),
            next.eq,
            ext_field_multiply(
                local.eq,
                ext_field_add::<AB::Expr>(
                    ext_field_multiply_scalar(local.xi, local.nth_bit),
                    ext_field_multiply_scalar::<AB::Expr>(
                        ext_field_one_minus(local.xi),
                        AB::Expr::ONE - local.nth_bit,
                    ),
                ),
            ),
        );

        self.batch_constraint_conductor_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.n + AB::Expr::from_canonical_usize(self.l_skip),
                value: local.xi.map(|x| x.into()),
            },
            local.n_at_least_n_lift * within_one_interaction,
        );

        // TODO constrain that air with this sort_idx has that n_lift,
        // TODO constrain that this pidx has that n_logup

        self.eq_3b_bus.send(
            builder,
            local.proof_idx,
            Eq3bMessage {
                sort_idx: local.sort_idx,
                interaction_idx: local.interaction_idx,
                eq_3b: local.eq,
            },
            next.is_first_in_interaction * local.is_valid,
        );
    }
}

pub(crate) struct StackedIdxRecord {
    sort_idx: usize,
    interaction_idx: usize,
    stacked_idx: usize,
    n_lift: usize,
    is_last_in_air: bool,
}

impl StackedIdxRecord {
    pub fn eq_mle(&self, xi: &[EF], l_skip: usize, n_logup: usize) -> EF {
        xi[l_skip + self.n_lift..l_skip + n_logup]
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if self.stacked_idx & (1 << (l_skip + self.n_lift + i)) > 0 {
                    x
                } else {
                    EF::ONE - x
                }
            })
            .fold(EF::ONE, |acc, x| acc * x)
    }
}

pub struct Eq3bBlob {
    pub(crate) all_stacked_ids: MultiProofVecVec<StackedIdxRecord>,
}

impl Eq3bBlob {
    fn new() -> Self {
        Self {
            all_stacked_ids: MultiProofVecVec::new(),
        }
    }
}

pub(crate) fn generate_eq_3b_blob(
    vk: &MultiStarkVerifyingKeyV2,
    preflights: &[Preflight],
) -> Eq3bBlob {
    let l_skip = vk.inner.params.l_skip;
    let mut blob = Eq3bBlob::new();
    for preflight in preflights.iter() {
        let mut row_idx = 0;
        let n_logup = preflight.proof_shape.n_logup;
        for (sort_idx, (air_idx, vdata)) in
            preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
        {
            let n_lift = vdata.log_height.saturating_sub(l_skip);
            let num_interactions = vk.inner.per_air[*air_idx].num_interactions();
            for i in 0..num_interactions {
                blob.all_stacked_ids.push(StackedIdxRecord {
                    sort_idx,
                    interaction_idx: i,
                    stacked_idx: row_idx,
                    n_lift,
                    is_last_in_air: i + 1 == num_interactions,
                });
                row_idx += 1 << (l_skip + n_lift);
                if row_idx == 1 << (l_skip + n_logup) {
                    row_idx = 0;
                }
            }
        }
        blob.all_stacked_ids.end_proof();
    }
    blob
}

pub(crate) fn generate_eq_3b_trace(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &Eq3bBlob,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = Eq3bColumns::<F>::width();
    let l_skip = vk.inner.params.l_skip;
    let heights = preflights
        .iter()
        .map(|p| (2 << p.proof_shape.n_logup) - 1)
        .collect_vec();
    let total_height = heights.iter().sum::<usize>();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let xi = &preflight.batch_constraint.xi;
        let n_logup = preflight.proof_shape.n_logup;

        let stacked_ids = &blob.all_stacked_ids[pidx];
        let one_height = n_logup + 1;
        trace[cur_height * width..(cur_height + one_height * stacked_ids.len()) * width]
            .par_chunks_exact_mut(one_height * width)
            .enumerate()
            .for_each(|(j, chunks)| {
                let record = &stacked_ids[j];
                let shifted_idx = record.stacked_idx >> l_skip;

                let mut cur_eq = EF::ONE;
                chunks
                    .chunks_exact_mut(width)
                    .enumerate()
                    .for_each(|(n, chunk)| {
                        let cols: &mut Eq3bColumns<_> = chunk.borrow_mut();
                        cols.is_valid = F::ONE;
                        cols.is_first = F::from_bool(j == 0 && n == 0);
                        cols.proof_idx = F::from_canonical_usize(pidx);

                        cols.sort_idx = F::from_canonical_usize(record.sort_idx);
                        cols.interaction_idx = F::from_canonical_usize(record.interaction_idx);
                        cols.n_lift = F::from_canonical_usize(record.n_lift);
                        cols.two_to_the_n_lift = F::from_canonical_usize(1 << record.n_lift);
                        cols.n = F::from_canonical_usize(n);
                        cols.n_at_least_n_lift = F::from_bool(n >= record.n_lift);
                        cols.hypercube_volume = F::from_canonical_usize(1 << n);
                        cols.is_first_in_air = F::from_bool(record.interaction_idx == 0 && n == 0);
                        cols.is_first_in_interaction = F::from_bool(n == 0);
                        cols.idx = F::from_canonical_usize(shifted_idx & ((1 << n) - 1));
                        cols.running_idx = F::from_canonical_usize(shifted_idx);
                        let nth_bit = (shifted_idx & (1 << n)) > 0;
                        cols.nth_bit = F::from_bool(nth_bit);
                        cols.loop_aux.is_transition[0] =
                            F::from_bool(j + 1 < stacked_ids.len() || n < n_logup);
                        cols.loop_aux.is_transition[1] =
                            F::from_bool(!record.is_last_in_air || n < n_logup);
                        let xi = if (record.n_lift..n_logup).contains(&n) {
                            xi[l_skip + n]
                        } else if nth_bit {
                            EF::ONE
                        } else {
                            EF::ZERO
                        };
                        cols.xi.copy_from_slice(xi.as_base_slice());
                        cols.eq.copy_from_slice(cur_eq.as_base_slice());
                        cur_eq *= if nth_bit { xi } else { EF::ONE - xi };
                    });
            });
        cur_height += one_height * stacked_ids.len();
    }

    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut Eq3bColumns<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_first_in_air = F::ONE;
            cols.is_first_in_interaction = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
