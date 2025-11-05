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
    D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, prover::poly::evals_eq_hypercubes,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, EqMleBus, EqMleMessage,
    },
    bus::TranscriptBus,
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{MultiProofVecVec, base_to_ext, ext_field_multiply, ext_field_one_minus},
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqMleColumns<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    n: T,
    is_l_skip: T,
    hypercube_volume: T, // 2^n
    xi_tidx: T,

    idx: T,
    idx_is_zero: T,
    num_traces: T,

    xi: [T; D_EF],
    eq: [T; D_EF],
}

pub struct EqMleAir {
    pub eq_mle_bus: EqMleBus, // We use this bus for both inner and outer communications
    // TODO(AG): ^^^ this probably breaks soundness, fix
    pub transcript_bus: TranscriptBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqMleAir {}
impl<F> PartitionedBaseAir<F> for EqMleAir {}

impl<F> BaseAir<F> for EqMleAir {
    fn width(&self) -> usize {
        EqMleColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqMleAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqMleColumns<AB::Var> = (*local).borrow();
        let next: &EqMleColumns<AB::Var> = (*next).borrow();

        // Summary:
        // - idx consistency: TODO to increase the nested-loop sub-AIR dimension; the first valid
        //   row forces `idx_is_zero`, `idx = 0`, and validity; every step either keeps `idx` at
        //   zero or increments it by one, invalid rows must have `idx = 0`, wrapping to zero implies
        //   the previous value hit `hypercube_volume` and sets `next.idx_is_zero`; additionally, the
        //   first transition after the header is forced to drop `idx` to zero to ensure the layer
        //   size condition.
        // - is_l_skip consistency: the flag is boolean, cleared on invalid rows, asserted on the
        //   last layer, preserved while the index keeps advancing, and cleared when a new layer
        //   begins.
        // - Xi and product consistency: boundary `eq` equals one on the first row, dropping `idx`
        //   decrements `xi_tidx` by `D_EF`, and the buses orchestrate `xi` queries plus `eq_mle`
        //   sends/receives so products update via `xi` and `1 - xi` factors.

        type LoopSubAir = NestedForLoopSubAir<1, 0>;
        LoopSubAir {}.eval(
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

        // =========================== idx consistency =============================
        // TODO: just increase the dimension of nested loop subair?
        builder
            .when(local.is_valid * local.is_first)
            .assert_one(local.idx_is_zero);
        builder.when(local.idx_is_zero).assert_zero(local.idx);
        builder.when(local.idx_is_zero).assert_one(local.is_valid);

        // Either idx becomes zero, or increases by one.
        builder.assert_zero(next.idx * (next.idx - local.idx - AB::Expr::ONE));
        // idx is always zero on invalid rows
        builder.when(not(local.is_valid)).assert_zero(local.idx);
        // If becomes zero, then it would have become hypercube_volume
        builder
            .when(next.idx - local.idx - AB::Expr::ONE)
            .when(local.is_valid)
            .assert_eq(local.idx + AB::Expr::ONE, local.hypercube_volume);
        // and, additionally, would set idx_is_zero, if next was even valid
        builder
            .when(next.idx - local.idx - AB::Expr::ONE)
            .when(next.is_valid)
            .assert_one(next.idx_is_zero);

        // Important: we need to enforce dropping idx to zero if it is going to become hypercube_volume.
        // We can leave this to inner interactions, but then we need to guarantee that the first layer
        // has size 1.
        builder
            .when(next.is_valid * local.is_first)
            .assert_one(next.idx_is_zero);

        // =========================== is_l_skip consistency ==============================
        builder
            .when(not(local.is_valid))
            .assert_zero(local.is_l_skip);
        builder.assert_bool(local.is_l_skip);
        // is_l_skip must only be set on the last layer
        builder
            .when(local.is_valid * local.is_last)
            .assert_one(local.is_l_skip);
        builder
            .when(not(next.idx_is_zero) * next.is_valid)
            .assert_eq(local.is_l_skip, next.is_l_skip);
        builder
            .when(next.idx_is_zero * not(next.is_first))
            .assert_zero(local.is_l_skip);

        // =========================== Xi and product consistency =============================
        // Boundary conditions
        assert_array_eq(
            &mut builder.when(local.is_valid * local.is_first),
            local.eq,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        // When we drop idx, xi_idx decreases
        builder
            .when(next.idx_is_zero * not(next.is_first))
            .assert_eq(
                local.xi_tidx - next.xi_tidx,
                AB::Expr::from_canonical_usize(D_EF),
            );

        self.batch_constraint_conductor_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.n + AB::Expr::from_canonical_usize(self.l_skip) - AB::Expr::ONE,
                value: local.xi.map(|x| x.into()),
            },
            local.idx_is_zero * (AB::Expr::ONE - local.is_l_skip),
        );

        self.eq_mle_bus.receive(
            builder,
            local.proof_idx,
            EqMleMessage {
                n: local.n,
                idx: local.idx,
                eq_mle: local.eq,
            },
            local.is_valid * (AB::Expr::ONE - local.is_first),
        );
        self.eq_mle_bus.send(
            builder,
            local.proof_idx,
            EqMleMessage {
                n: local.n - AB::Expr::ONE,
                idx: local.idx * AB::Expr::TWO + AB::Expr::ONE,
                eq_mle: ext_field_multiply::<AB::Expr>(local.eq, local.xi),
            },
            local.is_valid - local.is_l_skip,
        );
        self.eq_mle_bus.send(
            builder,
            local.proof_idx,
            EqMleMessage {
                n: local.n - AB::Expr::ONE,
                idx: local.idx * AB::Expr::TWO,
                eq_mle: ext_field_multiply::<AB::Expr>(local.eq, ext_field_one_minus(local.xi)),
            },
            local.is_valid - local.is_l_skip,
        );

        self.eq_mle_bus.send(
            builder,
            local.proof_idx,
            EqMleMessage {
                n: local.n.into(),
                idx: local.idx * AB::Expr::from_canonical_usize(1 << self.l_skip),
                eq_mle: local.eq.map(|x| x.into()),
            },
            local.num_traces,
        );
    }
}

pub struct EqMleBlob {
    pub(crate) products: MultiProofVecVec<EF>,
    pub(crate) all_stacked_ids: MultiProofVecVec<(u32, u32)>,
}

impl EqMleBlob {
    fn new() -> Self {
        Self {
            products: MultiProofVecVec::new(),
            all_stacked_ids: MultiProofVecVec::new(),
        }
    }
}

pub(crate) fn generate_eq_mle_blob(
    vk: &MultiStarkVerifyingKeyV2,
    preflights: &[Preflight],
) -> EqMleBlob {
    let l_skip = vk.inner.params.l_skip;
    let mut blob = EqMleBlob::new();
    for preflight in preflights.iter() {
        blob.products.extend(evals_eq_hypercubes(
            preflight.proof_shape.n_logup,
            preflight.batch_constraint.xi[l_skip..l_skip + preflight.proof_shape.n_logup]
                .iter()
                .rev(),
        ));
        blob.products.end_proof();

        let mut row_idx = 0;
        let n_logup = preflight.proof_shape.n_logup;
        for (air_idx, vdata) in preflight.proof_shape.sorted_trace_vdata.iter() {
            let n = vdata.hypercube_dim;
            let num_interactions = vk.inner.per_air[*air_idx].num_interactions();
            for _ in 0..num_interactions {
                blob.all_stacked_ids.push((n as u32, row_idx));
                row_idx += 1 << (l_skip + n);
                if row_idx == 1 << (l_skip + n_logup) {
                    row_idx = 0;
                }
            }
        }
        blob.all_stacked_ids.end_proof();
    }
    blob
}

pub(crate) fn generate_eq_mle_trace(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &EqMleBlob,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = EqMleColumns::<F>::width();
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
        let gkr_post_tidx = preflight.gkr.post_tidx;
        let n_logup = preflight.proof_shape.n_logup;
        let products = &blob.products[pidx];

        for i in 0..=n_logup {
            let height = 1 << i;
            trace[cur_height * width..(cur_height + height) * width]
                .par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(j, chunk)| {
                    let cols: &mut EqMleColumns<_> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(i == 0 && j == 0);
                    cols.is_last = F::from_bool(i == n_logup && j + 1 == height);
                    cols.proof_idx = F::from_canonical_usize(pidx);
                    cols.n = F::from_canonical_usize(n_logup - i);
                    if i == n_logup {
                        cols.is_l_skip = F::ONE;
                    } else {
                        cols.xi
                            .copy_from_slice(xi[l_skip + n_logup - 1 - i].as_base_slice());
                    }
                    cols.hypercube_volume = F::from_canonical_usize(height);
                    cols.xi_tidx = F::from_canonical_usize(gkr_post_tidx + (n_logup - i) * D_EF);
                    cols.idx = F::from_canonical_usize(j);
                    cols.idx_is_zero = F::from_bool(j == 0);
                    cols.eq
                        .copy_from_slice(products[(1 << i) - 1 + j].as_base_slice());
                });

            cur_height += height;
        }
        let this_proof_height = (2 << preflight.proof_shape.n_logup) - 1;
        for (n, row_idx) in blob.all_stacked_ids[pidx].iter() {
            let idx = ((1 << (n_logup - *n as usize)) - 1 + (*row_idx >> (l_skip + *n as usize)))
                as usize;
            let cols: &mut EqMleColumns<_> = trace[(cur_height - this_proof_height + idx) * width
                ..(cur_height - this_proof_height + idx + 1) * width]
                .borrow_mut();
            cols.num_traces += F::ONE;
        }
    }

    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqMleColumns<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
