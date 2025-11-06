use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    SubAir,
    utils::{not, or},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        bus::{Eq3bBus, EqMleBus, EqMleMessage},
        eq_airs::EqMleBlob,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{MultiProofVecVec, assert_zeros},
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Eq3bColumns<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    n: T,
    hypercube_volume: T,
    inverse_hypercube_volume: T, // 2^{-n}

    is_not_fictious: T, // as opposed, for example, to when there is no trace with such n
    is_first_for_a_valid_air: T,

    stacked_row_idx: T,
    sort_idx: T,
    col_idx: T,
    eq: [T; D_EF],
}

pub struct Eq3bAir {
    pub eq_mle_bus: EqMleBus,
    pub eq_3b_bus: Eq3bBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for Eq3bAir {}
impl<F> PartitionedBaseAir<F> for Eq3bAir {}

impl<F> BaseAir<F> for Eq3bAir {
    fn width(&self) -> usize {
        Eq3bColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for Eq3bAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &Eq3bColumns<AB::Var> = (*local).borrow();
        let next: &Eq3bColumns<AB::Var> = (*next).borrow();

        // Summary:
        // - n consistency: ensure `n` stays constant on non-reset transitions, reset it to zero on
        //   the first row, initialize `inverse_hypercube_volume` to one, double it when `n`
        //   decreases, keep it unchanged otherwise, and enforce `hypercube_volume *
        //   inverse_hypercube_volume = 1`; TODO: receive `n_logup` from another AIR when the first
        //   valid row appears.
        // - AIR column updates: zero `eq` on fictitious rows, require `is_not_fictious` implies
        //   validity, restrict `sort_idx` to change by at most one with fictitious rows preceding
        //   non-fictitious ones, force fictitious rows to have `col_idx = 0`, increment `col_idx`
        //   within an AIR when staying on non-fictitious rows, and mark the first valid row so that
        //   its `col_idx` resets.
        // - Row indexing: start `stacked_row_idx` at zero, keep it fixed on fictitious rows, and
        //   increase it by the scaled `hypercube_volume` whenever a non-fictitious row occurs.

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
        builder.assert_bool(local.is_not_fictious);

        // ============================= n consistency =====================================
        // TODO: receive n_logup = (n if is_first and is_valid) from some other air
        builder
            .when(not(next.is_first))
            .assert_bool(local.n - next.n);
        builder.when(next.is_first).assert_zero(local.n);
        builder
            .when(next.is_first)
            .when(local.is_valid)
            .assert_one(local.inverse_hypercube_volume);
        builder
            .when(not(next.is_first))
            .when(local.n - next.n)
            .assert_eq(
                local.inverse_hypercube_volume * AB::Expr::TWO,
                next.inverse_hypercube_volume,
            );
        builder
            .when(not(next.is_first))
            .when::<AB::Expr>(not(local.n - next.n))
            .assert_eq(
                local.inverse_hypercube_volume,
                next.inverse_hypercube_volume,
            );
        builder
            .when(local.is_valid)
            .assert_one(local.hypercube_volume * local.inverse_hypercube_volume);

        // ===================== air related cols update consistency =============================
        assert_zeros(&mut builder.when(not(local.is_not_fictious)), local.eq);
        builder
            .when(local.is_not_fictious)
            .assert_one(local.is_valid);
        // sort_idx always increases by 0/1
        builder
            .when(not(next.is_first))
            .assert_bool(next.sort_idx - local.sort_idx);
        // For each sort_idx, first we have fictious rows and then maybe non-fictious
        builder
            .when::<AB::Expr>(not(next.sort_idx - local.sort_idx))
            .when(not(next.is_first))
            .when(local.is_not_fictious)
            .assert_one(next.is_not_fictious);
        // For fictious rows, col_idx = 0
        builder
            .when(not(local.is_not_fictious))
            .assert_zero(local.col_idx);
        // For non-fictious within same sort_idx, col_idx always increases by one
        let is_transition_within_air = (AB::Expr::ONE - next.sort_idx + local.sort_idx)
            * local.is_not_fictious
            * not(next.is_first);
        builder
            .when(is_transition_within_air)
            .assert_one(next.col_idx - local.col_idx);
        // First non-fictious row within an AIR has zero col_idx
        builder.assert_eq::<AB::Var, AB::Expr>(
            next.is_first_for_a_valid_air,
            next.is_not_fictious
                * or::<AB::Expr>(
                    not(local.is_not_fictious),
                    or(next.sort_idx - local.sort_idx, next.is_first),
                ),
        );
        builder
            .when(next.is_first_for_a_valid_air)
            .assert_zero(next.col_idx);

        // Initially row_idx = 0
        builder
            .when(local.is_first)
            .assert_zero(local.stacked_row_idx);
        // On fictious rows it doesn't change
        builder
            .when(not(local.is_not_fictious))
            .when(not(next.is_first))
            .assert_eq(local.stacked_row_idx, next.stacked_row_idx);
        // On non-fictious rows it increases by hypercube_volume
        builder
            .when(local.is_not_fictious)
            .when(not(next.is_first))
            .assert_eq(
                next.stacked_row_idx,
                local.stacked_row_idx
                    + local.hypercube_volume * AB::Expr::from_canonical_usize(1 << self.l_skip),
            );

        self.eq_mle_bus.receive(
            builder,
            local.proof_idx,
            EqMleMessage {
                n: local.n.into(),
                idx: local.stacked_row_idx * local.inverse_hypercube_volume,
                eq_mle: local.eq.map(|x| x.into()),
            },
            local.is_not_fictious,
        );
        // self.eq_3b_bus.send(
        //     builder,
        //     local.proof_idx,
        //     Eq3bMessage {
        //         sort_idx: local.sort_idx,
        //         col_idx: local.col_idx,
        //         eq_mle: local.eq,
        //     },
        //     local.is_valid,
        // );
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Eq3bRecord {
    n: usize,
    stacked_row_idx: usize,
    sort_idx: usize,
    col_idx: usize,
    eq: [F; D_EF],
    is_not_fictious: bool,
}

/// TODO(AG): incorporate into EqMleBlob?
struct Eq3bBlob {
    records: MultiProofVecVec<Eq3bRecord>,
}

fn generate_eq_3b_blob(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &EqMleBlob,
    preflights: &[Preflight],
) -> Eq3bBlob {
    let mut res = Eq3bBlob {
        records: MultiProofVecVec::new(),
    };
    let l_skip = vk.inner.params.l_skip;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let n_logup = preflight.proof_shape.n_logup;
        let mut n = n_logup;
        let mut n_is_empty = true;
        let mut row_idx = 0;
        let products = &blob.products[pidx];
        let vdata = &preflight.proof_shape.sorted_trace_vdata;
        for (sort_idx, (air_idx, vdata)) in vdata.iter().enumerate() {
            let num_interactions = vk.inner.per_air[*air_idx].num_interactions();
            if num_interactions == 0 {
                res.records.push(Eq3bRecord {
                    n,
                    stacked_row_idx: row_idx,
                    sort_idx,
                    col_idx: 0,
                    eq: [F::ZERO; _],
                    is_not_fictious: false,
                });
                continue;
            }

            let n_lift = vdata.log_height.saturating_sub(l_skip);
            while n > n_lift {
                if n_is_empty {
                    res.records.push(Eq3bRecord {
                        n,
                        stacked_row_idx: row_idx,
                        sort_idx,
                        col_idx: 0,
                        eq: [F::ZERO; _],
                        is_not_fictious: false,
                    });
                }
                n -= 1;
                n_is_empty = true;
            }
            debug_assert_eq!(n, n_lift);

            for col_idx in 0..num_interactions {
                res.records.push(Eq3bRecord {
                    n,
                    stacked_row_idx: row_idx,
                    sort_idx,
                    col_idx,
                    eq: products[(1 << (n_logup - n)) - 1 + (row_idx >> (l_skip + n))]
                        .as_base_slice()
                        .try_into()
                        .unwrap(),
                    is_not_fictious: true,
                });

                n_is_empty = false;
                row_idx += 1 << (l_skip + n);
            }
        }
        debug_assert!(row_idx <= 1 << (l_skip + n_logup));
        for n in (0..=n).rev() {
            if n_is_empty {
                res.records.push(Eq3bRecord {
                    n,
                    stacked_row_idx: row_idx,
                    sort_idx: vdata.len(),
                    col_idx: 0,
                    eq: [F::ZERO; _],
                    is_not_fictious: false,
                });
            }
            n_is_empty = true;
        }
        res.records.end_proof();
    }
    res
}

pub(crate) fn generate_eq_3b_trace(
    vk: &MultiStarkVerifyingKeyV2,
    blob: &EqMleBlob,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = Eq3bColumns::<F>::width();

    let blob = generate_eq_3b_blob(vk, blob, preflights);
    let total_height = blob.records.len();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;

    for pidx in 0..blob.records.num_proofs() {
        let records = &blob.records[pidx];
        trace[cur_height * width..(cur_height + records.len()) * width]
            .par_chunks_exact_mut(width)
            .zip(records.par_iter())
            .for_each(|(chunk, record)| {
                let cols: &mut Eq3bColumns<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.n = F::from_canonical_usize(record.n);
                cols.hypercube_volume = F::from_canonical_usize(1 << record.n);
                cols.inverse_hypercube_volume = cols.hypercube_volume.inverse();
                cols.is_not_fictious = F::from_bool(record.is_not_fictious);
                cols.is_first_for_a_valid_air =
                    F::from_bool(record.is_not_fictious && record.col_idx == 0);
                cols.stacked_row_idx = F::from_canonical_usize(record.stacked_row_idx);
                cols.sort_idx = F::from_canonical_usize(record.sort_idx);
                cols.col_idx = F::from_canonical_usize(record.col_idx);
                cols.eq = record.eq;
            });
        {
            let cols: &mut Eq3bColumns<_> =
                trace[cur_height * width..(cur_height + 1) * width].borrow_mut();
            cols.is_first = F::ONE;
        }
        cur_height += records.len();
        {
            let cols: &mut Eq3bColumns<_> =
                trace[(cur_height - 1) * width..cur_height * width].borrow_mut();
            cols.is_last = F::ONE;
        }
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut Eq3bColumns<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
        });

    RowMajorMatrix::new(trace, width)
}
