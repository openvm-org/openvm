use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        Eq3bBus, EqMleBus, EqMleMessage, EqSharpUniBus, EqSharpUniMessage, EqZeroNBus,
        EqZeroNMessage,
    },
    bus::{
        AirShapeBus, AirShapeBusMessage, AirShapeProperty, ConstraintSumcheckRandomness,
        ConstraintSumcheckRandomnessBus, TranscriptBus, XiRandomnessBus, XiRandomnessMessage,
    },
    system::Preflight,
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqSharpUniCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    root: T,
    root_pow: T,
    root_order: T,

    xi_idx: T,
    xi_times_root_pow: [T; D_EF],
    product_before: [T; D_EF],

    iter_idx: T,
    is_first_iter: T,
}

pub struct EqSharpUniAir {
    pub xi_bus: XiRandomnessBus, // TODO: use another, inner bus instead?
    pub eq_bus: EqSharpUniBus,
    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqSharpUniAir {}
impl<F> PartitionedBaseAir<F> for EqSharpUniAir {}

impl<F> BaseAir<F> for EqSharpUniAir {
    fn width(&self) -> usize {
        EqSharpUniCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqSharpUniAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqSharpUniCols<AB::Var> = (*local).borrow();
        let _next: &EqSharpUniCols<AB::Var> = (*next).borrow();

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.xi_idx,
                xi: local.xi_times_root_pow,
            },
            local.is_first_iter, // TODO according to the design
        );
        // below is commented until actual tracegen is done

        // self.eq_bus.receive(
        //     builder,
        //     local.proof_idx,
        //     EqSharpUniMessage {
        //         xi_idx: local.xi_idx + AB::Expr::ONE,
        //         iter_idx: local.iter_idx.into(),
        //         product: local.product_before.map(|x| x.into()),
        //     },
        //     local.is_valid - local.is_first,
        // );
        // let product_after: [AB::Expr; D_EF] = local.product_before.map(|x| x.into()); // TODO
        // // multiply
        //
        // self.eq_bus.send(
        //     builder,
        //     local.proof_idx,
        //     EqSharpUniMessage {
        //         xi_idx: local.xi_idx.into(),
        //         iter_idx: local.iter_idx.into(),
        //         product: product_after.clone(),
        //     },
        //     local.is_valid - local.is_first,
        // );
        // self.eq_bus.send(
        //     builder,
        //     local.proof_idx,
        //     EqSharpUniMessage {
        //         xi_idx: local.xi_idx.into(),
        //         iter_idx: local.iter_idx.into() + local.root_order,
        //         product: product_after,
        //     },
        //     local.is_valid - local.is_first,
        // );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqSharpUniReceiverCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    idx: T,
    coeff: [T; D_EF],
    r: [T; D_EF],
    cur_sum: [T; D_EF],
}

pub struct EqSharpUniReceiverAir {
    pub r_bus: ConstraintSumcheckRandomnessBus, // TODO: use another, inner bus instead?
    pub eq_bus: EqSharpUniBus,
}

impl<F> BaseAirWithPublicValues<F> for EqSharpUniReceiverAir {}
impl<F> PartitionedBaseAir<F> for EqSharpUniReceiverAir {}

impl<F> BaseAir<F> for EqSharpUniReceiverAir {
    fn width(&self) -> usize {
        EqSharpUniReceiverCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqSharpUniReceiverAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqSharpUniReceiverCols<AB::Var> = (*local).borrow();
        let _next: &EqSharpUniReceiverCols<AB::Var> = (*next).borrow();

        self.r_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: AB::Expr::ZERO,
                challenge: local.r.map(|x| x.into()),
            },
            local.is_first,
        );
        self.eq_bus.receive(
            builder,
            local.proof_idx,
            EqSharpUniMessage {
                xi_idx: AB::Expr::ZERO,
                iter_idx: local.idx.into(),
                product: local.coeff.map(|x| x.into()),
            },
            local.is_first,
        );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqNsColumns<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    n: T,
    eq: [T; D_EF],
    eq_sharp: [T; D_EF],
    r_n: [T; D_EF],
    /// The number of traces with such `n`
    num_traces: T,
}

pub struct EqNsAir {
    pub zero_n_bus: EqZeroNBus,
    pub r_bus: ConstraintSumcheckRandomnessBus, // TODO: use another, inner bus instead?
}

impl<F> BaseAirWithPublicValues<F> for EqNsAir {}
impl<F> PartitionedBaseAir<F> for EqNsAir {}

impl<F> BaseAir<F> for EqNsAir {
    fn width(&self) -> usize {
        EqNsColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqNsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqNsColumns<AB::Var> = (*local).borrow();
        let _next: &EqNsColumns<AB::Var> = (*next).borrow();

        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ZERO,
                value: local.eq.map(|x| x.into()),
            },
            local.is_first,
        );
        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ONE,
                value: local.eq_sharp.map(|x| x.into()),
            },
            local.is_first,
        );

        self.r_bus.receive(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.n + AB::Expr::ONE,
                challenge: local.r_n.map(|x| x.into()),
            },
            local.is_valid,
        );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqMleColumns<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    n: T,
    xi_tidx: T,
    idx: T,
    mult: T,
    xi: [T; D_EF],
    eq: [T; D_EF],
    // TODO: add a bunch of hypercube related columns as in eqsharp air
}

pub struct EqMleAir {
    pub xi_bus: XiRandomnessBus, // TODO: use another, inner bus instead?
    pub eq_mle_bus: EqMleBus,
    pub transcript_bus: TranscriptBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqMleAir {}
impl<F> PartitionedBaseAir<F> for EqMleAir {}

impl<F> BaseAir<F> for EqMleAir {
    fn width(&self) -> usize {
        EqMleColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqMleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &EqMleColumns<AB::Var> = (*local).borrow();
        let _next: &EqMleColumns<AB::Var> = (*next).borrow();

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.n + AB::Expr::from_canonical_usize(self.l_skip),
                xi: local.xi.map(|x| x.into()),
            },
            local.is_valid,
        );

        self.eq_mle_bus.send(
            builder,
            local.proof_idx,
            EqMleMessage {
                idx: local.idx,
                n: local.n,
                eq_mle: local.eq,
            },
            local.mult,
        );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Eq3bColumns<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    n: T,
    stacked_row_idx: T,
    sort_idx: T,
    col_idx: T,
    eq: [T; D_EF],
    // TODO: add stuff
}

pub struct Eq3bAir {
    pub air_shape_bus: AirShapeBus,

    pub eq_mle_bus: EqMleBus,
    pub eq_3b_bus: Eq3bBus,
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

        // TODO: uncomment when dummy works
        // self.eq_mle_bus.receive(
        //     builder,
        //     local.proof_idx,
        //     EqMleMessage {
        //         n: local.n,
        //         idx: local.stacked_row_idx,
        //         eq_mle: local.eq,
        //     },
        //     local.is_valid,
        // );
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

        let is_last_for_this_air =
            (next.sort_idx - local.sort_idx) * (AB::Expr::ONE - local.is_last) + local.is_last;
        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::HypercubeDim.to_field(),
                value: local.n.into(),
            },
            is_last_for_this_air,
        );
    }
}

pub(crate) fn generate_eq_sharp_uni_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = EqSharpUniCols::<F>::width();
    let height = vk.inner.params.l_skip;
    let total_height = height * proofs.len();
    let padded_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

    for (pidx, preflight) in preflights.iter().enumerate() {
        trace[(pidx * height * width)..((pidx + 1) * height * width)]
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut EqSharpUniCols<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == height);
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.is_first_iter = F::ONE;
                cols.xi_idx = F::from_canonical_usize(i);
                cols.xi_times_root_pow
                    .copy_from_slice(preflight.gkr.xi[i].1.as_base_slice());
            });
    }
    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqSharpUniCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}

pub(crate) fn generate_eq_sharp_uni_receiver_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    _preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = EqSharpUniReceiverCols::<F>::width();
    RowMajorMatrix::new(vec![F::ZERO; width], width)
}

pub(crate) fn generate_eq_ns_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    _preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = EqNsColumns::<F>::width();
    RowMajorMatrix::new(vec![F::ZERO; width], width)
}

pub(crate) fn generate_eq_mle_trace(
    vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = EqMleColumns::<F>::width();
    let l_skip = vk.inner.params.l_skip;
    let heights = preflights
        .iter()
        .map(|p| p.proof_shape.n_global().max(1))
        .collect_vec();
    let total_height = heights.iter().sum::<usize>();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let xi = &preflight.batch_constraint.xi;
        let gkr_post_tidx = preflight.gkr.post_tidx;
        let height = heights[pidx];

        if height == 0 {
            todo!();
        }
        trace[cur_height * width..(cur_height + height) * width]
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut EqMleColumns<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.n = F::from_canonical_usize(i);
                cols.idx = F::from_canonical_usize(i);
                cols.xi.copy_from_slice(xi[l_skip + i].as_base_slice());
                cols.xi_tidx = F::from_canonical_usize(gkr_post_tidx + (i + 1) * D_EF);
            });

        cur_height += height;
    }

    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut EqMleColumns<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}

pub(crate) fn generate_eq_3b_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = Eq3bColumns::<F>::width();

    let vdatas_and_heights = preflights
        .iter()
        .map(|p| {
            let sorted = &p.proof_shape.sorted_trace_vdata;
            let height = sorted.len();
            debug_assert!(height > 0);
            (sorted, height)
        })
        .collect_vec();
    let total_height = vdatas_and_heights.iter().map(|x| x.1).sum::<usize>();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;

    for (pidx, &(sorted, height)) in vdatas_and_heights.iter().enumerate() {
        trace[cur_height * width..(cur_height + height) * width]
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut Eq3bColumns<_> = chunk.borrow_mut();
                let (_, v) = &sorted[i];
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == height);
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.sort_idx = F::from_canonical_usize(i);
                cols.n = F::from_canonical_usize(v.hypercube_dim);
            });
        cur_height += height;
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut Eq3bColumns<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
