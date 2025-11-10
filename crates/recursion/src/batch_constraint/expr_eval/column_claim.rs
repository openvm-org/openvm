use std::borrow::{Borrow, BorrowMut};

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
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage, AirShapeProperty,
        ColumnClaimsBus, ColumnClaimsMessage, HyperdimBus, HyperdimBusMessage, StackingModuleBus,
        StackingModuleMessage, TranscriptBus, TranscriptBusMessage,
    },
    system::Preflight,
};

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct ColumnClaimCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    air_idx: T,
    sort_idx: T,
    part_idx: T,
    col_idx: T,
    hypercube_dim: T,
    hypercube_is_neg: T,
    has_preprocessed: T,

    tidx: T,
    // TODO: remove this and send properly
    send_tidx: T,
    col_claim: [T; D_EF],
    rot_claim: [T; D_EF],
}

pub struct ColumnClaimAir {
    pub transcript_bus: TranscriptBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub hyperdim_bus: HyperdimBus,
    pub stacking_module_bus: StackingModuleBus,
}

impl<F> BaseAirWithPublicValues<F> for ColumnClaimAir {}
impl<F> PartitionedBaseAir<F> for ColumnClaimAir {}

impl<F> BaseAir<F> for ColumnClaimAir {
    fn width(&self) -> usize {
        ColumnClaimCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ColumnClaimAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &ColumnClaimCols<AB::Var> = (*local).borrow();
        let next: &ColumnClaimCols<AB::Var> = (*next).borrow();

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
        }
        self.stacking_module_bus.send(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(2 * D_EF),
            },
            local.send_tidx,
        );
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

        let last_row_of_this_air =
            (next.sort_idx - local.sort_idx) * (AB::Expr::ONE - local.is_last) + local.is_last;
        let last_row_of_this_part = (next.part_idx - local.part_idx)
            * (AB::Expr::ONE - last_row_of_this_air.clone())
            + last_row_of_this_air.clone();
        self.air_part_shape_bus.receive(
            builder,
            local.proof_idx,
            AirPartShapeBusMessage {
                idx: local.air_idx.into(),
                part: local.part_idx.into(),
                width: local.col_idx + AB::Expr::ONE,
            },
            last_row_of_this_part.clone(),
        );
        // air_id
        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::AirId.to_field(),
                value: local.air_idx.into(),
            },
            last_row_of_this_air.clone(),
        );
        // hypercube dim
        self.hyperdim_bus.receive(
            builder,
            local.proof_idx,
            HyperdimBusMessage {
                sort_idx: local.sort_idx.into(),
                n_abs: local.hypercube_dim
                    * (AB::Expr::ONE - local.hypercube_is_neg * AB::Expr::TWO),
                n_sign_bit: local.hypercube_is_neg.into(),
            },
            last_row_of_this_air.clone(),
        );
        // has_preprocessed
        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::HasPreprocessed.to_field(),
                value: local.has_preprocessed.into(),
            },
            last_row_of_this_air.clone(),
        );
        // num_main_parts
        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::NumMainParts.to_field(),
                value: local.part_idx + AB::Expr::ONE - local.has_preprocessed.into(),
            },
            last_row_of_this_air.clone(),
        );
    }
}

pub(crate) fn generate_column_claim_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = ColumnClaimCols::<F>::width();

    #[derive(Clone)]
    struct ColumnRowInfo {
        is_first: bool,
        is_last: bool,
        proof_idx: usize,
        tidx: usize,
        send_tidx: bool,
        air_idx: usize,
        sort_idx: usize,
        part_idx: usize,
        hypercube_dim: isize,
        has_preprocessed: bool,
        col_idx: usize,
        col_claim: [F; D_EF],
        rot_claim: [F; D_EF],
    }

    let mut rows = Vec::new();

    for (pidx, (proof, preflight)) in proofs.iter().zip(preflights.iter()).enumerate() {
        let vdata = &preflight.proof_shape.sorted_trace_vdata;
        let mut main_tidx = Vec::with_capacity(vdata.len());
        let mut nonmain_tidx = Vec::with_capacity(vdata.len());
        let mut cur_main_tidx = 0;
        let mut cur_nonmain_tidx = 0;
        for (air_id, _) in vdata.iter() {
            let ws = &vk.inner.per_air[*air_id].params.width;
            main_tidx.push(cur_main_tidx);
            cur_main_tidx += ws.common_main;
            nonmain_tidx.push(cur_nonmain_tidx);
            cur_nonmain_tidx += ws.total_width(0) - ws.common_main;
        }
        let height = cur_main_tidx + cur_nonmain_tidx;
        for x in main_tidx.iter_mut() {
            let tidx = preflight.batch_constraint.tidx_before_column_openings + *x * 2 * D_EF;
            *x = tidx;
        }
        for x in nonmain_tidx.iter_mut() {
            let tidx = preflight.batch_constraint.tidx_before_column_openings
                + (cur_main_tidx + *x) * 2 * D_EF;
            *x = tidx;
        }
        debug_assert!(height > 0);

        let initial_len = rows.len();
        for (sort_idx, (air_id, vdata)) in
            preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
        {
            let air_vk = &vk.inner.per_air[*air_id];
            let widths = &air_vk.params.width;
            let has_preprocessed = widths.preprocessed.is_some();
            let l_skip = vk.inner.params.l_skip;

            for col in 0..widths.common_main {
                let (col_claim, rot_claim) =
                    proof.batch_constraint_proof.column_openings[sort_idx][0][col];
                let mut col_claim_arr = [F::ZERO; D_EF];
                col_claim_arr.copy_from_slice(col_claim.as_base_slice());
                let mut rot_claim_arr = [F::ZERO; D_EF];
                rot_claim_arr.copy_from_slice(rot_claim.as_base_slice());
                let tidx = main_tidx[sort_idx] + col * 2 * D_EF;
                rows.push(ColumnRowInfo {
                    is_first: rows.len() == initial_len,
                    is_last: false,
                    proof_idx: pidx,
                    tidx,
                    send_tidx: tidx + 2 * D_EF == preflight.batch_constraint.post_tidx,
                    air_idx: *air_id,
                    sort_idx,
                    part_idx: 0,
                    hypercube_dim: vdata.log_height as isize - l_skip as isize,
                    has_preprocessed,
                    col_idx: col,
                    col_claim: col_claim_arr,
                    rot_claim: rot_claim_arr,
                });
            }

            let mut cur_tidx = nonmain_tidx[sort_idx];
            for (part, &w) in widths
                .preprocessed
                .iter()
                .chain(widths.cached_mains.iter())
                .enumerate()
            {
                for col in 0..w {
                    let (col_claim, rot_claim) =
                        proof.batch_constraint_proof.column_openings[sort_idx][part + 1][col];
                    let mut col_claim_arr = [F::ZERO; D_EF];
                    col_claim_arr.copy_from_slice(col_claim.as_base_slice());
                    let mut rot_claim_arr = [F::ZERO; D_EF];
                    rot_claim_arr.copy_from_slice(rot_claim.as_base_slice());
                    rows.push(ColumnRowInfo {
                        is_first: rows.len() == initial_len,
                        is_last: false,
                        proof_idx: pidx,
                        tidx: cur_tidx,
                        send_tidx: cur_tidx + 2 * D_EF == preflight.batch_constraint.post_tidx,
                        air_idx: *air_id,
                        sort_idx,
                        part_idx: part + 1,
                        hypercube_dim: vdata.log_height as isize - l_skip as isize,
                        has_preprocessed,
                        col_idx: col,
                        col_claim: col_claim_arr,
                        rot_claim: rot_claim_arr,
                    });
                    cur_tidx += 2 * D_EF;
                }
            }
        }
        rows.last_mut().unwrap().is_last = true;
    }

    let total_height = rows.len();
    let padded_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

    trace[..total_height * width]
        .par_chunks_exact_mut(width)
        .zip(rows.into_par_iter())
        .for_each(|(chunk, row)| {
            let cols: &mut ColumnClaimCols<_> = chunk.borrow_mut();
            let neg_hypercube = row.hypercube_dim < 0;
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(row.is_first);
            cols.is_last = F::from_bool(row.is_last);
            cols.proof_idx = F::from_canonical_usize(row.proof_idx);
            cols.tidx = F::from_canonical_usize(row.tidx);
            cols.send_tidx = F::from_bool(row.send_tidx);
            cols.air_idx = F::from_canonical_usize(row.air_idx);
            cols.sort_idx = F::from_canonical_usize(row.sort_idx);
            cols.part_idx = F::from_canonical_usize(row.part_idx);
            cols.hypercube_dim = if neg_hypercube {
                -F::from_canonical_usize(row.hypercube_dim.unsigned_abs())
            } else {
                F::from_canonical_usize(row.hypercube_dim as usize)
            };
            cols.hypercube_is_neg = F::from_bool(neg_hypercube);
            cols.has_preprocessed = F::from_bool(row.has_preprocessed);
            cols.col_idx = F::from_canonical_usize(row.col_idx);
            cols.col_claim.copy_from_slice(&row.col_claim);
            cols.rot_claim.copy_from_slice(&row.rot_claim);
        });
    trace[total_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut ColumnClaimCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
