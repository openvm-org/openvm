use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
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
        ExpressionClaimBus, ExpressionClaimMessage, InteractionsFoldingBus,
        InteractionsFoldingMessage, SymbolicExpressionBus,
    },
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage, AirShapeProperty,
        ColumnClaimsBus, ColumnClaimsMessage, StackingModuleBus, StackingModuleMessage,
        TranscriptBus, TranscriptBusMessage,
    },
    system::Preflight,
};

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct CachedSymbolicExpressionColumns<T> {
    is_valid: T,

    air_idx: T,
    node_idx: T,
    node_type: T,
    arg_ids: [T; 2],

    is_constraint: T,
    constraint_idx: T, /* TODO: we can use one `constraint_idx + 1` and make it 0 on
                        * non-constraints */

    /// Interactions will always be the duplicates of previous rows,
    /// but each describing a particular interaction component
    is_interaction: T,
    is_mult: T,
    interaction_idx: T,
    idx_in_message: T,
    interaction_mult: T,
}

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct SingleMainSymbolicExpressionColumns<T> {
    args: [[T; D_EF]; 2],
    tidx: T,
    is_existing_proof: T,
}

pub struct SymbolicExpressionAir {
    pub expr_bus: SymbolicExpressionBus,
    pub claim_bus: ExpressionClaimBus,
    pub stacking_module_bus: StackingModuleBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub interactions_folding_bus: InteractionsFoldingBus,

    pub cnt_proofs: usize,
}

impl<F> BaseAirWithPublicValues<F> for SymbolicExpressionAir {}
impl<F> PartitionedBaseAir<F> for SymbolicExpressionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![CachedSymbolicExpressionColumns::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        SingleMainSymbolicExpressionColumns::<F>::width() * self.cnt_proofs
    }
}

impl<F> BaseAir<F> for SymbolicExpressionAir {
    fn width(&self) -> usize {
        CachedSymbolicExpressionColumns::<F>::width()
            + SingleMainSymbolicExpressionColumns::<F>::width() * self.cnt_proofs
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for SymbolicExpressionAir {
    fn eval(&self, builder: &mut AB) {
        let main_local = builder.common_main().row_slice(0).to_vec();
        let cached_local = builder.cached_mains()[0].row_slice(0).to_vec();
        let single_main_width = SingleMainSymbolicExpressionColumns::<AB::Var>::width();
        let cached_cols: &CachedSymbolicExpressionColumns<AB::Var> = cached_local[..].borrow();
        let main_cols: Vec<&SingleMainSymbolicExpressionColumns<AB::Var>> = main_local
            .chunks(single_main_width)
            .map(|chunk| chunk.borrow())
            .collect();

        for (i, &cols) in main_cols.iter().enumerate() {
            let value = core::array::from_fn(|_| AB::Expr::ZERO);
            // TODO actually compute the value and uncomment
            // self.expr_bus.send(
            //     builder,
            //     AB::Expr::from_canonical_usize(i),
            //     SymbolicExpressionMessage {
            //         air_idx: cached_cols.air_idx.into(),
            //         node_idx: cached_cols.node_idx.into(),
            //         value: value.clone(),
            //     },
            //     cached_cols.is_valid,
            // );

            // self.expr_bus.receive(
            //     builder,
            //     AB::Expr::from_canonical_usize(i),
            //     SymbolicExpressionMessage {
            //         air_idx: cached_cols.air_idx,
            //         node_idx: cached_cols.arg_ids[0],
            //         value: local.arg_vals[i][0],
            //     },
            //     cached_cols.is_valid, // TODO the indicator that the node type is not constant
            // or something );
            // self.expr_bus.receive(
            //     builder,
            //     AB::Expr::from_canonical_usize(i),
            //     SymbolicExpressionMessage {
            //         air_idx: cached_cols.air_idx,
            //         node_idx: cached_cols.arg_ids[1],
            //         value: local.arg_vals[i][1],
            //     },
            //     cached_cols.is_valid, // TODO the indicator that the node type is a binary
            // operation );
            self.stacking_module_bus.send(
                builder,
                AB::Expr::from_canonical_usize(i),
                StackingModuleMessage { tidx: cols.tidx },
                cols.is_existing_proof,
            );
            self.claim_bus.send(
                builder,
                AB::Expr::from_canonical_usize(i),
                ExpressionClaimMessage {
                    is_interaction: AB::Expr::ZERO,
                    idx: cached_cols.constraint_idx.into(),
                    value: value.clone(),
                },
                cached_cols.is_constraint,
            );
            self.interactions_folding_bus.send(
                builder,
                AB::Expr::from_canonical_usize(i),
                InteractionsFoldingMessage {
                    node_idx: cached_cols.node_idx.into(),
                    interaction_idx: cached_cols.interaction_idx.into(),
                    is_mult: cached_cols.is_mult.into(),
                    idx_in_message: cached_cols.idx_in_message.into(),
                    value: value.clone(),
                },
                cached_cols.is_interaction,
            );
        }
    }
}

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
    has_preprocessed: T,

    tidx: T,
    col_claim: [T; D_EF],
    rot_claim: [T; D_EF],
}

pub struct ColumnClaimAir {
    pub transcript_bus: TranscriptBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
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
        self.column_claims_bus.send(
            builder,
            local.proof_idx,
            ColumnClaimsMessage {
                sort_idx: local.sort_idx,
                part_idx: local.part_idx,
                col_idx: local.col_idx,
                col_claim: local.col_claim,
                rot_claim: local.rot_claim,
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

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct ExpressionClaimCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    is_interaction: T,
    idx: T,
    lambda_tidx: T,
    lambda: [T; D_EF],
    lambda_pow: [T; D_EF],
    value: [T; D_EF],
}

pub struct ExpressionClaimAir {
    pub claim_bus: ExpressionClaimBus,
    pub transcript_bus: TranscriptBus,
}

impl<F> BaseAirWithPublicValues<F> for ExpressionClaimAir {}
impl<F> PartitionedBaseAir<F> for ExpressionClaimAir {}

impl<F> BaseAir<F> for ExpressionClaimAir {
    fn width(&self) -> usize {
        ExpressionClaimCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpressionClaimAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &ExpressionClaimCols<AB::Var> = (*local).borrow();
        let _next: &ExpressionClaimCols<AB::Var> = (*next).borrow();

        // self.claim_bus.receive(
        //     builder,
        //     local.proof_idx,
        //     ExpressionClaimMessage {
        //         is_interaction: local.is_interaction,
        //         idx: local.idx,
        //         value: local.value,
        //     },
        //     local.is_valid,
        // );

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.lambda_tidx + AB::Expr::from_canonical_usize(i),
                    value: local.lambda[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_first,
            );
        }
    }
}

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
struct InteractionsFoldingCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    beta_tidx: T,

    sort_idx: T,
    interaction_idx: T,
    node_idx: T,

    is_fictious: T, // sometimes we don't have interactions but still need to sample beta

    is_first_in_message: T, // aka "is_mult"
    is_last_in_message: T,
    idx_in_message: T,
    value: [T; D_EF],
    cur_sum: [T; D_EF],
    beta: [T; D_EF],
}

pub struct InteractionsFoldingAir {
    pub interaction_bus: InteractionsFoldingBus,
    pub air_shape_bus: AirShapeBus,
    pub transcript_bus: TranscriptBus,
}

impl<F> BaseAirWithPublicValues<F> for InteractionsFoldingAir {}
impl<F> PartitionedBaseAir<F> for InteractionsFoldingAir {}

impl<F> BaseAir<F> for InteractionsFoldingAir {
    fn width(&self) -> usize {
        InteractionsFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for InteractionsFoldingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &InteractionsFoldingCols<AB::Var> = (*local).borrow();
        let next: &InteractionsFoldingCols<AB::Var> = (*next).borrow();

        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.beta_tidx + AB::Expr::from_canonical_usize(i),
                    value: local.beta[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_first,
            );
        }

        let is_last_in_this_air =
            (next.sort_idx - local.sort_idx) * (AB::Expr::ONE - local.is_last) + local.is_last;
        self.air_shape_bus.receive(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::NumInteractions.to_field(),
                value: (local.interaction_idx + AB::Expr::ONE)
                    * (AB::Expr::ONE - local.is_fictious),
            },
            is_last_in_this_air,
        );
    }
}

/// Returns the common main trace.
pub(crate) fn generate_symbolic_expr_common_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
    max_num_proofs: usize,
) -> RowMajorMatrix<F> {
    let single_main_width = SingleMainSymbolicExpressionColumns::<F>::width();
    let main_width = single_main_width * max_num_proofs;

    let mut main_trace = vec![F::ZERO; main_width];
    for (preflight, chunk) in preflights
        .iter()
        .zip(main_trace[..].chunks_mut(single_main_width))
    {
        let cols: &mut SingleMainSymbolicExpressionColumns<_> = chunk[..].borrow_mut();
        cols.tidx = F::from_canonical_usize(preflight.batch_constraint.post_tidx);
        cols.is_existing_proof = F::ONE;
    }

    RowMajorMatrix::new(main_trace, main_width)
}

/// Returns the cached trace
pub(crate) fn generate_symbolic_expr_cached_trace(
    _child_vk: &MultiStarkVerifyingKeyV2,
) -> RowMajorMatrix<F> {
    let cached_width = CachedSymbolicExpressionColumns::<F>::width();

    let mut cached_trace = vec![F::ZERO; cached_width];
    // TODO: fill in cached trace
    {
        let cols: &mut CachedSymbolicExpressionColumns<_> =
            cached_trace[..cached_width].borrow_mut();
        cols.is_valid = F::ONE;
    }

    RowMajorMatrix::new(cached_trace, cached_width)
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
        air_idx: usize,
        sort_idx: usize,
        part_idx: usize,
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
        for (sort_idx, &(air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
        {
            let air_vk = &vk.inner.per_air[air_id];
            let widths = &air_vk.params.width;
            let has_preprocessed = widths.preprocessed.is_some();

            for col in 0..widths.common_main {
                let (col_claim, rot_claim) =
                    proof.batch_constraint_proof.column_openings[sort_idx][0][col];
                let mut col_claim_arr = [F::ZERO; D_EF];
                col_claim_arr.copy_from_slice(col_claim.as_base_slice());
                let mut rot_claim_arr = [F::ZERO; D_EF];
                rot_claim_arr.copy_from_slice(rot_claim.as_base_slice());
                rows.push(ColumnRowInfo {
                    is_first: rows.len() == initial_len,
                    is_last: false,
                    proof_idx: pidx,
                    tidx: main_tidx[sort_idx] + col * 2 * D_EF,
                    air_idx: air_id,
                    sort_idx,
                    part_idx: 0,
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
                        air_idx: air_id,
                        sort_idx,
                        part_idx: part + 1,
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
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(row.is_first);
            cols.is_last = F::from_bool(row.is_last);
            cols.proof_idx = F::from_canonical_usize(row.proof_idx);
            cols.tidx = F::from_canonical_usize(row.tidx);
            cols.air_idx = F::from_canonical_usize(row.air_idx);
            cols.sort_idx = F::from_canonical_usize(row.sort_idx);
            cols.part_idx = F::from_canonical_usize(row.part_idx);
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

pub(crate) fn generate_expression_claim_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = ExpressionClaimCols::<F>::width();

    let height = preflights.len();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    for (i, preflight) in preflights.iter().enumerate() {
        let cols: &mut ExpressionClaimCols<_> = trace[i * width..(i + 1) * width].borrow_mut();
        cols.is_first = F::ONE;
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_canonical_usize(i);
        let tidx = preflight.gkr.post_tidx;
        cols.lambda_tidx = F::from_canonical_usize(tidx);
        cols.lambda
            .copy_from_slice(&preflight.transcript.values()[tidx..tidx + D_EF]);
    }
    trace[height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut ExpressionClaimCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });
    RowMajorMatrix::new(trace, width)
}

pub(crate) fn generate_interactions_folding_trace(
    vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = InteractionsFoldingCols::<F>::width();
    let interactions = vk
        .inner
        .per_air
        .iter()
        .map(|vk| {
            vk.symbolic_constraints
                .interactions
                .iter()
                .cloned()
                .collect_vec()
        })
        .collect_vec();
    let height = interactions
        .iter()
        .map(|inters| inters.iter().map(|i| i.message.len() + 2).sum::<usize>())
        .sum::<usize>();

    let total_height = height.max(1) * preflights.len();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    if height == 0 {
        for (i, preflight) in preflights.iter().enumerate() {
            let beta_tidx = preflight.proof_shape.post_tidx + 2 + D_EF;
            let cols: &mut InteractionsFoldingCols<_> =
                trace[i * width..(i + 1) * width].borrow_mut();
            cols.is_valid = F::ONE;
            cols.is_first = F::ONE;
            cols.is_last = F::ONE;
            cols.is_fictious = F::ONE;
            cols.proof_idx = F::from_canonical_usize(i);
            cols.beta_tidx = F::from_canonical_usize(beta_tidx);
            cols.beta
                .copy_from_slice(&preflight.transcript.values()[beta_tidx..beta_tidx + D_EF]);
        }
        trace[preflights.len() * width..]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut InteractionsFoldingCols<F> = chunk.borrow_mut();
                cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
            });

        return RowMajorMatrix::new(trace, width);
    }

    #[derive(Clone)]
    struct InteractionRowInfo {
        is_first: bool,
        is_last: bool,
        is_first_in_message: bool,
        is_last_in_message: bool,
        proof_idx: usize,
        idx_in_message: usize,
        sort_idx: usize,
        interaction_idx: usize,
        node_idx: usize,
    }

    for (pidx, preflight) in preflights.iter().enumerate() {
        let mut rows = Vec::with_capacity(height);
        let beta_tidx = preflight.proof_shape.post_tidx + 2 + D_EF;

        for (sort_idx, inters) in interactions.iter().enumerate() {
            for (interaction_idx, inter) in inters.iter().enumerate() {
                let is_first = rows.is_empty();
                rows.push(InteractionRowInfo {
                    is_first,
                    is_last: false,
                    is_first_in_message: true,
                    is_last_in_message: false,
                    proof_idx: pidx,
                    idx_in_message: 0,
                    sort_idx,
                    interaction_idx,
                    node_idx: inter.count,
                });

                for (j, &node_idx) in inter.message.iter().enumerate() {
                    rows.push(InteractionRowInfo {
                        is_first: false,
                        is_last: false,
                        is_first_in_message: false,
                        is_last_in_message: false,
                        proof_idx: pidx,
                        idx_in_message: j,
                        sort_idx,
                        interaction_idx,
                        node_idx,
                    });
                }

                rows.push(InteractionRowInfo {
                    is_first: false,
                    is_last: false,
                    is_first_in_message: false,
                    is_last_in_message: true,
                    proof_idx: pidx,
                    idx_in_message: inter.message.len() + 1,
                    sort_idx,
                    interaction_idx,
                    node_idx: usize::from(inter.bus_index + 1),
                });
            }
        }

        debug_assert_eq!(rows.len(), height);
        rows.last_mut().unwrap().is_last = true;

        let beta_slice = &preflight.transcript.values()[beta_tidx..beta_tidx + D_EF];

        trace[pidx * height * width..]
            .par_chunks_exact_mut(width)
            .zip(rows.par_iter())
            .for_each(|(chunk, row)| {
                let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row.is_first);
                cols.is_last = F::from_bool(row.is_last);
                cols.proof_idx = F::from_canonical_usize(row.proof_idx);
                cols.beta_tidx = F::from_canonical_usize(beta_tidx);
                cols.beta.copy_from_slice(beta_slice);
                cols.sort_idx = F::from_canonical_usize(row.sort_idx);
                cols.interaction_idx = F::from_canonical_usize(row.interaction_idx);
                cols.node_idx = F::from_canonical_usize(row.node_idx);
                cols.idx_in_message = F::from_canonical_usize(row.idx_in_message);
                cols.is_first_in_message = F::from_bool(row.is_first_in_message);
                cols.is_last_in_message = F::from_bool(row.is_last_in_message);
                cols.value.fill(F::ZERO);
                cols.cur_sum.fill(F::ZERO);
            });
    }
    trace[preflights.len() * height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut InteractionsFoldingCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
