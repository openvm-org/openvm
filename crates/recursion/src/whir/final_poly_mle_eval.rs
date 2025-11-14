//! Trace layout for the Final Poly MLE evaluation AIR.
//!
//! Each row describes a node of the final polynomial evaluation tree: it consumes the two child
//! contributions for that node—coefficients when `layer = 1`, or previously combined node values on
//! higher layers—and produces the parent value `value = left + right * point`. The row enforces
//! these relationships locally, while internal buses ensure every produced value is consumed
//! exactly once.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, extension::BinomiallyExtendable};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, poly_common::Squarable, proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, WhirOpeningPointBus, WhirOpeningPointMessage},
    system::Preflight,
    utils::{ext_field_add, ext_field_multiply},
    whir::bus::{
        FinalPolyFoldingBus, FinalPolyFoldingMessage, FinalPolyMleEvalBus, FinalPolyMleEvalMessage,
        WhirEqAlphaUBus, WhirEqAlphaUMessage, WhirFinalPolyBus, WhirFinalPolyBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct FinalyPolyMleEvalCols<T> {
    is_enabled: T,
    proof_idx: T,
    is_root: T,
    layer: T,
    layer_inv: T,
    node_idx: T,
    node_idx_inv: T,
    tidx_final_poly_start: T,
    point: [T; D_EF],
    left_value: [T; D_EF],
    right_value: [T; D_EF],
    value: [T; D_EF],
    result: [T; D_EF],
    eq_alpha_u: [T; D_EF],
    num_nodes_in_layer: T,
}

pub struct FinalPolyMleEvalAir {
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    pub transcript_bus: TranscriptBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub final_poly_bus: WhirFinalPolyBus,
    pub folding_bus: FinalPolyFoldingBus,
    pub num_vars: usize,
    pub num_sumcheck_rounds: usize,
    pub num_whir_rounds: usize,
    pub num_whir_queries: usize,
}

impl BaseAirWithPublicValues<F> for FinalPolyMleEvalAir {}
impl PartitionedBaseAir<F> for FinalPolyMleEvalAir {}

impl<F> BaseAir<F> for FinalPolyMleEvalAir {
    fn width(&self) -> usize {
        FinalyPolyMleEvalCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FinalPolyMleEvalAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0);
        let local: &FinalyPolyMleEvalCols<AB::Var> = (*local_row).borrow();

        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_root);

        let is_nonleaf = local.layer_inv * local.layer;
        builder.when(local.layer).assert_one(is_nonleaf.clone());

        builder.when(local.is_root).assert_one(local.is_enabled);

        let num_vars = AB::Expr::from_canonical_usize(self.num_vars);
        builder
            .when(local.is_root)
            .assert_eq(local.layer, num_vars.clone());
        builder
            .when(local.is_root)
            .assert_eq(local.num_nodes_in_layer, AB::Expr::ONE);
        builder.when(local.is_root).assert_zero(local.node_idx);

        let is_node_idx_nonzero = local.node_idx * local.node_idx_inv;
        builder
            .when(local.node_idx)
            .assert_one(is_node_idx_nonzero.clone());

        let var_idx =
            AB::Expr::from_canonical_usize(self.num_sumcheck_rounds + self.num_vars) - local.layer;

        self.whir_opening_point_bus.receive(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: var_idx.clone(),
                value: local.point.map(Into::into),
            },
            is_nonleaf.clone(),
        );
        // copy it a bunch of times if we are node_idx 0
        // FIXME: technically we are violating the |multiplicity| <= 1 rule
        self.whir_opening_point_bus.send(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: var_idx,
                value: local.point.map(Into::into),
            },
            is_nonleaf.clone()
                * (local.is_enabled - is_node_idx_nonzero)
                * (local.num_nodes_in_layer - AB::Expr::ONE),
        );

        let left_idx = local.node_idx;
        let right_idx = left_idx + local.num_nodes_in_layer;
        let child_depth = local.layer - AB::Expr::ONE;

        self.folding_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyFoldingMessage {
                proof_idx: local.proof_idx.into(),
                depth: child_depth.clone(),
                node_idx: left_idx.into(),
                num_nodes_in_layer: local.num_nodes_in_layer * AB::Expr::TWO,
                value: local.left_value.map(Into::into),
            },
            is_nonleaf.clone(),
        );
        self.folding_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyFoldingMessage {
                proof_idx: local.proof_idx.into(),
                depth: child_depth,
                node_idx: right_idx.clone(),
                num_nodes_in_layer: local.num_nodes_in_layer * AB::Expr::TWO,
                value: local.right_value.map(Into::into),
            },
            is_nonleaf.clone(),
        );
        self.folding_bus.send(
            builder,
            local.proof_idx,
            FinalPolyFoldingMessage {
                proof_idx: local.proof_idx,
                depth: local.layer,
                node_idx: local.node_idx,
                num_nodes_in_layer: local.num_nodes_in_layer,
                value: local.value,
            },
            local.is_enabled - local.is_root,
        );

        assert_array_eq(
            &mut builder.when(is_nonleaf.clone()),
            local.value.map(Into::into),
            ext_field_add::<AB::Expr>(
                local.left_value,
                ext_field_multiply::<AB::Expr>(local.right_value, local.point),
            ),
        );
        assert_array_eq(
            &mut builder.when(local.is_root),
            local.value.map(Into::into),
            local.result.map(Into::into),
        );

        let is_leaf = local.is_enabled - is_nonleaf;
        let delta = AB::Expr::from_canonical_usize(D_EF);
        let tidx_node = local.tidx_final_poly_start + local.node_idx * delta;
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            tidx_node,
            local.value,
            is_leaf.clone(),
        );

        let total_whir_queries =
            AB::Expr::from_canonical_usize(self.num_whir_rounds * (self.num_whir_queries + 1));
        self.final_poly_bus.send(
            builder,
            local.proof_idx,
            WhirFinalPolyBusMessage {
                idx: local.node_idx,
                coeff: local.value,
            },
            is_leaf.clone() * total_whir_queries.clone(),
        );

        self.final_poly_mle_eval_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx_final_poly_start.into(),
                num_whir_rounds: AB::Expr::from_canonical_usize(self.num_whir_rounds),
                value: ext_field_multiply(local.value, local.eq_alpha_u),
            },
            local.is_root,
        );
        self.eq_alpha_u_bus.receive(
            builder,
            local.proof_idx,
            WhirEqAlphaUMessage {
                value: local.eq_alpha_u,
            },
            local.is_root,
        );
    }
}

#[tracing::instrument(name = "generate_trace(FinalPolyMleEvalAir)", skip_all)]
pub(crate) fn generate_trace(
    mvk: &MultiStarkVerifyingKeyV2,
    proofs: &[&Proof],
    preflights: &[&Preflight],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(proofs.len(), preflights.len());

    let params = mvk.inner.params;
    let num_vars = params.log_final_poly_len;

    let rows_per_proof = (1 << (num_vars + 1)) - 1;
    let total_valid_rows = rows_per_proof * proofs.len();
    let height = total_valid_rows.next_power_of_two();
    let width = FinalyPolyMleEvalCols::<F>::width();

    let mut trace = vec![F::ZERO; height * width];

    let tidx_base_offset = params.k_whir * D_EF * 3;
    let mut global_row = 0usize;

    for (proof_idx, proof) in proofs.iter().enumerate() {
        let preflight = &preflights[proof_idx];

        let num_sumcheck_rounds = params.n_stack + params.l_skip - num_vars;
        let (stacking_first, stacking_rest) =
            preflight.stacking.sumcheck_rnd.split_first().unwrap();
        let u_all = stacking_first
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(stacking_rest.iter().copied())
            .collect::<Vec<_>>();
        let eval_points = &u_all[num_sumcheck_rounds..num_sumcheck_rounds + num_vars];

        let result = preflight.whir.final_poly_at_u;
        let eq_alpha_u = preflight.whir.eq_partials.last().unwrap();

        let mut buf = proof.whir_proof.final_poly.clone();
        let tidx = preflight.whir.tidx_per_round.last().unwrap() + tidx_base_offset;

        let mut row_in_proof = 0usize;

        for layer in 0..=num_vars {
            let len = 1 << (num_vars - layer);
            let point = if layer > 0 {
                eval_points[num_vars - layer]
            } else {
                EF::ZERO
            };

            let (left_slice, right_slice) = buf.split_at_mut(len);
            for node_idx in 0..len {
                let row_idx = global_row + row_in_proof;
                let row = &mut trace[row_idx * width..(row_idx + 1) * width];
                let cols: &mut FinalyPolyMleEvalCols<F> = row.borrow_mut();

                let (left, right, value) = if layer == 0 {
                    let coeff = left_slice[node_idx];
                    (coeff, EF::ZERO, coeff)
                } else {
                    let l = left_slice[node_idx];
                    let r = right_slice[node_idx];
                    let value = l + r * point;
                    left_slice[node_idx] = value;
                    (l, r, value)
                };

                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.is_root = F::from_bool(layer == num_vars);
                cols.layer = F::from_canonical_usize(layer);
                cols.layer_inv = cols.layer.try_inverse().unwrap_or_default();
                cols.node_idx = F::from_canonical_usize(node_idx);
                cols.node_idx_inv = cols.node_idx.try_inverse().unwrap_or_default();
                cols.tidx_final_poly_start = F::from_canonical_usize(tidx);
                cols.point.copy_from_slice(point.as_base_slice());
                cols.left_value.copy_from_slice(left.as_base_slice());
                cols.right_value.copy_from_slice(right.as_base_slice());
                cols.value.copy_from_slice(value.as_base_slice());
                cols.result.copy_from_slice(result.as_base_slice());
                cols.eq_alpha_u.copy_from_slice(eq_alpha_u.as_base_slice());
                cols.num_nodes_in_layer = F::from_canonical_usize(1 << (num_vars - layer));

                row_in_proof += 1;
            }
        }

        debug_assert_eq!(row_in_proof, rows_per_proof);
        global_row += row_in_proof;
    }

    RowMajorMatrix::new(trace, width)
}
