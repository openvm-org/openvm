//! Trace layout for the Final Poly MLE evaluation AIR.
//!
//! Each row describes a node of the final polynomial evaluation tree: it consumes the two child
//! contributions for that node—coefficients when `layer = 1`, or previously combined node values on
//! higher layers—and produces the parent value `value = left + right * point`. The row enforces
//! these relationships locally, while internal buses ensure every produced value is consumed
//! exactly once.

use core::borrow::Borrow;

use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, WhirOpeningPointBus, WhirOpeningPointLookupBus, WhirOpeningPointMessage},
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
    whir::bus::{
        FinalPolyFoldingBus, FinalPolyFoldingMessage, FinalPolyMleEvalBus, FinalPolyMleEvalMessage,
        WhirEqAlphaUBus, WhirEqAlphaUMessage, WhirFinalPolyBus, WhirFinalPolyBusMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FinalyPolyMleEvalCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub is_root: T,
    pub layer: T,
    pub layer_inv: T,
    pub node_idx: T,
    pub node_idx_inv: T,
    pub tidx_final_poly_start: T,
    pub is_nonleaf_and_first_in_layer: T,
    pub point: [T; D_EF],
    pub left_value: [T; D_EF],
    pub right_value: [T; D_EF],
    pub value: [T; D_EF],
    pub result: [T; D_EF],
    pub eq_alpha_u: [T; D_EF],
    pub num_nodes_in_layer: T,
}

pub struct FinalPolyMleEvalAir {
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub whir_opening_point_lookup_bus: WhirOpeningPointLookupBus,
    pub final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    pub transcript_bus: TranscriptBus,
    pub eq_alpha_u_bus: WhirEqAlphaUBus,
    pub final_poly_bus: WhirFinalPolyBus,
    pub folding_bus: FinalPolyFoldingBus,
    pub num_vars: usize,
    pub num_sumcheck_rounds: usize,
    pub num_whir_rounds: usize,
    pub total_whir_queries: usize,
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
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("window should have two elements");
        let local: &FinalyPolyMleEvalCols<AB::Var> = (*local_row).borrow();

        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_root);
        builder.when(local.layer).assert_one(local.is_enabled);
        builder.when(local.node_idx).assert_one(local.is_enabled);

        let is_nonleaf = local.layer_inv * local.layer;
        builder.when(local.layer).assert_one(is_nonleaf.clone());

        builder.when(local.is_root).assert_one(local.is_enabled);

        let num_vars = AB::Expr::from_usize(self.num_vars);
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

        let var_idx = AB::Expr::from_usize(self.num_sumcheck_rounds + self.num_vars) - local.layer;

        builder.assert_eq(
            local.is_nonleaf_and_first_in_layer,
            is_nonleaf.clone() * (local.is_enabled - is_node_idx_nonzero.clone()),
        );
        let is_nonleaf_and_not_first_in_layer = is_nonleaf.clone() * is_node_idx_nonzero;

        // Each non-leaf node needs the opening point for its layer. The first
        // node (node_idx=0) receives it from the permutation bus and registers
        // it on a lookup bus for the remaining nodes in this layer. The fact
        // that we receive on the permutation bus receive ensures each key is
        // registered exactly once on the lookup bus (since the producers send
        // each key once).
        self.whir_opening_point_bus.receive(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: var_idx.clone(),
                value: local.point.map(Into::into),
            },
            local.is_nonleaf_and_first_in_layer,
        );
        self.whir_opening_point_lookup_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: var_idx.clone(),
                value: local.point.map(Into::into),
            },
            local.is_nonleaf_and_first_in_layer * (local.num_nodes_in_layer - AB::Expr::ONE),
        );
        self.whir_opening_point_lookup_bus.lookup_key(
            builder,
            local.proof_idx,
            WhirOpeningPointMessage {
                idx: var_idx,
                value: local.point.map(Into::into),
            },
            is_nonleaf_and_not_first_in_layer,
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
                tidx_final_poly_start: local.tidx_final_poly_start.into(),
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
                tidx_final_poly_start: local.tidx_final_poly_start.into(),
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
                tidx_final_poly_start: local.tidx_final_poly_start,
                value: local.value,
            },
            local.is_enabled - local.is_root,
        );

        // Evaluation-form MLE folding: value = left + (right - left) * point
        assert_array_eq(
            &mut builder.when(is_nonleaf.clone()),
            local.value.map(Into::into),
            ext_field_add::<AB::Expr>(
                local.left_value,
                ext_field_multiply::<AB::Expr>(
                    ext_field_subtract::<AB::Expr>(local.right_value, local.left_value),
                    local.point,
                ),
            ),
        );
        assert_array_eq(
            &mut builder.when(local.is_root),
            local.value.map(Into::into),
            local.result.map(Into::into),
        );

        let is_leaf = local.is_enabled - is_nonleaf;
        let delta = AB::Expr::from_usize(D_EF);
        let tidx_node = local.tidx_final_poly_start + local.node_idx * delta;
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            tidx_node,
            local.value,
            is_leaf.clone(),
        );

        let total_whir_queries = AB::Expr::from_usize(self.total_whir_queries);
        self.final_poly_bus.add_key_with_lookups(
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
                num_whir_rounds: AB::Expr::from_usize(self.num_whir_rounds),
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
