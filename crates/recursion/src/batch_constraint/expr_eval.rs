use core::{array, iter::zip};
use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit_primitives::{
    SubAir,
    encoder::Encoder,
    utils::{assert_array_eq, not},
};
use openvm_stark_backend::{
    air_builders::{
        PartitionedAirBuilder,
        symbolic::{SymbolicExpressionNode, symbolic_variable::Entry},
    },
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

use crate::{
    batch_constraint::{
        BatchConstraintBlobCpu,
        bus::{
            ExpressionClaimBus, ExpressionClaimMessage, InteractionsFoldingBus,
            InteractionsFoldingMessage, SymbolicExpressionBus, SymbolicExpressionMessage,
        },
    },
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage, AirShapeProperty,
        ColumnClaimsBus, ColumnClaimsMessage, HyperdimBus, HyperdimBusMessage, PublicValuesBus,
        PublicValuesBusMessage, StackingModuleBus, StackingModuleMessage, TranscriptBus,
        TranscriptBusMessage,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
    utils::{
        MultiProofVecVec, base_to_ext, ext_field_add, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_subtract, scalar_subtract_ext_field,
    },
};

const NUM_FLAGS: usize = 4;

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct CachedSymbolicExpressionColumns<T> {
    flags: [T; NUM_FLAGS],

    air_idx: T,
    node_or_interaction_idx: T,
    /// Attributes that define this gate. For binary gates such as Add, Mul, etc.,
    /// this contains the node_idx's. For InteractionMsgComp, it gives (node_idx, idx_in_message).
    /// See [[NodeKind]].
    attrs: [T; 3],
    fanout: T,

    is_constraint: T,
    constraint_idx: T,
}

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct SingleMainSymbolicExpressionColumns<T> {
    is_present: T,
    // Dynamic arguments. For Add, Mul, Sub, this breaks into two [T; 4].
    // For selectors, args[0..4] give the value and args[4] gives the dimension.
    args: [T; 8],
}

pub struct SymbolicExpressionAir {
    pub expr_bus: SymbolicExpressionBus,
    pub claim_bus: ExpressionClaimBus,
    pub hyperdim_bus: HyperdimBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub interactions_folding_bus: InteractionsFoldingBus,
    pub public_values_bus: PublicValuesBus,

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

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for SymbolicExpressionAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main_local = builder.common_main().row_slice(0).to_vec();
        let cached_local = builder.cached_mains()[0].row_slice(0).to_vec();
        let single_main_width = SingleMainSymbolicExpressionColumns::<AB::Var>::width();
        let cached_cols: &CachedSymbolicExpressionColumns<AB::Var> = cached_local[..].borrow();
        let main_cols: Vec<&SingleMainSymbolicExpressionColumns<AB::Var>> = main_local
            .chunks(single_main_width)
            .map(|chunk| chunk.borrow())
            .collect();

        let enc = Encoder::new(NodeKind::COUNT, 2, true);
        assert_eq!(enc.width(), NUM_FLAGS);
        let flags = cached_cols.flags;

        let is_arg0_node_idx = enc.contains_flag::<AB>(
            &flags,
            &[
                NodeKind::Add,
                NodeKind::Sub,
                NodeKind::Mul,
                NodeKind::Neg,
                NodeKind::InteractionMult,
                NodeKind::InteractionMsgComp,
            ]
            .map(|x| x as usize),
        );
        let is_arg1_node_idx = enc.contains_flag::<AB>(
            &flags,
            &[NodeKind::Add, NodeKind::Sub, NodeKind::Mul].map(|x| x as usize),
        );

        for (proof_idx, &cols) in main_cols.iter().enumerate() {
            let proof_idx = AB::F::from_canonical_usize(proof_idx);

            let arg_ef0: [AB::Var; 4] = cols.args[..D_EF].try_into().unwrap();
            let arg_ef1: [AB::Var; 4] = cols.args[D_EF..2 * D_EF].try_into().unwrap();

            builder.assert_bool(cols.is_present);

            let mut value = [AB::Expr::ZERO; D_EF];
            for node_kind in NodeKind::iter() {
                // deg 2
                let sel = enc.get_flag_expr::<AB>(node_kind as usize, &flags);
                let expr = match node_kind {
                    NodeKind::Add => ext_field_add::<AB::Expr>(arg_ef0, arg_ef1),
                    NodeKind::Sub => ext_field_subtract::<AB::Expr>(arg_ef0, arg_ef1),
                    NodeKind::Neg => scalar_subtract_ext_field::<AB::Expr>(AB::F::ZERO, arg_ef0),
                    NodeKind::Mul => ext_field_multiply::<AB::Expr>(arg_ef0, arg_ef1),
                    NodeKind::Constant => base_to_ext(cached_cols.attrs[0]),
                    NodeKind::VarPublicValue => base_to_ext(cols.args[0]),
                    NodeKind::VarPreprocessed
                    | NodeKind::VarMain
                    | NodeKind::SelIsFirst
                    | NodeKind::SelIsLast
                    | NodeKind::SelIsTransition
                    | NodeKind::InteractionMult
                    | NodeKind::InteractionMsgComp => arg_ef0.map(Into::into),
                };
                // deg <= 4
                value = ext_field_add::<AB::Expr>(
                    value,
                    ext_field_multiply_scalar::<AB::Expr>(expr, sel),
                );
            }

            self.expr_bus.send(
                builder,
                proof_idx,
                SymbolicExpressionMessage {
                    air_idx: cached_cols.air_idx.into(),
                    node_idx: cached_cols.node_or_interaction_idx.into(),
                    value: value.clone(),
                },
                cols.is_present * cached_cols.fanout,
            );
            self.expr_bus.receive(
                builder,
                proof_idx,
                SymbolicExpressionMessage {
                    air_idx: cached_cols.air_idx,
                    node_idx: cached_cols.attrs[0],
                    value: arg_ef0,
                },
                cols.is_present * is_arg0_node_idx.clone(),
            );
            self.expr_bus.receive(
                builder,
                proof_idx,
                SymbolicExpressionMessage {
                    air_idx: cached_cols.air_idx,
                    node_idx: cached_cols.attrs[1],
                    value: arg_ef1,
                },
                cols.is_present * is_arg1_node_idx.clone(),
            );

            let _is_var = enc.contains_flag::<AB>(
                &flags,
                &[NodeKind::VarMain, NodeKind::VarPreprocessed].map(|x| x as usize),
            );
            self.column_claims_bus.receive(
                builder,
                proof_idx,
                ColumnClaimsMessage {
                    sort_idx: cols.args[D_EF],
                    part_idx: cached_cols.attrs[1],
                    col_idx: cached_cols.attrs[0],
                    claim: array::from_fn(|i| cols.args[i]),
                    is_rot: cached_cols.attrs[2],
                },
                AB::Expr::ZERO,
                // TODO: _is_var
            );
            self.public_values_bus.receive(
                builder,
                proof_idx,
                PublicValuesBusMessage {
                    air_idx: cached_cols.air_idx,
                    pv_idx: cached_cols.attrs[0],
                    value: cols.args[0],
                },
                AB::Expr::ZERO,
                // enc.get_flag_expr::<AB>(NodeKind::VarPublicValue as usize, &flags),
            );
            self.claim_bus.send(
                builder,
                proof_idx,
                ExpressionClaimMessage {
                    is_interaction: AB::Expr::ZERO,
                    idx: cached_cols.constraint_idx.into(),
                    value: value.clone(),
                },
                AB::Expr::ZERO,
                // TODO: cached_cols.is_constraint,
            );
            let _is_sel = enc.contains_flag::<AB>(
                &flags,
                &[
                    NodeKind::SelIsFirst,
                    NodeKind::SelIsLast,
                    NodeKind::SelIsTransition,
                ]
                .map(|x| x as usize),
            );
            self.hyperdim_bus.receive(
                builder,
                proof_idx,
                HyperdimBusMessage {
                    sort_idx: cols.args[D_EF],
                    n_abs: cols.args[D_EF + 1],
                    n_sign_bit: cols.args[D_EF + 2],
                },
                AB::Expr::ZERO,
                // TODO: _is_sel,
            );
            // TODO: Constrain cols.args[..D_EF] when _is_sel.
            let is_mult = enc.get_flag_expr::<AB>(NodeKind::InteractionMult as usize, &flags);
            let _is_interaction = enc.contains_flag::<AB>(
                &flags,
                &[NodeKind::InteractionMult, NodeKind::InteractionMsgComp].map(|x| x as usize),
            );
            self.interactions_folding_bus.send(
                builder,
                proof_idx,
                InteractionsFoldingMessage {
                    interaction_idx: cached_cols.node_or_interaction_idx.into(),
                    is_mult,
                    idx_in_message: cached_cols.attrs[1].into(),
                    value: value.clone(),
                },
                AB::Expr::ZERO,
                // TODO: _is_interaction,
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
    hypercube_dim: T,
    hypercube_is_neg: T,
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
            local.is_last,
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

    has_interactions: T,

    is_first_in_air: T,
    is_first_in_message: T, // aka "is_mult"

    loop_aux: NestedForLoopAuxCols<T, 2>,

    idx_in_message: T,
    value: [T; D_EF],
    cur_sum: [T; D_EF],
    beta: [T; D_EF],
}

pub struct InteractionsFoldingAir {
    pub interaction_bus: InteractionsFoldingBus,
    pub air_shape_bus: AirShapeBus,
    pub transcript_bus: TranscriptBus,
    pub expression_claim_bus: ExpressionClaimBus,
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
                idx: local.sort_idx.into(),
                value: local.cur_sum.map(Into::into),
            },
            // local.is_first_in_message,
            AB::Expr::ZERO,
        );
        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ZERO,
                idx: local.sort_idx.into(),
                value: next.cur_sum.map(Into::into),
            },
            // local.is_first_in_message,
            AB::Expr::ZERO,
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
    }
}

/// Returns the common main trace.
pub(in crate::batch_constraint) fn generate_symbolic_expr_common_trace(
    child_vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
    max_num_proofs: usize,
    blob: &BatchConstraintBlobCpu,
) -> RowMajorMatrix<F> {
    let params = child_vk.inner.params;

    let single_main_width = SingleMainSymbolicExpressionColumns::<F>::width();
    let main_width = single_main_width * max_num_proofs;

    struct Record {
        args: [F; 8],
    }
    let mut records = vec![];

    for (proof_idx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        for (air_idx, vk) in child_vk.inner.per_air.iter().enumerate() {
            let constraints = &vk.symbolic_constraints.constraints;
            let expr_evals = &blob.expr_evals[proof_idx][air_idx];

            // TODO: don't do any pushes at all for absent traces
            if expr_evals.is_empty() {
                for _ in 0..constraints.nodes.len() {
                    records.push(None);
                }
                for interaction in &vk.symbolic_constraints.interactions {
                    records.push(None);
                    for _ in 0..interaction.message.len() {
                        records.push(None);
                    }
                }
                continue;
            }

            let sort_idx = preflight
                .proof_shape
                .sorted_trace_vdata
                .iter()
                .position(|(idx, _)| *idx == air_idx)
                .unwrap();
            let sort_idx = F::from_canonical_usize(sort_idx);

            // TODO sort_idx in trace
            let log_height = proof.trace_vdata[air_idx].as_ref().unwrap().log_height;
            let (n_abs, n_sign_bit) = if log_height < params.l_skip {
                (F::from_canonical_usize(params.l_skip - log_height), F::ONE)
            } else {
                (F::from_canonical_usize(log_height - params.l_skip), F::ZERO)
            };

            for (node_idx, node) in constraints.nodes.iter().enumerate() {
                let mut record = Record { args: [F::ZERO; 8] };
                match node {
                    SymbolicExpressionNode::Variable(var) => match var.entry {
                        Entry::Preprocessed { .. } => {
                            record.args[..D_EF]
                                .copy_from_slice(expr_evals[node_idx].as_base_slice());
                            record.args[D_EF] = sort_idx;
                        }
                        Entry::Main { .. } => {
                            record.args[..D_EF]
                                .copy_from_slice(expr_evals[node_idx].as_base_slice());
                            record.args[D_EF] = sort_idx;
                        }
                        Entry::Permutation { .. } => unreachable!(),
                        Entry::Public => record.args[..D_EF]
                            .copy_from_slice(expr_evals[node_idx].as_base_slice()),
                        Entry::Challenge => unreachable!(),
                        Entry::Exposed => unreachable!(),
                    },
                    SymbolicExpressionNode::IsFirstRow => {
                        record.args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                        record.args[D_EF] = sort_idx;
                        record.args[D_EF + 1] = n_abs;
                        record.args[D_EF + 2] = n_sign_bit;
                    }
                    SymbolicExpressionNode::IsLastRow => {
                        record.args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                        record.args[D_EF] = sort_idx;
                        record.args[D_EF + 1] = n_abs;
                        record.args[D_EF + 2] = n_sign_bit;
                    }
                    SymbolicExpressionNode::IsTransition => {
                        record.args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                        record.args[D_EF] = sort_idx;
                        record.args[D_EF + 1] = n_abs;
                        record.args[D_EF + 2] = n_sign_bit;
                    }
                    SymbolicExpressionNode::Constant(_) => {}
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx,
                        degree_multiple: _,
                    } => {
                        record.args[..D_EF].copy_from_slice(expr_evals[*left_idx].as_base_slice());
                        record.args[D_EF..].copy_from_slice(expr_evals[*right_idx].as_base_slice());
                    }
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx,
                        degree_multiple: _,
                    } => {
                        record.args[..D_EF].copy_from_slice(expr_evals[*left_idx].as_base_slice());
                        record.args[D_EF..].copy_from_slice(expr_evals[*right_idx].as_base_slice());
                    }
                    SymbolicExpressionNode::Neg {
                        idx,
                        degree_multiple: _,
                    } => {
                        record.args[..D_EF].copy_from_slice(expr_evals[*idx].as_base_slice());
                    }
                    SymbolicExpressionNode::Mul {
                        left_idx,
                        right_idx,
                        degree_multiple: _,
                    } => {
                        record.args[..D_EF].copy_from_slice(expr_evals[*left_idx].as_base_slice());
                        record.args[D_EF..].copy_from_slice(expr_evals[*right_idx].as_base_slice());
                    }
                };
                records.push(Some(record));
            }
            for interaction in &vk.symbolic_constraints.interactions {
                let mut args = [F::ZERO; 8];
                args[..D_EF].copy_from_slice(expr_evals[interaction.count].as_base_slice());
                records.push(Some(Record { args }));

                for &node_idx in &interaction.message {
                    let mut args = [F::ZERO; 8];
                    args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                    records.push(Some(Record { args }));
                }
            }
        }
    }

    // records are ordered per proof; we now interleave them

    let num_valid_rows = records.len() / proofs.len();
    let height = num_valid_rows.next_power_of_two();
    let mut main_trace = F::zero_vec(main_width * height);

    main_trace
        .par_chunks_exact_mut(single_main_width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let row_idx = i / max_num_proofs;
            let proof_idx = i % max_num_proofs;

            if proof_idx >= proofs.len() {
                return;
            }
            if row_idx >= num_valid_rows {
                return;
            }

            let record_idx = proof_idx * num_valid_rows + row_idx;
            let record = &records[record_idx];

            if record.is_none() {
                return;
            }
            let record = record.as_ref().unwrap();

            let cols: &mut SingleMainSymbolicExpressionColumns<_> = chunk.borrow_mut();
            cols.is_present = F::ONE;
            cols.args = record.args;
        });

    RowMajorMatrix::new(main_trace, main_width)
}

#[derive(Debug, Clone, Copy, EnumIter, EnumCount)]
enum NodeKind {
    // Args: (col_idx, is_next)
    VarPreprocessed = 0,
    // Args: (col_idx, is_next)
    VarMain = 1,
    // Args: (pv_idx,)
    VarPublicValue = 2,
    // Args: ()
    SelIsFirst = 3,
    // Args: ()
    SelIsLast = 4,
    // Args: ()
    SelIsTransition = 5,
    // Args: (val,)
    Constant = 6,
    // Args: (left_node_idx, right_node_idx)
    Add = 7,
    // Args: (left_node_idx, right_node_idx)
    Sub = 8,
    // Args: (node_idx,)
    Neg = 9,
    // Args: (left_node_idx, right_node_idx)
    Mul = 10,
    // Args: (node_idx,)
    InteractionMult = 11,
    // Args: (node_idx, idx_in_message)
    InteractionMsgComp = 12,
}

/// Returns the cached trace
pub(crate) fn generate_symbolic_expr_cached_trace(
    child_vk: &MultiStarkVerifyingKeyV2,
) -> RowMajorMatrix<F> {
    // 3 var types: main, preprocessed, public value
    // 3 selectors: is_first, is_last, is_transition
    // 1 constant type
    // 4 gates: add, sub, neg, mul
    let encoder = Encoder::new(NodeKind::COUNT, 2, true);
    assert_eq!(encoder.width(), NUM_FLAGS);

    let cached_width = CachedSymbolicExpressionColumns::<F>::width();

    struct Record {
        kind: NodeKind,
        node_idx: usize,
        attrs: [usize; 3],
        is_constraint: bool,
        constraint_idx: usize,
        fanout: usize,
    }
    let mut records = vec![];

    let mut fanout_per_air = vec![];
    for vk in &child_vk.inner.per_air {
        let nodes = &vk.symbolic_constraints.constraints.nodes;
        let mut fanout = vec![0usize; nodes.len()];

        for node in nodes.iter() {
            match node {
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    fanout[*left_idx] += 1;
                    fanout[*right_idx] += 1;
                }
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    fanout[*left_idx] += 1;
                    fanout[*right_idx] += 1;
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    fanout[*idx] += 1;
                }
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    fanout[*left_idx] += 1;
                    fanout[*right_idx] += 1;
                }
                _ => {}
            }
        }
        for interaction in vk.symbolic_constraints.interactions.iter() {
            fanout[interaction.count] += 1;
            for &node_idx in &interaction.message {
                fanout[node_idx] += 1;
            }
        }
        fanout_per_air.push(fanout);
    }

    for (vk, fanout_per_node) in zip(child_vk.inner.per_air.iter(), fanout_per_air.into_iter()) {
        let constraints = &vk.symbolic_constraints.constraints;
        let constraint_idxs = &constraints.constraint_idx;

        #[cfg(debug_assertions)]
        {
            for i in 1..constraint_idxs.len() {
                debug_assert!(constraint_idxs[i - 1] < constraint_idxs[i]);
            }
        }

        let mut j = 0;

        for (node_idx, (node, &fanout)) in
            zip(constraints.nodes.iter(), fanout_per_node.iter()).enumerate()
        {
            if j < constraint_idxs.len() && constraint_idxs[j] < node_idx {
                j += 1;
            }
            let is_constraint = j < constraint_idxs.len() && constraint_idxs[j] == node_idx;

            let mut record = Record {
                kind: NodeKind::Constant,
                node_idx,
                attrs: [0; 3],
                is_constraint,
                constraint_idx: if is_constraint { 0 } else { j },
                fanout,
            };

            match node {
                SymbolicExpressionNode::Variable(var) => {
                    record.attrs[0] = var.index;
                    match var.entry {
                        Entry::Preprocessed { offset } => {
                            record.kind = NodeKind::VarPreprocessed;
                            record.attrs[1] = 1;
                            record.attrs[2] = offset;
                        }
                        Entry::Main { part_index, offset } => {
                            record.kind = NodeKind::VarMain;
                            record.attrs[1] = if part_index == 0 {
                                0
                            } else {
                                part_index + vk.preprocessed_data.is_some() as usize
                            };
                            record.attrs[2] = offset;
                        }
                        Entry::Permutation { .. } => unreachable!(),
                        Entry::Public => {
                            record.kind = NodeKind::VarPublicValue;
                        }
                        Entry::Challenge => unreachable!(),
                        Entry::Exposed => unreachable!(),
                    }
                }
                SymbolicExpressionNode::IsFirstRow => {
                    record.kind = NodeKind::SelIsFirst;
                }
                SymbolicExpressionNode::IsLastRow => {
                    record.kind = NodeKind::SelIsLast;
                }
                SymbolicExpressionNode::IsTransition => {
                    record.kind = NodeKind::SelIsTransition;
                }
                SymbolicExpressionNode::Constant(val) => {
                    record.kind = NodeKind::Constant;
                    record.attrs[0] = val.as_canonical_u32() as usize;
                }
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    degree_multiple: _,
                } => {
                    record.kind = NodeKind::Add;
                    record.attrs[0] = *left_idx;
                    record.attrs[1] = *right_idx;
                }
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    degree_multiple: _,
                } => {
                    record.kind = NodeKind::Sub;
                    record.attrs[0] = *left_idx;
                    record.attrs[1] = *right_idx;
                }
                SymbolicExpressionNode::Neg {
                    idx,
                    degree_multiple: _,
                } => {
                    record.kind = NodeKind::Neg;
                    record.attrs[0] = *idx;
                }
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    degree_multiple: _,
                } => {
                    record.kind = NodeKind::Mul;
                    record.attrs[0] = *left_idx;
                    record.attrs[1] = *right_idx;
                }
            };
            records.push(record);
        }
        for (interaction_idx, interaction) in
            vk.symbolic_constraints.interactions.iter().enumerate()
        {
            records.push(Record {
                kind: NodeKind::InteractionMult,
                node_idx: interaction_idx,
                attrs: [interaction.count, 0, 0],
                is_constraint: false,
                constraint_idx: 0,
                fanout: 0,
            });
            for (idx_in_message, &node_idx) in interaction.message.iter().enumerate() {
                records.push(Record {
                    kind: NodeKind::InteractionMsgComp,
                    node_idx: interaction_idx,
                    attrs: [node_idx, idx_in_message, 0],
                    is_constraint: false,
                    constraint_idx: 0,
                    fanout: 0,
                });
            }
        }
    }

    let height = records.len().next_power_of_two();
    let mut cached_trace = F::zero_vec(cached_width * height);
    cached_trace
        .par_chunks_exact_mut(cached_width)
        .zip(records)
        .for_each(|(row, record)| {
            let cols: &mut CachedSymbolicExpressionColumns<_> = row.borrow_mut();

            for (i, x) in encoder
                .get_flag_pt(record.kind as usize)
                .into_iter()
                .enumerate()
            {
                cols.flags[i] = F::from_canonical_u32(x);
                cols.node_or_interaction_idx = F::from_canonical_usize(record.node_idx);
                cols.attrs = record.attrs.map(F::from_canonical_usize);
                cols.is_constraint = F::from_bool(record.is_constraint);
                cols.constraint_idx = F::from_canonical_usize(record.constraint_idx);
                cols.fanout = F::from_canonical_usize(record.fanout);
            }
        });

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
                rows.push(ColumnRowInfo {
                    is_first: rows.len() == initial_len,
                    is_last: false,
                    proof_idx: pidx,
                    tidx: main_tidx[sort_idx] + col * 2 * D_EF,
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

#[derive(Copy, Clone)]
#[repr(C)]
struct InteractionsFoldingRecord {
    sort_idx: usize,
    interaction_idx: usize,
    node_idx: usize,
    idx_in_message: usize,
    has_interactions: bool,
    is_first_in_air: bool,
    is_last_in_air: bool,
    is_mult: bool,
}

struct InteractionsFoldingBlob {
    records: MultiProofVecVec<InteractionsFoldingRecord>,
}

fn generate_interactions_folding_blob(
    vk: &MultiStarkVerifyingKeyV2,
    preflights: &[Preflight],
) -> InteractionsFoldingBlob {
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

    let mut records = MultiProofVecVec::new();
    for preflight in preflights.iter() {
        let vdata = &preflight.proof_shape.sorted_trace_vdata;
        for (sort_idx, (air_idx, _)) in vdata.iter().enumerate() {
            let inters = &interactions[*air_idx];
            if inters.is_empty() {
                records.push(InteractionsFoldingRecord {
                    sort_idx,
                    interaction_idx: 0,
                    node_idx: 0,
                    idx_in_message: 0,
                    has_interactions: false,
                    is_first_in_air: true,
                    is_last_in_air: true,
                    is_mult: false,
                });
            } else {
                for (interaction_idx, inter) in inters.iter().enumerate() {
                    records.push(InteractionsFoldingRecord {
                        sort_idx,
                        interaction_idx,
                        node_idx: inter.count,
                        idx_in_message: 0,
                        has_interactions: true,
                        is_first_in_air: interaction_idx == 0,
                        is_last_in_air: false,
                        is_mult: true,
                    });

                    for (j, &node_idx) in inter.message.iter().enumerate() {
                        records.push(InteractionsFoldingRecord {
                            sort_idx,
                            interaction_idx,
                            node_idx,
                            idx_in_message: j,
                            has_interactions: true,
                            is_first_in_air: false,
                            is_last_in_air: false,
                            is_mult: false,
                        });
                    }

                    records.push(InteractionsFoldingRecord {
                        sort_idx,
                        interaction_idx,
                        node_idx: inter.bus_index as usize,
                        idx_in_message: inter.message.len() + 1,
                        has_interactions: true,
                        is_first_in_air: false,
                        is_last_in_air: interaction_idx + 1 == inters.len(),
                        is_mult: false,
                    });
                }
            }
        }
        records.end_proof();
    }
    InteractionsFoldingBlob { records }
}

pub(in crate::batch_constraint) fn generate_interactions_folding_trace(
    vk: &MultiStarkVerifyingKeyV2,
    expr_blob: &BatchConstraintBlobCpu,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = InteractionsFoldingCols::<F>::width();

    let blob = generate_interactions_folding_blob(vk, preflights);

    let total_height = blob.records.len();
    let padding_height = total_height.next_power_of_two();
    let mut trace = vec![F::ZERO; padding_height * width];

    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let beta_tidx = preflight.proof_shape.post_tidx + 2 + D_EF;
        let beta_slice = &preflight.transcript.values()[beta_tidx..beta_tidx + D_EF];
        let records = &blob.records[pidx];

        let node_claims = &expr_blob.expr_evals[pidx];

        trace[cur_height * width..(cur_height + records.len()) * width]
            .par_chunks_exact_mut(width)
            .zip(records.par_iter())
            .for_each(|(chunk, record)| {
                let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                let air_idx = preflight.proof_shape.sorted_trace_vdata[record.sort_idx].0;
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.beta_tidx = F::from_canonical_usize(beta_tidx);
                cols.sort_idx = F::from_canonical_usize(record.sort_idx);
                cols.interaction_idx = F::from_canonical_usize(record.interaction_idx);
                cols.node_idx = F::from_canonical_usize(record.node_idx);
                cols.has_interactions = F::from_bool(record.has_interactions);
                cols.is_first_in_air = F::from_bool(record.is_first_in_air);
                cols.is_first_in_message = F::from_bool(record.is_mult || !record.has_interactions);
                cols.idx_in_message = F::from_canonical_usize(record.idx_in_message);
                cols.loop_aux.is_transition[0] = F::ONE;
                cols.loop_aux.is_transition[1] = F::from_bool(!record.is_last_in_air);
                cols.value
                    .copy_from_slice(node_claims[air_idx][record.node_idx].as_base_slice());
                cols.beta.copy_from_slice(beta_slice);
            });

        // Setting `cur_sum`
        let mut cur_sum = EF::ZERO;
        let beta = EF::from_base_slice(beta_slice);
        trace[cur_height * width..(cur_height + records.len()) * width]
            .chunks_exact_mut(width)
            .rev()
            .for_each(|chunk| {
                let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                if cols.is_first_in_message == F::ONE {
                    cols.cur_sum.copy_from_slice(&cols.value);
                    cur_sum = EF::ZERO;
                } else {
                    cur_sum = cur_sum * beta + EF::from_base_slice(&cols.value);
                    cols.cur_sum.copy_from_slice(cur_sum.as_base_slice());
                }
            });

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
