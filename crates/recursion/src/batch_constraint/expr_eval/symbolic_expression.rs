use core::{array, cmp::min, iter::zip};
use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq};
use openvm_stark_backend::{
    air_builders::{
        PartitionedAirBuilder,
        symbolic::{SymbolicExpressionNode, symbolic_variable::Entry},
    },
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, BaseAir};
use p3_field::{
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField,
    extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{Squarable, eval_eq_uni_at_one},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

use crate::{
    batch_constraint::{
        BatchConstraintBlobCpu,
        bus::{
            ConstraintsFoldingBus, ConstraintsFoldingMessage, EqNegInternalBus, ExpressionClaimBus,
            InteractionsFoldingBus, InteractionsFoldingMessage, SymbolicExpressionBus,
            SymbolicExpressionMessage,
        },
    },
    bus::{
        AirShapeBus, AirShapeBusMessage, ColumnClaimsBus, ColumnClaimsMessage, HyperdimBus,
        HyperdimBusMessage, PublicValuesBus, PublicValuesBusMessage, SelHypercubeBus,
        SelHypercubeBusMessage, SelUniBus, SelUniBusMessage,
    },
    system::Preflight,
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_subtract, scalar_subtract_ext_field,
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
    // Dynamic arguments. For Add/Mul/Sub, this splits into two extension-field elements.
    // For selectors:
    //   args[0..D_EF)   = sel_uni witness (base or rotated depending on selector type).
    //   args[D_EF..2*D_EF) = eq-prefix witness (prod r_i or prod (1-r_i)).
    args: [T; 2 * D_EF],
    sort_idx: T,
    n_abs: T,
    is_n_neg: T,
}

pub struct SymbolicExpressionAir {
    pub expr_bus: SymbolicExpressionBus,
    pub claim_bus: ExpressionClaimBus,
    pub hyperdim_bus: HyperdimBus,
    pub air_shape_bus: AirShapeBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub interactions_folding_bus: InteractionsFoldingBus,
    pub constraints_folding_bus: ConstraintsFoldingBus,
    pub public_values_bus: PublicValuesBus,
    pub sel_hypercube_bus: SelHypercubeBus,
    pub sel_uni_bus: SelUniBus,
    pub eq_neg_internal_bus: EqNegInternalBus,

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

            let arg_ef0: [AB::Var; D_EF] = cols.args[..D_EF].try_into().unwrap();
            let arg_ef1: [AB::Var; D_EF] = cols.args[D_EF..2 * D_EF].try_into().unwrap();

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
                    NodeKind::SelIsFirst => ext_field_multiply(arg_ef0, arg_ef1),
                    NodeKind::SelIsLast => ext_field_multiply(arg_ef0, arg_ef1),
                    NodeKind::SelIsTransition => scalar_subtract_ext_field(
                        AB::Expr::ONE,
                        ext_field_multiply(arg_ef0, arg_ef1),
                    ),
                    NodeKind::VarPreprocessed
                    | NodeKind::VarMain
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

            let is_var = enc.contains_flag::<AB>(
                &flags,
                &[NodeKind::VarMain, NodeKind::VarPreprocessed].map(|x| x as usize),
            );
            self.column_claims_bus.receive(
                builder,
                proof_idx,
                ColumnClaimsMessage {
                    sort_idx: cols.sort_idx.into(),
                    part_idx: cached_cols.attrs[1].into(),
                    col_idx: cached_cols.attrs[0].into(),
                    claim: array::from_fn(|i| cols.args[i].into()),
                    is_rot: cached_cols.attrs[2].into(),
                },
                is_var * cols.is_present,
            );
            self.public_values_bus.receive(
                builder,
                proof_idx,
                PublicValuesBusMessage {
                    air_idx: cached_cols.air_idx,
                    pv_idx: cached_cols.attrs[0],
                    value: cols.args[0],
                },
                enc.get_flag_expr::<AB>(NodeKind::VarPublicValue as usize, &flags)
                    * cols.is_present,
            );
            self.air_shape_bus.receive(
                builder,
                proof_idx,
                AirShapeBusMessage {
                    sort_idx: cols.sort_idx.into(),
                    property_idx: AB::Expr::ZERO,
                    value: cached_cols.air_idx.into(),
                },
                cols.is_present,
            );
            self.hyperdim_bus.receive(
                builder,
                proof_idx,
                HyperdimBusMessage {
                    sort_idx: cols.sort_idx,
                    n_abs: cols.n_abs,
                    n_sign_bit: cols.is_n_neg,
                },
                cols.is_present,
            );
            // Selector
            {
                let is_sel = enc.contains_flag::<AB>(
                    &flags,
                    &[
                        NodeKind::SelIsFirst,
                        NodeKind::SelIsLast,
                        NodeKind::SelIsTransition,
                    ]
                    .map(|x| x as usize),
                );

                let is_first = enc.get_flag_expr::<AB>(NodeKind::SelIsFirst as usize, &flags);
                self.sel_uni_bus.receive(
                    builder,
                    proof_idx,
                    SelUniBusMessage {
                        n: AB::Expr::NEG_ONE * cols.n_abs * cols.is_n_neg,
                        is_first: is_first.clone(),
                        value: arg_ef0.map(Into::into),
                    },
                    cols.is_present * is_sel.clone(),
                );
                self.sel_hypercube_bus.receive(
                    builder,
                    proof_idx,
                    SelHypercubeBusMessage {
                        n: cols.n_abs.into(),
                        is_first: is_first.clone(),
                        value: arg_ef1.map(Into::into),
                    },
                    cols.is_present * is_sel.clone() * (AB::Expr::ONE - cols.is_n_neg),
                );
                assert_array_eq(
                    &mut builder.when(cols.is_present * is_sel.clone() * cols.is_n_neg),
                    arg_ef1,
                    [
                        AB::Expr::ONE,
                        AB::Expr::ZERO,
                        AB::Expr::ZERO,
                        AB::Expr::ZERO,
                    ],
                );
            }
            let is_mult = enc.get_flag_expr::<AB>(NodeKind::InteractionMult as usize, &flags);
            let is_interaction = enc.contains_flag::<AB>(
                &flags,
                &[NodeKind::InteractionMult, NodeKind::InteractionMsgComp].map(|x| x as usize),
            );
            self.interactions_folding_bus.send(
                builder,
                proof_idx,
                InteractionsFoldingMessage {
                    air_idx: cached_cols.air_idx.into(),
                    interaction_idx: cached_cols.node_or_interaction_idx.into(),
                    is_mult,
                    idx_in_message: cached_cols.attrs[1].into(),
                    value: value.clone(),
                },
                is_interaction * cols.is_present,
            );
            self.constraints_folding_bus.send(
                builder,
                proof_idx,
                ConstraintsFoldingMessage {
                    air_idx: cached_cols.air_idx.into(),
                    constraint_idx: cached_cols.constraint_idx.into(),
                    value: value.clone(),
                },
                cached_cols.is_constraint * cols.is_present,
            );
        }
    }
}

/// Returns the common main trace.
#[tracing::instrument(name = "generate_trace(SymbolicExpressionAir)", skip_all)]
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
        args: [F; 2 * D_EF],
        sort_idx: usize,
        n_abs: usize,
        is_n_neg: usize,
    }
    let mut records = vec![];

    for (proof_idx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        let rs = &preflight.batch_constraint.sumcheck_rnd;
        let (&rs_0, rs_rest) = rs.split_first().unwrap();
        let mut is_first_uni_by_log_height = vec![];
        let mut is_last_uni_by_log_height = vec![];

        for (log_height, &r_pow) in rs_0
            .exp_powers_of_2()
            .take(params.l_skip + 1)
            .collect::<Vec<_>>()
            .iter()
            .rev()
            .enumerate()
        {
            is_first_uni_by_log_height.push(eval_eq_uni_at_one(log_height, r_pow));
            is_last_uni_by_log_height.push(eval_eq_uni_at_one(
                log_height,
                r_pow * F::two_adic_generator(log_height),
            ));
        }
        let mut is_first_mle_by_n = vec![EF::ONE];
        let mut is_last_mle_by_n = vec![EF::ONE];
        for (i, &r) in rs_rest.iter().enumerate() {
            is_first_mle_by_n.push(is_first_mle_by_n[i] * (EF::ONE - r));
            is_last_mle_by_n.push(is_last_mle_by_n[i] * r);
        }

        for (air_idx, vk) in child_vk.inner.per_air.iter().enumerate() {
            let constraints = &vk.symbolic_constraints.constraints;
            let expr_evals = &blob.expr_evals[proof_idx][air_idx];

            // TODO: don't do any pushes at all for absent traces
            if expr_evals.is_empty() {
                let n = constraints.nodes.len()
                    + vk.symbolic_constraints
                        .interactions
                        .iter()
                        .map(|i| 1 + i.message.len())
                        .sum::<usize>()
                    + vk.unused_variables.len();

                records.resize_with(records.len() + n, || None);
                continue;
            }

            let sort_idx = preflight
                .proof_shape
                .sorted_trace_vdata
                .iter()
                .position(|(idx, _)| *idx == air_idx)
                .unwrap();

            // TODO sort_idx in trace
            let log_height = proof.trace_vdata[air_idx].as_ref().unwrap().log_height;
            let (n_abs, is_n_neg) = if log_height < params.l_skip {
                (params.l_skip - log_height, 1)
            } else {
                (log_height - params.l_skip, 0)
            };

            for (node_idx, node) in constraints.nodes.iter().enumerate() {
                let mut record = Record {
                    args: [F::ZERO; 2 * D_EF],
                    sort_idx,
                    n_abs,
                    is_n_neg,
                };
                match node {
                    SymbolicExpressionNode::Variable(var) => match var.entry {
                        Entry::Preprocessed { .. } => {
                            record.args[..D_EF]
                                .copy_from_slice(expr_evals[node_idx].as_base_slice());
                        }
                        Entry::Main { .. } => {
                            record.args[..D_EF]
                                .copy_from_slice(expr_evals[node_idx].as_base_slice());
                        }
                        Entry::Permutation { .. } => unreachable!(),
                        Entry::Public => record.args[..D_EF]
                            .copy_from_slice(expr_evals[node_idx].as_base_slice()),
                        Entry::Challenge => unreachable!(),
                        Entry::Exposed => unreachable!(),
                    },
                    SymbolicExpressionNode::IsFirstRow => {
                        record.args[..D_EF].copy_from_slice(
                            is_first_uni_by_log_height[min(log_height, params.l_skip)]
                                .as_base_slice(),
                        );
                        record.args[D_EF..2 * D_EF].copy_from_slice(
                            is_first_mle_by_n[log_height.saturating_sub(params.l_skip)]
                                .as_base_slice(),
                        );
                    }
                    SymbolicExpressionNode::IsLastRow | SymbolicExpressionNode::IsTransition => {
                        record.args[..D_EF].copy_from_slice(
                            is_last_uni_by_log_height[min(log_height, params.l_skip)]
                                .as_base_slice(),
                        );
                        record.args[D_EF..2 * D_EF].copy_from_slice(
                            is_last_mle_by_n[log_height.saturating_sub(params.l_skip)]
                                .as_base_slice(),
                        );
                    }
                    SymbolicExpressionNode::Constant(_) => {}
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx,
                        degree_multiple: _,
                    } => {
                        record.args[..D_EF].copy_from_slice(expr_evals[*left_idx].as_base_slice());
                        record.args[D_EF..2 * D_EF]
                            .copy_from_slice(expr_evals[*right_idx].as_base_slice());
                    }
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx,
                        degree_multiple: _,
                    } => {
                        record.args[..D_EF].copy_from_slice(expr_evals[*left_idx].as_base_slice());
                        record.args[D_EF..2 * D_EF]
                            .copy_from_slice(expr_evals[*right_idx].as_base_slice());
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
                        record.args[D_EF..2 * D_EF]
                            .copy_from_slice(expr_evals[*right_idx].as_base_slice());
                    }
                };
                records.push(Some(record));
            }
            for interaction in &vk.symbolic_constraints.interactions {
                let mut args = [F::ZERO; 2 * D_EF];
                args[..D_EF].copy_from_slice(expr_evals[interaction.count].as_base_slice());
                records.push(Some(Record {
                    args,
                    sort_idx,
                    n_abs,
                    is_n_neg,
                }));

                for &node_idx in &interaction.message {
                    let mut args = [F::ZERO; 2 * D_EF];
                    args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                    records.push(Some(Record {
                        args,
                        sort_idx,
                        n_abs,
                        is_n_neg,
                    }));
                }
            }
            let mut node_idx = constraints.nodes.len();
            for unused_var in &vk.unused_variables {
                match unused_var.entry {
                    Entry::Preprocessed { .. } => {
                        let mut args = [F::ZERO; 2 * D_EF];
                        args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                        records.push(Some(Record {
                            args,
                            sort_idx,
                            n_abs,
                            is_n_neg,
                        }));
                    }
                    Entry::Main { .. } => {
                        let mut args = [F::ZERO; 2 * D_EF];
                        args[..D_EF].copy_from_slice(expr_evals[node_idx].as_base_slice());
                        records.push(Some(Record {
                            args,
                            sort_idx,
                            n_abs,
                            is_n_neg,
                        }));
                    }
                    Entry::Permutation { .. }
                    | Entry::Public
                    | Entry::Challenge
                    | Entry::Exposed => {
                        unreachable!()
                    }
                }
                node_idx += 1;
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
            cols.sort_idx = F::from_canonical_usize(record.sort_idx);
            cols.n_abs = F::from_canonical_usize(record.n_abs);
            cols.is_n_neg = F::from_canonical_usize(record.is_n_neg);
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
#[tracing::instrument(name = "generate_cached_trace(SymbolicExpressionAir)", skip_all)]
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
        air_idx: usize,
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

    for (air_idx, (vk, fanout_per_node)) in
        zip(child_vk.inner.per_air.iter(), fanout_per_air.into_iter()).enumerate()
    {
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
                air_idx,
                node_idx,
                attrs: [0; 3],
                is_constraint,
                constraint_idx: if !is_constraint { 0 } else { j },
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
                            record.attrs[1] = vk.dag_main_part_index_to_commit_index(part_index);
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
                air_idx,
                node_idx: interaction_idx,
                attrs: [interaction.count, 0, 0],
                is_constraint: false,
                constraint_idx: 0,
                fanout: 0,
            });
            for (idx_in_message, &node_idx) in interaction.message.iter().enumerate() {
                records.push(Record {
                    kind: NodeKind::InteractionMsgComp,
                    air_idx,
                    node_idx: interaction_idx,
                    attrs: [node_idx, idx_in_message, 0],
                    is_constraint: false,
                    constraint_idx: 0,
                    fanout: 0,
                });
            }
        }
        let mut node_idx = constraints.nodes.len();
        for unused_var in &vk.unused_variables {
            let record = match unused_var.entry {
                Entry::Preprocessed { offset } => Record {
                    kind: NodeKind::VarPreprocessed,
                    air_idx,
                    node_idx,
                    attrs: [unused_var.index, 1, offset],
                    is_constraint: false,
                    constraint_idx: 0,
                    fanout: 0,
                },
                Entry::Main { part_index, offset } => {
                    let part = vk.dag_main_part_index_to_commit_index(part_index);
                    Record {
                        kind: NodeKind::VarMain,
                        air_idx,
                        node_idx,
                        attrs: [unused_var.index, part, offset],
                        is_constraint: false,
                        constraint_idx: 0,
                        fanout: 0,
                    }
                }
                Entry::Permutation { .. } | Entry::Public | Entry::Challenge | Entry::Exposed => {
                    unreachable!()
                }
            };
            node_idx += 1;
            records.push(record);
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
                cols.air_idx = F::from_canonical_usize(record.air_idx);
                cols.node_or_interaction_idx = F::from_canonical_usize(record.node_idx);
                cols.attrs = record.attrs.map(F::from_canonical_usize);
                cols.is_constraint = F::from_bool(record.is_constraint);
                cols.constraint_idx = F::from_canonical_usize(record.constraint_idx);
                cols.fanout = F::from_canonical_usize(record.fanout);
            }
        });

    RowMajorMatrix::new(cached_trace, cached_width)
}
