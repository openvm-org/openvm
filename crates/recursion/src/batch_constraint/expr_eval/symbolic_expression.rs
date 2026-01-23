use core::{array, cmp::min, iter::zip};
use std::borrow::{Borrow, BorrowMut};

use itertools::{Itertools, fold};
use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq};
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
    FieldAlgebra, FieldExtensionAlgebra, PrimeField32, TwoAdicField,
    extension::BinomiallyExtendable,
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    D_EF, DIGEST_SIZE, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{Squarable, eval_eq_uni_at_one},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

use crate::{
    batch_constraint::bus::{
        ConstraintsFoldingBus, ConstraintsFoldingMessage, EqNegInternalBus, ExpressionClaimBus,
        InteractionsFoldingBus, InteractionsFoldingMessage, SymbolicExpressionBus,
        SymbolicExpressionMessage,
    },
    bus::{
        AirShapeBus, AirShapeBusMessage, ColumnClaimsBus, ColumnClaimsMessage, DagCommitBus,
        DagCommitBusMessage, HyperdimBus, HyperdimBusMessage, PublicValuesBus,
        PublicValuesBusMessage, SelHypercubeBus, SelHypercubeBusMessage, SelUniBus,
        SelUniBusMessage,
    },
    system::Preflight,
    utils::{
        MultiVecWithBounds, base_to_ext, ext_field_add, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_subtract, scalar_subtract_ext_field,
    },
};
const NUM_FLAGS: usize = 4;
const ENCODER_MAX_DEGREE: u32 = 2;
const FLAG_MODULUS: u32 = ENCODER_MAX_DEGREE + 1;

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
    pub dag_commit_bus: DagCommitBus,

    pub cnt_proofs: usize,
    pub has_cached: bool,
}

impl<F> BaseAirWithPublicValues<F> for SymbolicExpressionAir {}
impl<F> PartitionedBaseAir<F> for SymbolicExpressionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        if self.has_cached {
            vec![CachedSymbolicExpressionColumns::<F>::width()]
        } else {
            vec![]
        }
    }

    fn common_main_width(&self) -> usize {
        SingleMainSymbolicExpressionColumns::<F>::width() * self.cnt_proofs
            + if self.has_cached {
                0
            } else {
                1 + CachedSymbolicExpressionColumns::<F>::width()
            }
    }
}

impl<F> BaseAir<F> for SymbolicExpressionAir {
    fn width(&self) -> usize {
        CachedSymbolicExpressionColumns::<F>::width()
            + SingleMainSymbolicExpressionColumns::<F>::width() * self.cnt_proofs
            + if self.has_cached { 0 } else { 1 }
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for SymbolicExpressionAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main_local = builder.common_main().row_slice(0).to_vec();

        let single_main_width = SingleMainSymbolicExpressionColumns::<AB::Var>::width();
        let cached_width = CachedSymbolicExpressionColumns::<AB::Var>::width();

        let cached_local_vec = self
            .has_cached
            .then(|| builder.cached_mains()[0].row_slice(0).to_vec());

        let (cached_cols, main_slice) = if self.has_cached {
            let cached_slice = cached_local_vec.as_ref().unwrap().as_slice();
            let main_slice = main_local.as_slice();
            let cached_cols: &CachedSymbolicExpressionColumns<AB::Var> = cached_slice.borrow();
            (cached_cols, main_slice)
        } else {
            // No cached trace: common main is prefixed by `row_idx` and then cached columns
            let main_next = builder.common_main().row_slice(1).to_vec();

            let (prefix_local, rest_local) = main_local.as_slice().split_at(1);
            let (prefix_next, _rest_next) = main_next.as_slice().split_at(1);
            let (cached_slice, main_slice) = rest_local.split_at(cached_width);
            let cached_cols: &CachedSymbolicExpressionColumns<AB::Var> = cached_slice.borrow();

            let row_idx_local = prefix_local[0];
            let row_idx_next = prefix_next[0];

            builder.when_first_row().assert_zero(row_idx_local);
            builder
                .when_transition()
                .assert_eq(row_idx_next, row_idx_local + AB::Expr::ONE);

            debug_assert_eq!(FLAG_MODULUS, 3);
            builder.assert_bool(cached_cols.is_constraint);
            for flag in cached_cols.flags {
                builder.assert_tern(flag);
            }

            let cached_slice_expr = cached_slice
                .to_vec()
                .into_iter()
                .map(Into::into)
                .collect_vec();

            self.dag_commit_bus.send(
                builder,
                AB::Expr::ZERO,
                DagCommitBusMessage {
                    idx: row_idx_local.into(),
                    values: cached_symbolic_expr_cols_to_digest(&cached_slice_expr),
                },
                AB::Expr::ONE,
            );

            (cached_cols, main_slice)
        };

        let main_cols: Vec<&SingleMainSymbolicExpressionColumns<AB::Var>> = main_slice
            .chunks(single_main_width)
            .map(|chunk| chunk.borrow())
            .collect();

        let enc = Encoder::new(NodeKind::COUNT, ENCODER_MAX_DEGREE, true);
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
            builder.when(cols.is_n_neg).assert_one(cols.is_present);

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
                    is_sel.clone() * (cols.is_present - cols.is_n_neg),
                );
                assert_array_eq(
                    &mut builder.when(is_sel.clone() * cols.is_n_neg),
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
#[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
pub(in crate::batch_constraint) fn generate_symbolic_expr_common_trace(
    child_vk: &MultiStarkVerifyingKeyV2,
    proofs: &[&Proof],
    preflights: &[&Preflight],
    max_num_proofs: usize,
    has_cached: bool,
    expr_evals: &MultiVecWithBounds<EF, 2>,
) -> RowMajorMatrix<F> {
    let l_skip = child_vk.inner.params.l_skip;

    let single_main_width = SingleMainSymbolicExpressionColumns::<F>::width();
    let cached_width = CachedSymbolicExpressionColumns::<F>::width();
    let main_width =
        single_main_width * max_num_proofs + if has_cached { 0 } else { 1 + cached_width };

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
            .take(l_skip + 1)
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
            let expr_evals = &expr_evals[[proof_idx, air_idx]];

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
            let (n_abs, is_n_neg) = if log_height < l_skip {
                (l_skip - log_height, 1)
            } else {
                (log_height - l_skip, 0)
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
                            is_first_uni_by_log_height[min(log_height, l_skip)].as_base_slice(),
                        );
                        record.args[D_EF..2 * D_EF].copy_from_slice(
                            is_first_mle_by_n[log_height.saturating_sub(l_skip)].as_base_slice(),
                        );
                    }
                    SymbolicExpressionNode::IsLastRow | SymbolicExpressionNode::IsTransition => {
                        record.args[..D_EF].copy_from_slice(
                            is_last_uni_by_log_height[min(log_height, l_skip)].as_base_slice(),
                        );
                        record.args[D_EF..2 * D_EF].copy_from_slice(
                            is_last_mle_by_n[log_height.saturating_sub(l_skip)].as_base_slice(),
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

    let (encoder, cached_records) = if has_cached {
        (None, None)
    } else {
        let encoder = Encoder::new(NodeKind::COUNT, ENCODER_MAX_DEGREE, true);
        assert_eq!(encoder.width(), NUM_FLAGS);
        let cached_records = build_cached_records(child_vk);
        debug_assert_eq!(cached_records.len(), num_valid_rows);
        (Some(encoder), Some(cached_records))
    };

    main_trace
        .par_chunks_exact_mut(main_width)
        .enumerate()
        .for_each(|(row_idx, row)| {
            // When has_cached == false we prepend row_idx for all rows
            let row_offset = if has_cached {
                0
            } else {
                row[0] = F::from_canonical_usize(row_idx);
                1
            };

            if row_idx >= num_valid_rows {
                return;
            }

            let row = &mut row[row_offset..];

            let main_offset = if has_cached {
                0
            } else {
                let record = &cached_records.as_ref().unwrap()[row_idx];
                let encoder = encoder.as_ref().unwrap();
                let cols: &mut CachedSymbolicExpressionColumns<_> =
                    row[..cached_width].borrow_mut();

                for (i, x) in encoder
                    .get_flag_pt(record.kind as usize)
                    .into_iter()
                    .enumerate()
                {
                    cols.flags[i] = F::from_canonical_u32(x);
                }
                cols.air_idx = F::from_canonical_usize(record.air_idx);
                cols.node_or_interaction_idx = F::from_canonical_usize(record.node_idx);
                cols.attrs = record.attrs.map(F::from_canonical_usize);
                cols.is_constraint = F::from_bool(record.is_constraint);
                cols.constraint_idx = F::from_canonical_usize(record.constraint_idx);
                cols.fanout = F::from_canonical_usize(record.fanout);

                cached_width
            };

            for proof_idx in 0..max_num_proofs {
                if proof_idx >= proofs.len() {
                    continue;
                }

                let record_idx = proof_idx * num_valid_rows + row_idx;
                let Some(record) = records[record_idx].as_ref() else {
                    continue;
                };

                let start = main_offset + proof_idx * single_main_width;
                let end = start + single_main_width;
                let cols: &mut SingleMainSymbolicExpressionColumns<_> =
                    row[start..end].borrow_mut();
                cols.is_present = F::ONE;
                cols.args = record.args;
                cols.sort_idx = F::from_canonical_usize(record.sort_idx);
                cols.n_abs = F::from_canonical_usize(record.n_abs);
                cols.is_n_neg = F::from_canonical_usize(record.is_n_neg);
            }
        });

    RowMajorMatrix::new(main_trace, main_width)
}

#[derive(Debug, Clone, Copy, EnumIter, EnumCount)]
pub(crate) enum NodeKind {
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct CachedRecord {
    kind: NodeKind,
    air_idx: usize,
    pub(crate) node_idx: usize,
    pub(crate) attrs: [usize; 3],
    pub(crate) is_constraint: bool,
    pub(crate) constraint_idx: usize,
    pub(crate) fanout: usize,
}

pub(crate) fn build_cached_records(child_vk: &MultiStarkVerifyingKeyV2) -> Vec<CachedRecord> {
    let mut fanout_per_air = Vec::with_capacity(child_vk.inner.per_air.len());
    for vk in &child_vk.inner.per_air {
        let nodes = &vk.symbolic_constraints.constraints.nodes;
        let mut fanout = vec![0usize; nodes.len()];

        for node in nodes.iter() {
            match node {
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                }
                | SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                }
                | SymbolicExpressionNode::Mul {
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

    let mut records = vec![];
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

            let mut record = CachedRecord {
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
            records.push(CachedRecord {
                kind: NodeKind::InteractionMult,
                air_idx,
                node_idx: interaction_idx,
                attrs: [interaction.count, 0, 0],
                is_constraint: false,
                constraint_idx: 0,
                fanout: 0,
            });
            for (idx_in_message, &node_idx) in interaction.message.iter().enumerate() {
                records.push(CachedRecord {
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
                Entry::Preprocessed { offset } => CachedRecord {
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
                    CachedRecord {
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
    records
}

/// Returns the cached trace
#[tracing::instrument(
    name = "generate_cached_trace",
    skip_all,
    fields(air = "SymbolicExpressionAir")
)]
pub(crate) fn generate_symbolic_expr_cached_trace(
    child_vk: &MultiStarkVerifyingKeyV2,
) -> RowMajorMatrix<F> {
    // 3 var types: main, preprocessed, public value
    // 3 selectors: is_first, is_last, is_transition
    // 1 constant type
    // 4 gates: add, sub, neg, mul
    let encoder = Encoder::new(NodeKind::COUNT, ENCODER_MAX_DEGREE, true);
    assert_eq!(encoder.width(), NUM_FLAGS);

    let cached_width = CachedSymbolicExpressionColumns::<F>::width();
    let records = build_cached_records(child_vk);

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

/// Compress a CachedSymbolicExpressionColumns row into a digest. The 0-index element
/// is a composition of the ternary (i.e. in [0, FLAG_MODULUS)) Encoder flags and the
/// boolean flag is_constraint, all of which should fit into any 13-bit (or higher)
/// Field. Each of the remaining cached columns is its own digest element.
///
/// WARNING: To use this in an AIR you MUST constrain that is_constraint is boolean
/// and that each flag is in [0, FLAG_MODULUS)
pub fn cached_symbolic_expr_cols_to_digest<F: FieldAlgebra>(cached_cols: &[F]) -> [F; DIGEST_SIZE] {
    let cached_cols: &CachedSymbolicExpressionColumns<_> = cached_cols.borrow();
    let mut ret = [F::ZERO; DIGEST_SIZE];
    ret[0] = fold(
        cached_cols.flags.iter().enumerate(),
        cached_cols.is_constraint.clone(),
        |acc, (pow_exp, flag)| {
            acc + (flag.clone() * F::from_canonical_u32(FLAG_MODULUS.pow(pow_exp as u32) << 1))
        },
    );
    ret[1] = cached_cols.air_idx.clone();
    ret[2] = cached_cols.node_or_interaction_idx.clone();
    ret[3] = cached_cols.attrs[0].clone();
    ret[4] = cached_cols.attrs[1].clone();
    ret[5] = cached_cols.attrs[2].clone();
    ret[6] = cached_cols.fanout.clone();
    ret[7] = cached_cols.constraint_idx.clone();
    ret
}

#[cfg(feature = "cuda")]
pub(in crate::batch_constraint) mod cuda {

    use cuda_backend_v2::{EF, F};
    use openvm_cuda_backend::base::DeviceMatrix;
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};

    use super::*;
    use crate::{
        batch_constraint::{cuda_abi::sym_expr_common_tracegen, cuda_utils::*},
        cuda::{preflight::PreflightGpu, proof::ProofGpu, to_device_or_nullptr},
    };

    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    pub fn generate_sym_expr_trace(
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[ProofGpu],
        preflights: &[PreflightGpu],
        max_num_proofs: usize,
        has_cached: bool,
        expr_evals: &MultiVecWithBounds<EF, 2>,
    ) -> DeviceMatrix<F> {
        debug_assert_eq!(proofs.len(), preflights.len());

        let num_airs = child_vk.inner.per_air.len();

        let mut constraint_nodes = MultiVecWithBounds::<_, 1>::new();

        let mut interactions = MultiVecWithBounds::<_, 1>::new();

        let mut interaction_messages = Vec::new();

        let mut unused_variables = MultiVecWithBounds::<_, 1>::new();

        let mut record_bounds = Vec::with_capacity(num_airs + 1);
        record_bounds.push(0);

        let mut total_rows = 0;

        for vk in &child_vk.inner.per_air {
            let constraints = &vk.symbolic_constraints.constraints;
            for node in &constraints.nodes {
                constraint_nodes.push(flatten_constraint_node(vk, node));
            }
            constraint_nodes.close_level(0);

            for interaction in &vk.symbolic_constraints.interactions {
                let message_start = interaction_messages.len();
                interaction_messages.extend(&interaction.message);
                interactions.push(FlatInteraction {
                    count: interaction.count as u32,
                    message_start: message_start as u32,
                    message_len: interaction.message.len() as u32,
                    bus_index: u32::from(interaction.bus_index),
                    count_weight: interaction.count_weight,
                });
            }
            interactions.close_level(0);

            for unused in &vk.unused_variables {
                unused_variables.push(flatten_unused_symbolic_variable(unused));
            }
            unused_variables.close_level(0);

            let interaction_message_rows: usize = vk
                .symbolic_constraints
                .interactions
                .iter()
                .map(|interaction| interaction.message.len())
                .sum();
            let rows_for_air = constraints.nodes.len()
                + vk.symbolic_constraints.interactions.len()
                + interaction_message_rows
                + vk.unused_variables.len();
            total_rows += rows_for_air;
            record_bounds.push(total_rows as u32);
        }

        let mut air_ids_per_record = vec![0; total_rows];
        for i in 0..(record_bounds.len() - 1) {
            air_ids_per_record[(record_bounds[i] as usize)..(record_bounds[i + 1] as usize)]
                .fill(i as u32);
        }

        let height = total_rows.max(1).next_power_of_two();
        let cached_width = CachedSymbolicExpressionColumns::<F>::width();
        let width = SingleMainSymbolicExpressionColumns::<F>::width() * max_num_proofs
            + if has_cached { 0 } else { cached_width + 1 };
        let trace = DeviceMatrix::with_capacity(height, width);

        let d_log_heights = proofs
            .iter()
            .flat_map(|proof| {
                proof
                    .cpu
                    .trace_vdata
                    .iter()
                    .map(|v| v.as_ref().map_or(0, |td| td.log_height))
            })
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();

        let mut sort_idx_by_air_idx = vec![0usize; num_airs * proofs.len()];
        for (chunk, preflight) in sort_idx_by_air_idx
            .chunks_exact_mut(num_airs)
            .zip(preflights.iter())
        {
            for (sort_idx, (air_idx, _)) in preflight
                .cpu
                .proof_shape
                .sorted_trace_vdata
                .iter()
                .enumerate()
            {
                chunk[*air_idx] = sort_idx;
            }
        }
        let d_sort_idx_by_air_idx = sort_idx_by_air_idx.to_device().unwrap();

        let d_expr_evals = expr_evals.data.to_device().unwrap();
        let d_ee_bounds_0 = expr_evals.bounds[0].to_device().unwrap();
        let d_ee_bounds_1 = expr_evals.bounds[1].to_device().unwrap();

        let d_constraint_nodes = constraint_nodes.data.to_device().unwrap();
        let d_constraint_nodes_bounds = constraint_nodes.bounds[0].to_device().unwrap();
        let d_interactions = to_device_or_nullptr(&interactions.data).unwrap();
        let d_interactions_bounds = interactions.bounds[0].to_device().unwrap();
        let d_interaction_messages = to_device_or_nullptr(&interaction_messages).unwrap();
        let d_unused_variables = to_device_or_nullptr(&unused_variables.data).unwrap();
        let d_unused_variables_bounds = unused_variables.bounds[0].to_device().unwrap();
        let d_record_bounds = record_bounds.to_device().unwrap();
        let d_air_ids_per_record = air_ids_per_record.to_device().unwrap();

        let mut sumcheck_data = Vec::new();
        let mut sumcheck_bounds = Vec::with_capacity(preflights.len() + 1);
        sumcheck_bounds.push(0);
        for preflight in preflights {
            sumcheck_data.extend_from_slice(&preflight.cpu.batch_constraint.sumcheck_rnd);
            sumcheck_bounds.push(sumcheck_data.len());
        }
        let d_sumcheck_rnds = if sumcheck_data.is_empty() {
            DeviceBuffer::new()
        } else {
            sumcheck_data.to_device().unwrap()
        };
        let d_sumcheck_bounds = sumcheck_bounds.to_device().unwrap();

        let d_cached_records = (!has_cached)
            .then(|| build_cached_gpu_records(child_vk).to_device())
            .transpose()
            .unwrap();

        unsafe {
            sym_expr_common_tracegen(
                trace.buffer(),
                height,
                child_vk.inner.params.l_skip,
                &d_log_heights,
                &d_sort_idx_by_air_idx,
                num_airs,
                proofs.len(),
                max_num_proofs,
                &d_expr_evals,
                &d_ee_bounds_0,
                &d_ee_bounds_1,
                &d_constraint_nodes,
                &d_constraint_nodes_bounds,
                &d_interactions,
                &d_interactions_bounds,
                &d_interaction_messages,
                &d_unused_variables,
                &d_unused_variables_bounds,
                &d_record_bounds,
                &d_air_ids_per_record,
                total_rows,
                &d_sumcheck_rnds,
                &d_sumcheck_bounds,
                d_cached_records.as_ref(),
            )
            .unwrap();
        }
        trace
    }
}
