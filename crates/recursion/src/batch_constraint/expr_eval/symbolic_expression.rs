use core::{array, cmp::min, iter::zip};
use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use itertools::Itertools;
use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    air_builders::{
        symbolic::{symbolic_variable::Entry, SymbolicExpressionNode},
        PartitionedAirBuilder,
    },
    interaction::InteractionBuilder,
    keygen::types::MultiStarkVerifyingKey,
    poly_common::{eval_eq_uni_at_one, Squarable},
    proof::Proof,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
    PrimeField32, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use stark_recursion_circuit_derive::AlignedBorrow;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

use crate::{
    batch_constraint::{
        bus::{
            ConstraintsFoldingBus, ConstraintsFoldingMessage, EqNegInternalBus, ExpressionClaimBus,
            InteractionsFoldingBus, InteractionsFoldingMessage, SymbolicExpressionBus,
            SymbolicExpressionMessage,
        },
        expr_eval::{
            dag_commit_cols_to_cached_cols, default_poseidon2_sub_chip, generate_dag_commit_info,
            DagCommitCols, DagCommitInfo, DagCommitPvs, DagCommitSubAir,
        },
    },
    bus::{
        AirShapeBus, AirShapeBusMessage, ColumnClaimsBus, ColumnClaimsMessage, HyperdimBus,
        HyperdimBusMessage, PublicValuesBus, PublicValuesBusMessage, SelHypercubeBus,
        SelHypercubeBusMessage, SelUniBus, SelUniBusMessage,
    },
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_subtract, scalar_subtract_ext_field, MultiVecWithBounds,
    },
};

pub(in crate::batch_constraint) const NUM_FLAGS: usize = 4;
pub(in crate::batch_constraint) const ENCODER_MAX_DEGREE: u32 = 2;
pub(in crate::batch_constraint) const FLAG_MODULUS: u32 = ENCODER_MAX_DEGREE + 1;

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct CachedSymbolicExpressionColumns<T> {
    pub(in crate::batch_constraint) flags: [T; NUM_FLAGS],

    pub(in crate::batch_constraint) air_idx: T,
    pub(in crate::batch_constraint) node_or_interaction_idx: T,
    /// Attributes that define this gate. For binary gates such as Add, Mul, etc.,
    /// this contains the node_idx's. For InteractionMsgComp, it gives (node_idx, idx_in_message).
    /// See [[NodeKind]].
    pub(in crate::batch_constraint) attrs: [T; 3],
    pub(in crate::batch_constraint) fanout: T,

    pub(in crate::batch_constraint) is_constraint: T,
    pub(in crate::batch_constraint) constraint_idx: T,
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

pub struct SymbolicExpressionAir<F: Field> {
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
    pub dag_commit_subair: Option<Arc<DagCommitSubAir<F>>>,
}

impl<F: Field> SymbolicExpressionAir<F> {
    fn has_cached(&self) -> bool {
        self.dag_commit_subair.is_none()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for SymbolicExpressionAir<F> {
    fn num_public_values(&self) -> usize {
        if self.has_cached() {
            0
        } else {
            DagCommitPvs::<F>::width()
        }
    }
}

impl<F: Field> PartitionedBaseAir<F> for SymbolicExpressionAir<F> {
    fn cached_main_widths(&self) -> Vec<usize> {
        if self.has_cached() {
            vec![CachedSymbolicExpressionColumns::<F>::width()]
        } else {
            vec![]
        }
    }

    fn common_main_width(&self) -> usize {
        SingleMainSymbolicExpressionColumns::<F>::width() * self.cnt_proofs
            + if self.has_cached() {
                0
            } else {
                DagCommitCols::<F>::width()
            }
    }
}

impl<F: Field> BaseAir<F> for SymbolicExpressionAir<F> {
    fn width(&self) -> usize {
        let single_main_width = SingleMainSymbolicExpressionColumns::<F>::width();
        if self.has_cached() {
            CachedSymbolicExpressionColumns::<F>::width() + single_main_width * self.cnt_proofs
        } else {
            DagCommitCols::<F>::width() + single_main_width * self.cnt_proofs
        }
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for SymbolicExpressionAir<AB::F>
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main_local = builder
            .common_main()
            .row_slice(0)
            .expect("window should have at least one row")
            .to_vec();

        let (cached_local_vec, main_slice) = if let Some(subair) = self.dag_commit_subair.as_ref() {
            // No cached trace: DagCommitCols come before the regular columns
            debug_assert!(!self.has_cached());
            let main_next = builder
                .common_main()
                .row_slice(1)
                .expect("window should have at least two rows")
                .to_vec();

            let commit_width = DagCommitCols::<AB::Var>::width();
            let (commit_local, rest_local) = main_local.as_slice().split_at(commit_width);
            let (commit_next, _rest_next) = main_next.as_slice().split_at(commit_width);
            subair.eval(builder, (commit_local, commit_next));

            let cached_local_vec = dag_commit_cols_to_cached_cols(commit_local).to_vec();
            (cached_local_vec, rest_local)
        } else {
            debug_assert!(self.has_cached());
            let cached_local_vec = builder.cached_mains()[0]
                .row_slice(0)
                .expect("window should have at least one row")
                .to_vec();
            let main_slice = main_local.as_slice();
            (cached_local_vec, main_slice)
        };

        let cached_cols: &CachedSymbolicExpressionColumns<AB::Var> =
            cached_local_vec.as_slice().borrow();
        let main_cols: Vec<&SingleMainSymbolicExpressionColumns<AB::Var>> = main_slice
            .chunks(SingleMainSymbolicExpressionColumns::<AB::Var>::width())
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
            let proof_idx = AB::F::from_usize(proof_idx);

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

            self.expr_bus.add_key_with_lookups(
                builder,
                proof_idx,
                SymbolicExpressionMessage {
                    air_idx: cached_cols.air_idx.into(),
                    node_idx: cached_cols.node_or_interaction_idx.into(),
                    value: value.clone(),
                },
                cols.is_present * cached_cols.fanout,
            );
            self.expr_bus.lookup_key(
                builder,
                proof_idx,
                SymbolicExpressionMessage {
                    air_idx: cached_cols.air_idx,
                    node_idx: cached_cols.attrs[0],
                    value: arg_ef0,
                },
                cols.is_present * is_arg0_node_idx.clone(),
            );
            self.expr_bus.lookup_key(
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
            self.air_shape_bus.lookup_key(
                builder,
                proof_idx,
                AirShapeBusMessage {
                    sort_idx: cols.sort_idx.into(),
                    property_idx: AB::Expr::ZERO,
                    value: cached_cols.air_idx.into(),
                },
                cols.is_present,
            );
            self.hyperdim_bus.lookup_key(
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
                self.sel_uni_bus.lookup_key(
                    builder,
                    proof_idx,
                    SelUniBusMessage {
                        n: AB::Expr::NEG_ONE * cols.n_abs * cols.is_n_neg,
                        is_first: is_first.clone(),
                        value: arg_ef0.map(Into::into),
                    },
                    cols.is_present * is_sel.clone(),
                );
                self.sel_hypercube_bus.lookup_key(
                    builder,
                    proof_idx,
                    SelHypercubeBusMessage {
                        n: cols.n_abs.into(),
                        is_first: is_first.clone(),
                        value: arg_ef1.map(Into::into),
                    },
                    // OK: cols.is_n_neg => cols.is_present
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

pub struct SymbolicExpressionTraceGenerator {
    pub max_num_proofs: usize,
    pub has_cached: bool,
}

pub(crate) struct SymbolicExpressionCtx<'a> {
    pub vk: &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    pub proofs: &'a [&'a Proof<BabyBearPoseidon2Config>],
    pub preflights: &'a [&'a Preflight],
    pub expr_evals: &'a MultiVecWithBounds<EF, 2>,
    pub cached_trace_record: &'a Option<&'a CachedTraceRecord>,
}

impl RowMajorChip<F> for SymbolicExpressionTraceGenerator {
    type Ctx<'a> = SymbolicExpressionCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let child_vk = ctx.vk;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        let max_num_proofs = self.max_num_proofs;
        let has_cached = self.has_cached;
        let expr_evals = ctx.expr_evals;
        let trace_height = required_height;
        let l_skip = child_vk.inner.params.l_skip;

        let single_main_width = SingleMainSymbolicExpressionColumns::<F>::width();
        let dag_commit_width = DagCommitCols::<F>::width();
        let main_width =
            single_main_width * max_num_proofs + if has_cached { 0 } else { dag_commit_width };

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
                                record.args[..D_EF].copy_from_slice(
                                    expr_evals[node_idx].as_basis_coefficients_slice(),
                                );
                            }
                            Entry::Main { .. } => {
                                record.args[..D_EF].copy_from_slice(
                                    expr_evals[node_idx].as_basis_coefficients_slice(),
                                );
                            }
                            Entry::Permutation { .. } => unreachable!(),
                            Entry::Public => record.args[..D_EF].copy_from_slice(
                                expr_evals[node_idx].as_basis_coefficients_slice(),
                            ),
                            Entry::Challenge => unreachable!(),
                            Entry::Exposed => unreachable!(),
                        },
                        SymbolicExpressionNode::IsFirstRow => {
                            record.args[..D_EF].copy_from_slice(
                                is_first_uni_by_log_height[min(log_height, l_skip)]
                                    .as_basis_coefficients_slice(),
                            );
                            record.args[D_EF..2 * D_EF].copy_from_slice(
                                is_first_mle_by_n[log_height.saturating_sub(l_skip)]
                                    .as_basis_coefficients_slice(),
                            );
                        }
                        SymbolicExpressionNode::IsLastRow
                        | SymbolicExpressionNode::IsTransition => {
                            record.args[..D_EF].copy_from_slice(
                                is_last_uni_by_log_height[min(log_height, l_skip)]
                                    .as_basis_coefficients_slice(),
                            );
                            record.args[D_EF..2 * D_EF].copy_from_slice(
                                is_last_mle_by_n[log_height.saturating_sub(l_skip)]
                                    .as_basis_coefficients_slice(),
                            );
                        }
                        SymbolicExpressionNode::Constant(_) => {}
                        SymbolicExpressionNode::Add {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            record.args[..D_EF].copy_from_slice(
                                expr_evals[*left_idx].as_basis_coefficients_slice(),
                            );
                            record.args[D_EF..2 * D_EF].copy_from_slice(
                                expr_evals[*right_idx].as_basis_coefficients_slice(),
                            );
                        }
                        SymbolicExpressionNode::Sub {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            record.args[..D_EF].copy_from_slice(
                                expr_evals[*left_idx].as_basis_coefficients_slice(),
                            );
                            record.args[D_EF..2 * D_EF].copy_from_slice(
                                expr_evals[*right_idx].as_basis_coefficients_slice(),
                            );
                        }
                        SymbolicExpressionNode::Neg {
                            idx,
                            degree_multiple: _,
                        } => {
                            record.args[..D_EF]
                                .copy_from_slice(expr_evals[*idx].as_basis_coefficients_slice());
                        }
                        SymbolicExpressionNode::Mul {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            record.args[..D_EF].copy_from_slice(
                                expr_evals[*left_idx].as_basis_coefficients_slice(),
                            );
                            record.args[D_EF..2 * D_EF].copy_from_slice(
                                expr_evals[*right_idx].as_basis_coefficients_slice(),
                            );
                        }
                    };
                    records.push(Some(record));
                }
                for interaction in &vk.symbolic_constraints.interactions {
                    let mut args = [F::ZERO; 2 * D_EF];
                    args[..D_EF].copy_from_slice(
                        expr_evals[interaction.count].as_basis_coefficients_slice(),
                    );
                    records.push(Some(Record {
                        args,
                        sort_idx,
                        n_abs,
                        is_n_neg,
                    }));

                    for &node_idx in &interaction.message {
                        let mut args = [F::ZERO; 2 * D_EF];
                        args[..D_EF]
                            .copy_from_slice(expr_evals[node_idx].as_basis_coefficients_slice());
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
                            args[..D_EF].copy_from_slice(
                                expr_evals[node_idx].as_basis_coefficients_slice(),
                            );
                            records.push(Some(Record {
                                args,
                                sort_idx,
                                n_abs,
                                is_n_neg,
                            }));
                        }
                        Entry::Main { .. } => {
                            let mut args = [F::ZERO; 2 * D_EF];
                            args[..D_EF].copy_from_slice(
                                expr_evals[node_idx].as_basis_coefficients_slice(),
                            );
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
        let height = if let Some(height) = trace_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let mut main_trace = F::zero_vec(main_width * height);

        let (encoder, cached_records, poseidon2_rows) = if has_cached {
            (None, None, None)
        } else {
            let encoder = Encoder::new(NodeKind::COUNT, ENCODER_MAX_DEGREE, true);
            assert_eq!(encoder.width(), NUM_FLAGS);

            let cached_trace_record = ctx.cached_trace_record.unwrap();
            debug_assert_eq!(cached_trace_record.records.len(), num_valid_rows);

            let poseidon2_subchip = default_poseidon2_sub_chip();
            let poseidon2_trace = poseidon2_subchip.generate_trace(
                cached_trace_record
                    .dag_commit_info
                    .as_ref()
                    .unwrap()
                    .poseidon2_inputs
                    .clone(),
            );
            let poseidon2_rows = poseidon2_trace
                .rows()
                .map(|row| row.collect_vec())
                .collect_vec();

            (
                Some(encoder),
                Some(&cached_trace_record.records),
                Some(poseidon2_rows),
            )
        };

        main_trace
            .par_chunks_exact_mut(main_width)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let main_offset = if has_cached {
                    0
                } else {
                    // Poseidon2 data must be written for ALL rows (including padding) to
                    // keep the onion hash chain valid.
                    let poseidon2_row = &poseidon2_rows.as_ref().unwrap()[row_idx];
                    row[..poseidon2_row.len()].copy_from_slice(poseidon2_row);

                    if row_idx < num_valid_rows {
                        let record = &cached_records.as_ref().unwrap()[row_idx];
                        let encoder = encoder.as_ref().unwrap();
                        let cols: &mut DagCommitCols<_> = row[..dag_commit_width].borrow_mut();
                        for (i, x) in encoder
                            .get_flag_pt(record.kind as usize)
                            .into_iter()
                            .enumerate()
                        {
                            cols.flags[i] = F::from_u32(x);
                        }
                        cols.is_constraint = F::from_bool(record.is_constraint);
                    }

                    dag_commit_width
                };

                if row_idx >= num_valid_rows {
                    return;
                }

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
                    cols.sort_idx = F::from_usize(record.sort_idx);
                    cols.n_abs = F::from_usize(record.n_abs);
                    cols.is_n_neg = F::from_usize(record.is_n_neg);
                }
            });

        Some(RowMajorMatrix::new(main_trace, main_width))
    }
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
    pub(crate) kind: NodeKind,
    pub(crate) air_idx: usize,
    pub(crate) node_idx: usize,
    pub(crate) attrs: [usize; 3],
    pub(crate) is_constraint: bool,
    pub(crate) constraint_idx: usize,
    pub(crate) fanout: usize,
}

#[derive(Debug, Clone)]
pub struct CachedTraceRecord {
    pub(crate) records: Vec<CachedRecord>,
    pub dag_commit_info: Option<DagCommitInfo<F>>,
}

pub(crate) fn build_cached_trace_record(
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    has_cached: bool,
) -> CachedTraceRecord {
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

    let dag_commit_info = (!has_cached).then(|| {
        let encoder = Encoder::new(NodeKind::COUNT, ENCODER_MAX_DEGREE, true);
        generate_dag_commit_info(&records, encoder)
    });

    CachedTraceRecord {
        records,
        dag_commit_info,
    }
}

/// Returns the cached trace
#[tracing::instrument(
    name = "generate_cached_trace",
    skip_all,
    fields(air = "SymbolicExpressionAir")
)]
pub(crate) fn generate_symbolic_expr_cached_trace(
    cached_trace_record: &CachedTraceRecord,
) -> RowMajorMatrix<F> {
    // 3 var types: main, preprocessed, public value
    // 3 selectors: is_first, is_last, is_transition
    // 1 constant type
    // 4 gates: add, sub, neg, mul
    let encoder = Encoder::new(NodeKind::COUNT, ENCODER_MAX_DEGREE, true);
    assert_eq!(encoder.width(), NUM_FLAGS);

    let cached_width = CachedSymbolicExpressionColumns::<F>::width();
    let records = &cached_trace_record.records;

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
                cols.flags[i] = F::from_u32(x);
                cols.air_idx = F::from_usize(record.air_idx);
                cols.node_or_interaction_idx = F::from_usize(record.node_idx);
                cols.attrs = record.attrs.map(F::from_usize);
                cols.is_constraint = F::from_bool(record.is_constraint);
                cols.constraint_idx = F::from_usize(record.constraint_idx);
                cols.fanout = F::from_usize(record.fanout);
            }
        });

    RowMajorMatrix::new(cached_trace, cached_width)
}

#[cfg(feature = "cuda")]
pub(in crate::batch_constraint) mod cuda {

    use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        batch_constraint::{cuda_abi::sym_expr_common_tracegen, cuda_utils::*},
        cuda::{preflight::PreflightGpu, proof::ProofGpu, to_device_or_nullptr},
        tracegen::ModuleChip,
    };

    pub struct SymbolicExpressionGpuCtx<'a> {
        pub vk: &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        pub proofs: &'a [ProofGpu],
        pub preflights: &'a [PreflightGpu],
        pub expr_evals: &'a MultiVecWithBounds<openvm_cuda_backend::prelude::EF, 2>,
        pub cached_trace_record: &'a Option<&'a CachedTraceRecord>,
    }

    impl ModuleChip<GpuBackend> for SymbolicExpressionTraceGenerator {
        type Ctx<'a> = SymbolicExpressionGpuCtx<'a>;

        #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            let child_vk = ctx.vk;
            let proofs = ctx.proofs;
            let preflights = ctx.preflights;
            let max_num_proofs = self.max_num_proofs;
            let has_cached = self.has_cached;
            let expr_evals = ctx.expr_evals;

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

            let height = if let Some(height) = required_height {
                if height < total_rows {
                    return None;
                }
                height
            } else {
                total_rows.max(1).next_power_of_two()
            };
            let commit_width = DagCommitCols::<F>::width();
            let width = SingleMainSymbolicExpressionColumns::<F>::width() * max_num_proofs
                + if has_cached { 0 } else { commit_width };
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

            let d_cached_records = ctx
                .cached_trace_record
                .map(|data| build_cached_gpu_records(data).unwrap().to_device().unwrap());

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

            Some(AirProvingContext::simple_no_pis(trace))
        }
    }
}
