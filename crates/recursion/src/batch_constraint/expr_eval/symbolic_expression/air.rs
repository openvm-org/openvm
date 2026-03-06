use core::array;
use std::{borrow::Borrow, sync::Arc};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::EnumIter;

use crate::{
    batch_constraint::{
        bus::{
            ConstraintsFoldingBus, ConstraintsFoldingMessage, InteractionsFoldingBus,
            InteractionsFoldingMessage, SymbolicExpressionBus, SymbolicExpressionMessage,
        },
        expr_eval::{dag_commit_cols_to_cached_cols, DagCommitCols, DagCommitPvs, DagCommitSubAir},
    },
    bus::{
        AirShapeBus, AirShapeBusMessage, AirShapeProperty, ColumnClaimsBus, ColumnClaimsMessage,
        HyperdimBus, HyperdimBusMessage, PublicValuesBus, PublicValuesBusMessage, SelHypercubeBus,
        SelHypercubeBusMessage, SelUniBus, SelUniBusMessage,
    },
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_subtract, scalar_subtract_ext_field,
    },
};

pub(in crate::batch_constraint) const NUM_FLAGS: usize = 4;
pub(in crate::batch_constraint) const ENCODER_MAX_DEGREE: u32 = 2;
pub(in crate::batch_constraint) const FLAG_MODULUS: u32 = ENCODER_MAX_DEGREE + 1;

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
    // Args: (node_idx,)
    InteractionBusIndex = 13,
}

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
    pub(in crate::batch_constraint) is_present: T,
    // Dynamic arguments. For Add/Mul/Sub, this splits into two extension-field elements.
    // For selectors:
    //   args[0..D_EF)   = sel_uni witness (base or rotated depending on selector type).
    //   args[D_EF..2*D_EF) = eq-prefix witness (prod r_i or prod (1-r_i)).
    pub(in crate::batch_constraint) args: [T; 2 * D_EF],
    pub(in crate::batch_constraint) sort_idx: T,
    pub(in crate::batch_constraint) n_abs: T,
    pub(in crate::batch_constraint) is_n_neg: T,
}

pub struct SymbolicExpressionAir<F: Field> {
    pub expr_bus: SymbolicExpressionBus,
    pub hyperdim_bus: HyperdimBus,
    pub air_shape_bus: AirShapeBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub interactions_folding_bus: InteractionsFoldingBus,
    pub constraints_folding_bus: ConstraintsFoldingBus,
    pub public_values_bus: PublicValuesBus,
    pub sel_hypercube_bus: SelHypercubeBus,
    pub sel_uni_bus: SelUniBus,

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
            builder
                .when(cols.is_present)
                .assert_one(enc.is_valid::<AB>(&flags));

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
                    NodeKind::InteractionBusIndex => {
                        base_to_ext(cached_cols.attrs[0] + AB::Expr::ONE)
                    }
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
                    property_idx: AirShapeProperty::AirId.to_field(),
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
            let is_bus_index =
                enc.get_flag_expr::<AB>(NodeKind::InteractionBusIndex as usize, &flags);
            // is_interaction doesn't include bus index
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
            self.interactions_folding_bus.send(
                builder,
                proof_idx,
                InteractionsFoldingMessage {
                    air_idx: cached_cols.air_idx.into(),
                    interaction_idx: cached_cols.node_or_interaction_idx.into(),
                    is_mult: AB::Expr::ZERO,
                    idx_in_message: AB::Expr::NEG_ONE,
                    value: value.clone(),
                },
                is_bus_index * cols.is_present,
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
