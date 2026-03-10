use std::{array::from_fn, borrow::Borrow};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, PreHashBus, PreHashMessage, PublicValuesBus,
        PublicValuesBusMessage,
    },
    utils::assert_zeros,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{
    VerifierBasePvs, VerifierDefPvs, CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID,
};

use crate::circuit::{
    inner::bus::{PvsAirConsistencyBus, PvsAirConsistencyMessage},
    root::NUM_DIGESTS_IN_VK_COMMIT,
    subair::{HashSliceCtx, HashSliceSubAir},
    utils::{assert_dag_commit_eq, assert_dag_commit_unset, vk_commit_components},
    CONSTRAINT_EVAL_CACHED_INDEX,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierPvsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub has_verifier_pvs: F,
    pub child_pvs: VerifierBasePvs<F>,
}

pub struct VerifierPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub pre_hash_bus: PreHashBus,
    pub pvs_air_consistency_bus: PvsAirConsistencyBus,
    pub deferral_config: VerifierDeferralConfig,
}

impl<F> BaseAir<F> for VerifierPvsAir {
    fn width(&self) -> usize {
        VerifierPvsCols::<u8>::width() + self.deferral_config.width()
    }
}
impl<F> BaseAirWithPublicValues<F> for VerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width() + self.deferral_config.num_public_values()
    }
}
impl<F> PartitionedBaseAir<F> for VerifierPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for VerifierPvsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let base_cols_width = VerifierPvsCols::<AB::Var>::width();
        let (base_local, def_local) = local.split_at(base_cols_width);
        let (base_next, def_next) = next.split_at(base_cols_width);

        let local: &VerifierPvsCols<AB::Var> = (*base_local).borrow();
        let next: &VerifierPvsCols<AB::Var> = (*base_next).borrow();

        /*
         * This AIR can optionally handle deferrals, the constraints for which are defined in
         * function eval_deferrals. We expect dag_commit_cond to be a boolean value that is
         * true iff local and next's app, leaf, and internal-for-leaf DAG commits should be
         * constrained for equality.
         */
        let (dag_commit_cond, deferral_flag, consistency_mult) = match &self.deferral_config {
            VerifierDeferralConfig::Enabled {
                hash_slice_subair: subair,
            } => {
                let def_local: &VerifierDeferralCols<AB::Var> = (*def_local).borrow();
                let def_next: &VerifierDeferralCols<AB::Var> = (*def_next).borrow();
                self.eval_deferrals(builder, local, next, def_local, def_next, subair)
            }
            VerifierDeferralConfig::Disabled => {
                debug_assert_eq!(def_local.len(), 0);
                (
                    and(local.is_valid, next.is_valid),
                    AB::Expr::ZERO,
                    AB::Expr::ONE,
                )
            }
        };

        /*
         * Constrain basic features about the non-pvs columns.
         */
        builder.assert_bool(local.is_valid);
        builder.when_first_row().assert_one(local.is_valid);
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);

        builder.when_first_row().assert_zero(local.proof_idx);
        builder
            .when_transition()
            .when(and(local.is_valid, next.is_valid))
            .assert_eq(local.proof_idx + AB::F::ONE, next.proof_idx);

        builder.assert_bool(local.has_verifier_pvs);
        builder
            .when(local.has_verifier_pvs)
            .assert_one(local.is_valid);

        /*
         * We constrain the consistency of verifier-specific public values. We can determine
         * what layer a verifier is at using the has_verifier_pvs and internal_flag columns.
         * There are several cases we cover:
         * - has_verifier_pvs == 0: leaf verifier, app (or deferral circuit) children
         * - has_verifier_pvs == 1 && internal_flag == 0: internal verifier with leaf children
         * - has_verifier_pvs == 1 && internal_flag == 1: internal_for_leaf children
         * - has_verifier_pvs == 1 && internal_flag == 2: internal_recursive children
         *   - recursion_flag == 1: 2nd (i.e. index 1) internal_recursive layer
         *   - recursion_flag == 1: 3rd internal_recursive layer or beyond
         */
        // constrain the verifier pvs flags and internal_recursive_dag_tommit are the same
        // across all valid rows
        let both_valid = and(local.is_valid, next.is_valid);
        let mut when_both_valid = builder.when(both_valid.clone());

        when_both_valid.assert_eq(local.has_verifier_pvs, next.has_verifier_pvs);
        when_both_valid.assert_eq(local.child_pvs.internal_flag, next.child_pvs.internal_flag);
        when_both_valid.assert_eq(
            local.child_pvs.recursion_flag,
            next.child_pvs.recursion_flag,
        );

        assert_dag_commit_eq(
            &mut when_both_valid,
            local.child_pvs.internal_recursive_dag_commit,
            next.child_pvs.internal_recursive_dag_commit,
        );

        // constrain the other commits are the same when needed
        let mut when_dag_compare = builder.when(dag_commit_cond);

        assert_dag_commit_eq(
            &mut when_dag_compare,
            local.child_pvs.app_dag_commit,
            next.child_pvs.app_dag_commit,
        );
        assert_dag_commit_eq(
            &mut when_dag_compare,
            local.child_pvs.leaf_dag_commit,
            next.child_pvs.leaf_dag_commit,
        );
        assert_dag_commit_eq(
            &mut when_dag_compare,
            local.child_pvs.internal_for_leaf_dag_commit,
            next.child_pvs.internal_for_leaf_dag_commit,
        );

        // constrain that the flags are ternary
        builder.assert_tern(local.child_pvs.internal_flag);
        builder.assert_tern(local.child_pvs.recursion_flag);

        // constrain that internal_flag is 2 when recursion_flag is set, and not 2 otherwise
        builder
            .when(local.child_pvs.recursion_flag)
            .assert_eq(local.child_pvs.internal_flag, AB::F::TWO);
        builder
            .when(
                (local.child_pvs.recursion_flag - AB::F::ONE)
                    * (local.child_pvs.recursion_flag - AB::F::TWO),
            )
            .assert_bool(local.child_pvs.internal_flag);

        // constrain that child commits are 0 when they shouldn't be defined
        let is_leaf = not(local.has_verifier_pvs);
        let is_internal = local.has_verifier_pvs;

        builder
            .when(is_leaf.clone())
            .assert_zero(local.child_pvs.internal_flag);

        assert_dag_commit_unset(
            &mut builder.when(is_leaf.clone()),
            local.child_pvs.app_dag_commit,
        );
        assert_dag_commit_unset(
            &mut builder.when(
                (local.child_pvs.internal_flag - AB::F::ONE)
                    * (local.child_pvs.internal_flag - AB::F::TWO),
            ),
            local.child_pvs.leaf_dag_commit,
        );
        assert_dag_commit_unset(
            &mut builder.when(local.child_pvs.internal_flag - AB::F::TWO),
            local.child_pvs.internal_for_leaf_dag_commit,
        );
        assert_dag_commit_unset(
            &mut builder.when(local.child_pvs.recursion_flag - AB::F::TWO),
            local.child_pvs.internal_recursive_dag_commit,
        );

        /*
         * We need to receive public values from ProofShapeModule to ensure the values being read
         * here are correct. This AIR will only read values if it's internal.
         */
        let verifier_pvs_id = AB::Expr::from_usize(VERIFIER_PVS_AIR_ID);

        for (pv_idx, value) in local.child_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                local.is_valid * is_internal,
            );
        }

        /*
         * We also need to receive cached commits from ProofShapeModule. Note that the
         * app/deferral circuit cached commits are received in another AIR, so only the
         * internal verifier will receive them here.
         */
        let half = AB::F::TWO.inverse();
        let is_internal_flag_zero = (local.child_pvs.internal_flag - AB::F::ONE)
            * (local.child_pvs.internal_flag - AB::F::TWO)
            * half;
        let is_internal_flag_one =
            (AB::Expr::TWO - local.child_pvs.internal_flag) * local.child_pvs.internal_flag;
        let is_recursion_flag_one =
            (AB::Expr::TWO - local.child_pvs.recursion_flag) * local.child_pvs.recursion_flag;
        let is_recursion_flag_two =
            (local.child_pvs.recursion_flag - AB::F::ONE) * local.child_pvs.recursion_flag * half;
        let cached_commit = from_fn(|i| {
            is_internal_flag_zero.clone() * local.child_pvs.app_dag_commit.cached_commit[i]
                + is_internal_flag_one.clone() * local.child_pvs.leaf_dag_commit.cached_commit[i]
                + is_recursion_flag_one.clone()
                    * local.child_pvs.internal_for_leaf_dag_commit.cached_commit[i]
                + is_recursion_flag_two.clone()
                    * local.child_pvs.internal_recursive_dag_commit.cached_commit[i]
        });

        self.cached_commit_bus.receive(
            builder,
            local.proof_idx,
            CachedCommitBusMessage {
                air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                cached_commit,
            },
            local.is_valid * is_internal,
        );

        /*
         * We provide proof metadata for lookup here to ensure consistency between AIRs that
         * process public values.
         */
        self.pvs_air_consistency_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            PvsAirConsistencyMessage {
                deferral_flag,
                has_verifier_pvs: local.has_verifier_pvs.into(),
            },
            local.is_valid * consistency_mult,
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. Note that we only impose constraints for layers below the current
         * one - it is impossible for the current layer to know its own commit, and future layers
         * will catch if we preemptively define a current or future verifier commit.
         */
        let base_pvs_width = VerifierBasePvs::<AB::Var>::width();
        let &VerifierBasePvs::<_> {
            internal_flag,
            app_dag_commit,
            leaf_dag_commit,
            internal_for_leaf_dag_commit,
            recursion_flag,
            internal_recursive_dag_commit,
        } = builder.public_values()[0..base_pvs_width].borrow();

        // constrain internal_flag is 0 at the leaf level
        builder
            .when(and(local.is_valid, is_leaf.clone()))
            .assert_zero(internal_flag);

        // constrain recursion_flag is 0 at the leaf and internal_for_leaf levels
        builder
            .when(
                local.is_valid
                    * (local.child_pvs.internal_flag - AB::F::ONE)
                    * (local.child_pvs.internal_flag - AB::F::TWO),
            )
            .assert_zero(recursion_flag);

        // constraint internal_flag is incremented properly at internal levels
        builder
            .when(is_internal)
            .when_ne(local.child_pvs.internal_flag, AB::F::TWO)
            .assert_eq(internal_flag, local.child_pvs.internal_flag + AB::F::ONE);

        // constrain app_dag_commit is set at all internal levels and matches the first row
        assert_dag_commit_eq(
            &mut builder.when_first_row().when(is_internal),
            local.child_pvs.app_dag_commit,
            app_dag_commit,
        );

        // constrain verifier-specific pvs at all internal_recursive levels
        builder
            .when(local.child_pvs.internal_flag)
            .assert_zero(internal_flag.into() - AB::F::TWO);
        assert_dag_commit_eq(
            &mut builder.when_first_row().when(local.child_pvs.internal_flag),
            local.child_pvs.leaf_dag_commit,
            leaf_dag_commit,
        );

        // constrain recursion_flag is 1 at the first internal_recursive level
        builder
            .when(local.child_pvs.internal_flag * (local.child_pvs.internal_flag - AB::F::TWO))
            .assert_one(recursion_flag);

        // constrain verifier-specific pvs at internal_recursive levels after the first
        builder
            .when(local.child_pvs.recursion_flag)
            .assert_eq(recursion_flag, AB::F::TWO);
        assert_dag_commit_eq(
            &mut builder
                .when_first_row()
                .when(local.child_pvs.recursion_flag),
            local.child_pvs.internal_for_leaf_dag_commit,
            internal_for_leaf_dag_commit,
        );

        // constrain verifier-specific pvs at internal_recursive levels after the second
        assert_dag_commit_eq(
            &mut builder.when(
                local.child_pvs.recursion_flag * (local.child_pvs.recursion_flag - AB::F::ONE),
            ),
            local.child_pvs.internal_recursive_dag_commit,
            internal_recursive_dag_commit,
        );

        /*
         * Unlike the cached commits, we receive the pre-hash on the same layer that it's
         * observed. We receive it from ProofShapeModule also.
         */
        let internal_flag = internal_flag.into();
        let recursion_flag = recursion_flag.into();

        let is_internal_flag_zero = (internal_flag.clone() - AB::Expr::ONE)
            * (internal_flag.clone() - AB::Expr::TWO)
            * half;
        let is_internal_flag_one = (AB::Expr::TWO - internal_flag.clone()) * internal_flag.clone();
        let is_recursion_flag_one =
            (AB::Expr::TWO - recursion_flag.clone()) * recursion_flag.clone();
        let is_recursion_flag_two =
            (recursion_flag.clone() - AB::Expr::ONE) * recursion_flag.clone() * half;

        let vk_pre_hash = from_fn(|i| {
            is_internal_flag_zero.clone() * app_dag_commit.vk_pre_hash[i].into()
                + is_internal_flag_one.clone() * leaf_dag_commit.vk_pre_hash[i].into()
                + is_recursion_flag_one.clone() * internal_for_leaf_dag_commit.vk_pre_hash[i].into()
                + is_recursion_flag_two.clone()
                    * internal_recursive_dag_commit.vk_pre_hash[i].into()
        });

        self.pre_hash_bus.receive(
            builder,
            local.proof_idx,
            PreHashMessage { vk_pre_hash },
            local.is_valid,
        );
    }
}

///////////////////////////////////////////////////////////////////////////////
// DEFERRAL SUPPORT
///////////////////////////////////////////////////////////////////////////////

pub enum VerifierDeferralConfig {
    Enabled { hash_slice_subair: HashSliceSubAir },
    Disabled,
}

impl VerifierDeferralConfig {
    pub fn width(&self) -> usize {
        match self {
            VerifierDeferralConfig::Enabled { .. } => VerifierDeferralCols::<u8>::width(),
            VerifierDeferralConfig::Disabled => 0,
        }
    }

    pub fn num_public_values(&self) -> usize {
        match self {
            VerifierDeferralConfig::Enabled { .. } => VerifierDefPvs::<u8>::width(),
            VerifierDeferralConfig::Disabled => 0,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierDeferralCols<F> {
    pub is_last: F,
    pub intermediate_states: [[F; POSEIDON2_WIDTH]; NUM_DIGESTS_IN_VK_COMMIT - 1],
    pub child_pvs: VerifierDefPvs<F>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierCombinedPvs<F> {
    pub base: VerifierBasePvs<F>,
    pub def: VerifierDefPvs<F>,
}

impl VerifierPvsAir {
    fn eval_deferrals<AB>(
        &self,
        builder: &mut AB,
        base_local: &VerifierPvsCols<AB::Var>,
        base_next: &VerifierPvsCols<AB::Var>,
        def_local: &VerifierDeferralCols<AB::Var>,
        def_next: &VerifierDeferralCols<AB::Var>,
        hash_slice_subair: &HashSliceSubAir,
    ) -> (AB::Expr, AB::Expr, AB::Expr)
    where
        AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues,
    {
        /*
         * The deferral_flag should be 0 if a proof has only VM public values defined, 1 if
         * only deferral public values, and 2 if both. There are 4 valid cases:
         * - All valid rows have deferral_flag == 0
         * - All valid rows have deferral_flag == 1
         * - There are exactly two rows with deferral_flag == row_idx
         * - There is exactly one row with deferral_flag == 2
         */
        let delta = def_next.child_pvs.deferral_flag - def_local.child_pvs.deferral_flag;
        builder.assert_tern(def_local.child_pvs.deferral_flag);

        // constrain that is_last is correctly set on the last valid row
        builder.assert_bool(def_local.is_last);
        builder
            .when(def_local.is_last)
            .assert_one(base_local.is_valid);
        builder
            .when(and(base_local.is_valid, not(def_local.is_last)))
            .assert_one(base_next.is_valid);
        builder
            .when(def_local.is_last)
            .assert_zero(base_next.is_valid * base_next.proof_idx);
        builder
            .when_last_row()
            .when(base_local.is_valid)
            .assert_one(def_local.is_last);

        // constrain that delta is 0 or 1
        builder.when_transition().assert_bool(delta.clone());

        // constrain that if deferral_flag is 1 or 2, it cannot change later (note that if
        // deferral_flag is 1 or 2 there may only be 1 or 2 rows)
        builder
            .when_transition()
            .when(def_local.child_pvs.deferral_flag)
            .assert_zero(delta.clone());

        // constrain that the 0->1 transition happens only on the first row
        builder
            .when_transition()
            .when(base_local.proof_idx)
            .assert_zero(delta.clone());

        // constrain that if first row is 2, it must be the only valid row
        builder
            .when(def_local.child_pvs.deferral_flag)
            .when_ne(def_local.child_pvs.deferral_flag, AB::F::ONE)
            .assert_one(def_local.is_last);

        // constrain row 1 to be the last on the 0->1 transition
        builder
            .when_transition()
            .when(delta.clone())
            .assert_one(def_next.is_last);

        /*
         * We also need to constrain the deferral-related public values. In particular, the
         * def_hook_vk_commit should be defined exactly when internal_for_leaf_dag_commit
         * is for deferral_flag == 1.
         */
        // constrain that delta == 1 only at some internal_recursive layer
        builder
            .when(delta.clone())
            .assert_eq(base_local.child_pvs.internal_flag, AB::F::TWO);
        builder
            .when(def_local.child_pvs.deferral_flag)
            .when_ne(def_local.child_pvs.deferral_flag, AB::F::ONE)
            .assert_eq(base_local.child_pvs.internal_flag, AB::F::TWO);

        // constrain that def_hook_vk_commit is unset when internal_for_leaf_dag_commit is
        assert_zeros(
            &mut builder.when(base_local.child_pvs.internal_flag - AB::F::TWO),
            def_local.child_pvs.def_hook_vk_commit,
        );

        /*
         * We need to receive dedeferral-specific public values from ProofShapeModule to
         * ensure the values being read are correct.
         */
        let verifier_pvs_id = AB::Expr::from_usize(VERIFIER_PVS_AIR_ID);
        let pvs_offset = VerifierBasePvs::<AB::Var>::width();

        for (pv_idx, value) in def_local.child_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                base_local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx + pvs_offset),
                    value: (*value).into(),
                },
                base_local.is_valid * base_local.has_verifier_pvs,
            );
        }

        /*
         * Finally, we need to constrain that the deferral-specific public values this AIR
         * produces are consistent with the child's.
         */
        let &VerifierCombinedPvs::<_> {
            base: base_pvs,
            def: def_pvs,
        } = builder.public_values().borrow();

        let base_pvs_ref: &VerifierBasePvs<_> = base_pvs.as_slice().borrow();
        let &VerifierBasePvs::<_> { internal_flag, .. } = base_pvs_ref;

        let &VerifierDefPvs::<_> {
            deferral_flag,
            def_hook_vk_commit,
        } = def_pvs.as_slice().borrow();

        // constrain deferral_flag either matches each row, or is 2 when delta is non-zero
        builder
            .when(delta.clone())
            .assert_eq(deferral_flag, AB::F::TWO);
        builder
            .when_ne(delta.clone(), AB::F::ONE)
            .when_ne(delta.clone(), -AB::F::ONE)
            .assert_eq(deferral_flag, def_local.child_pvs.deferral_flag);

        // constrain def_hook_vk_commit matches if set in child_pvs
        assert_array_eq(
            &mut builder
                .when(base_local.child_pvs.recursion_flag)
                .when(def_local.child_pvs.deferral_flag),
            def_local.child_pvs.def_hook_vk_commit,
            def_hook_vk_commit,
        );

        // constrain the child def_hook_vk_commit is defined when internal_flag is 2 and
        // deferral_flag is non-zero
        let is_child_def_hook_vk_defined = base_local.child_pvs.internal_flag
            * (base_local.child_pvs.internal_flag - AB::Expr::ONE)
            * def_local.child_pvs.deferral_flag;

        assert_array_eq(
            &mut builder.when(is_child_def_hook_vk_defined),
            def_local.child_pvs.def_hook_vk_commit,
            def_hook_vk_commit,
        );

        // constrain def_hook_vk_commit's value when internal_dag is 2 and deferral_flag
        // is 1
        let compute_def_hook_commit = internal_flag.into()
            * (internal_flag.into() - AB::Expr::ONE)
            * deferral_flag.into()
            * (AB::Expr::TWO - deferral_flag.into())
            * AB::F::TWO.inverse();

        hash_slice_subair.eval(
            builder,
            HashSliceCtx {
                elements: &vk_commit_components(base_pvs_ref)
                    .into_iter()
                    .map(|c| c.map(Into::into))
                    .collect::<Vec<_>>(),
                intermediate: def_local
                    .intermediate_states
                    .map(|v| v.map(Into::into))
                    .as_slice(),
                result: &def_hook_vk_commit.map(Into::into),
                enabled: &compute_def_hook_commit,
            },
        );

        /*
         * Finally, we need to generate some expressions for use in the outer constraints.
         * dag_commit_cond is non-zero iff on a transition row and all deferral flags are
         * the same, and consistency_mult is the number of lookups this AIR will receive
         * on the PvsAirConsistencyBus.
         */
        let dag_commit_cond =
            and(base_local.is_valid, not(def_local.is_last)) * (AB::Expr::ONE - delta);
        let deferral_flag = def_local.child_pvs.deferral_flag.into();
        let consistency_mult = base_local.has_verifier_pvs + AB::Expr::ONE;

        (dag_commit_cond, deferral_flag, consistency_mult)
    }
}
