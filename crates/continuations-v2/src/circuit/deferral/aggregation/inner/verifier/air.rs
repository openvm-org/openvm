use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::utils::{and, assert_array_eq, not};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{CachedCommitBus, CachedCommitBusMessage, PublicValuesBus, PublicValuesBusMessage},
    utils::assert_zeros,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{VerifierBasePvs, CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID};

use crate::circuit::{
    deferral::aggregation::inner::bus::{DefPvsConsistencyBus, DefPvsConsistencyMessage},
    CONSTRAINT_EVAL_CACHED_INDEX,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralVerifierPvsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub has_verifier_pvs: F,
    pub child_pvs: VerifierBasePvs<F>,
}

pub enum DeferralChildLevel {
    App,
    Leaf,
    InternalForLeaf,
    InternalRecursive,
}

pub struct DeferralVerifierPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub def_pvs_consistency_bus: DefPvsConsistencyBus,
}

impl<F> BaseAir<F> for DeferralVerifierPvsAir {
    fn width(&self) -> usize {
        DeferralVerifierPvsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralVerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width()
    }
}
impl<F> PartitionedBaseAir<F> for DeferralVerifierPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for DeferralVerifierPvsAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &DeferralVerifierPvsCols<AB::Var> = (*local).borrow();
        let next: &DeferralVerifierPvsCols<AB::Var> = (*next).borrow();

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
         * what layer a verifier is at using the has_verifier_pvs and internal_flag columns. If
         * has_verifier_pvs is 0, then we have a leaf verifier that is verifying a deferral
         * circuit proof. If has_verifier_pvs is 1 and internal_flag is 0, then
         * we have an internal verifier that has leaf children, and if internal_flag is 1 then
         * we have internal_for_leaf children. Finally, if internal_flag is 2 then we are
         * verifying internal_recursive children.
         *
         * Similarly, if the child recursion_flag is 1 then we know that we are on the 2nd (i.e.
         * index 1) internal_recursive layer, and if it's 2 then we are on the 3rd or beyond.
         */
        // constrain verifier pvs columns are the same across all rows (note app_dag_commit is
        // def_dag_commit here)
        let mut when_both = builder.when(and(local.is_valid, next.is_valid));
        let is_leaf = not(local.has_verifier_pvs);
        let is_internal = local.has_verifier_pvs;

        when_both.assert_eq(local.has_verifier_pvs, next.has_verifier_pvs);
        when_both.assert_eq(local.child_pvs.internal_flag, next.child_pvs.internal_flag);
        when_both.assert_eq(
            local.child_pvs.recursion_flag,
            next.child_pvs.recursion_flag,
        );
        assert_array_eq(
            &mut when_both,
            local.child_pvs.app_dag_commit,
            next.child_pvs.app_dag_commit,
        );
        assert_array_eq(
            &mut when_both,
            local.child_pvs.leaf_dag_commit,
            next.child_pvs.leaf_dag_commit,
        );
        assert_array_eq(
            &mut when_both,
            local.child_pvs.internal_for_leaf_dag_commit,
            next.child_pvs.internal_for_leaf_dag_commit,
        );
        assert_array_eq(
            &mut when_both,
            local.child_pvs.internal_recursive_dag_commit,
            next.child_pvs.internal_recursive_dag_commit,
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
        builder
            .when(is_leaf.clone())
            .assert_zero(local.child_pvs.internal_flag);

        assert_zeros(
            &mut builder.when(is_leaf.clone()),
            local.child_pvs.app_dag_commit,
        );
        assert_zeros(
            &mut builder.when(
                (local.child_pvs.internal_flag - AB::F::ONE)
                    * (local.child_pvs.internal_flag - AB::F::TWO),
            ),
            local.child_pvs.leaf_dag_commit,
        );
        assert_zeros(
            &mut builder.when(local.child_pvs.internal_flag - AB::F::TWO),
            local.child_pvs.internal_for_leaf_dag_commit,
        );
        assert_zeros(
            &mut builder.when(local.child_pvs.recursion_flag - AB::F::TWO),
            local.child_pvs.internal_recursive_dag_commit,
        );

        /*
         * If has_verifier_pvs is true (i.e. we are on some internal level) we need to receive
         * public values from ProofShapeModule to ensure the values being read here are correct.
         * Each inner public value should have air_idx VERIFIER_PVS_AIR_ID.
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
         * We want to ensure consistency between AIRs that process public values, and we do so
         * using the def_pvs_consistency_bus.
         */
        self.def_pvs_consistency_bus.send(
            builder,
            local.proof_idx,
            DefPvsConsistencyMessage {
                has_verifier_pvs: local.has_verifier_pvs,
            },
            local.is_valid,
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. Note that we only impose constraints for layers below the current
         * one - it is impossible for the current layer to know its own commit, and future layers
         * will catch if we preemptively define a current or future verifier commit.
         */
        let &VerifierBasePvs::<_> {
            internal_flag,
            app_dag_commit: def_dag_commit,
            leaf_dag_commit,
            internal_for_leaf_dag_commit,
            recursion_flag,
            internal_recursive_dag_commit,
        } = builder.public_values().borrow();

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

        // constrain def_dag_commit is set at all internal levels
        assert_array_eq(
            &mut builder.when(is_internal),
            local.child_pvs.app_dag_commit,
            def_dag_commit,
        );

        // constrain verifier-specific pvs at all internal_recursive levels
        builder
            .when(local.child_pvs.internal_flag)
            .assert_zero(internal_flag.into() - AB::F::TWO);
        assert_array_eq(
            &mut builder.when(local.child_pvs.internal_flag),
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
        assert_array_eq(
            &mut builder.when(local.child_pvs.recursion_flag),
            local.child_pvs.internal_for_leaf_dag_commit,
            internal_for_leaf_dag_commit,
        );

        // constrain verifier-specific pvs at internal_recursive levels after the second
        assert_array_eq(
            &mut builder.when(
                local.child_pvs.recursion_flag * (local.child_pvs.recursion_flag - AB::F::ONE),
            ),
            local.child_pvs.internal_recursive_dag_commit,
            internal_recursive_dag_commit,
        );

        /*
         * We also need to receive cached commits from ProofShapeModule. Note that the deferral
         * circuit cached commits are received in another AIR, so only the internal verifier will
         * receive them here.
         */
        let is_internal_flag_zero = (local.child_pvs.internal_flag - AB::F::ONE)
            * (local.child_pvs.internal_flag - AB::F::TWO)
            * AB::F::TWO.inverse();
        let is_internal_flag_one =
            (AB::Expr::TWO - local.child_pvs.internal_flag) * local.child_pvs.internal_flag;
        let is_recursion_flag_one =
            (AB::Expr::TWO - local.child_pvs.recursion_flag) * local.child_pvs.recursion_flag;
        let is_recursion_flag_two = (local.child_pvs.recursion_flag - AB::F::ONE)
            * local.child_pvs.recursion_flag
            * AB::F::TWO.inverse();
        let cached_commit = from_fn(|i| {
            is_internal_flag_zero.clone() * local.child_pvs.app_dag_commit[i]
                + is_internal_flag_one.clone() * local.child_pvs.leaf_dag_commit[i]
                + is_recursion_flag_one.clone() * local.child_pvs.internal_for_leaf_dag_commit[i]
                + is_recursion_flag_two.clone() * local.child_pvs.internal_recursive_dag_commit[i]
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
    }
}
