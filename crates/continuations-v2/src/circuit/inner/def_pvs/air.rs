use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::utils::{assert_array_eq, not};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PublicValuesBus, PublicValuesBusMessage,
    },
    prelude::DIGEST_SIZE,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{DeferralPvs, CONSTRAINT_EVAL_AIR_ID, DEF_PVS_AIR_ID};

use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::DEF_HOOK_PVS_AIR_ID,
        inner::bus::{PvsAirConsistencyBus, PvsAirConsistencyMessage},
        CONSTRAINT_EVAL_CACHED_INDEX,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralPvsCols<F> {
    pub row_idx: F,
    pub deferral_flag: F,
    pub has_verifier_pvs: F,

    pub proof_idx: F,
    pub is_present: F,
    pub single_present_is_right: F,

    pub child_pvs: DeferralPvs<F>,
}

pub struct DeferralPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub poseidon2_bus: Poseidon2CompressBus,
    pub pvs_air_consistency_bus: PvsAirConsistencyBus,

    pub expected_def_hook_commit: CommitBytes,
}

impl<F> BaseAir<F> for DeferralPvsAir {
    fn width(&self) -> usize {
        DeferralPvsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralPvsAir {
    fn num_public_values(&self) -> usize {
        DeferralPvs::<u8>::width()
    }
}
impl<F> PartitionedBaseAir<F> for DeferralPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for DeferralPvsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &DeferralPvsCols<AB::Var> = (*local).borrow();
        let next: &DeferralPvsCols<AB::Var> = (*next).borrow();

        /*
         * This AIR may have 1 or 2 rows. There are 4 valid 1-row cases:
         * - deferral_flag == 0: child deferral pvs are unset
         * - deferral_flag == 1 && proof_idx == 0: wrapping a deferral proof
         * - deferral_flag == 1 && proof_idx == 1: combining a VM and deferral proof
         * - deferral_flag == 2: wrapping a combined proof
         *
         * There are 2 valid 2-row cases, both with deferral_flag == 1:
         * - Both child proofs are present
         * - The first proof is present and the second is absent
         */
        // constrain that when hash_pvs is set we have exactly 2 def rows
        builder.assert_bool(local.row_idx);
        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_one(next.row_idx - local.row_idx);

        let has_two_rows = (next.row_idx - local.row_idx).square();
        let has_one_row = not::<AB::Expr>(has_two_rows.clone());

        // constrain that all the present rows are at the beginning
        builder.assert_bool(local.is_present);
        builder
            .when_transition()
            .assert_bool(local.is_present - next.is_present);

        // constrain if deferral_flag is set, there is at least one present proof
        builder.assert_tern(local.deferral_flag);
        builder.assert_eq(local.deferral_flag, next.deferral_flag);
        builder
            .when(local.deferral_flag)
            .assert_bool(local.is_present + next.is_present - AB::Expr::ONE);

        // basic constraints for consistency columns
        builder.when_first_row().assert_bool(local.proof_idx);
        builder
            .when_first_row()
            .when(local.proof_idx)
            .assert_one(has_one_row.clone());
        builder.assert_bool(local.has_verifier_pvs);
        builder.assert_eq(local.has_verifier_pvs, next.has_verifier_pvs);

        // constrain single_present_is_right is set when there is 1 present and
        // 1 absent row
        builder.assert_bool(local.single_present_is_right);
        builder.assert_eq(local.single_present_is_right, next.single_present_is_right);
        builder
            .when(local.single_present_is_right)
            .assert_one(local.is_present + next.is_present);

        /*
         * When deferral_flag is unset, there must be a single row with zeros for
         * public values.
         */
        let mut when_flag_not_one = builder.when_ne(local.deferral_flag, AB::Expr::ONE);
        let mut when_invalid = when_flag_not_one.when_ne(local.deferral_flag, AB::Expr::TWO);

        when_invalid.assert_one(has_one_row.clone());
        when_invalid.assert_zero(local.is_present);
        for child_pv in local.child_pvs.as_slice() {
            when_invalid.assert_zero(*child_pv);
        }

        /*
         * If there are two rows and a proof is absent, it represents an accumulator
         * Merkle subtree that has been left untouched. We constrain its initial and
         * final accumulator hashes to be equal. Additionally, if there are two rows
         * then the child_pvs depth should be equal.
         */
        assert_array_eq(
            &mut builder
                .when(has_two_rows.clone())
                .when(not(local.is_present)),
            local.child_pvs.initial_acc_hash,
            local.child_pvs.final_acc_hash,
        );

        builder
            .when(has_two_rows.clone())
            .assert_eq(local.child_pvs.depth, next.child_pvs.depth);

        /*
         * If this row is present then we need to receive the child public values
         * from ProofShapeModule. At the hook level this is at DEF_HOOK_PVS_AIR_ID,
         * at every other level it will be at DEF_PVS_AIR_ID.
         */
        let def_pvs_air_idx = AB::Expr::from_usize(DEF_PVS_AIR_ID) * local.has_verifier_pvs
            + AB::Expr::from_usize(DEF_HOOK_PVS_AIR_ID) * not(local.has_verifier_pvs);
        for (pv_idx, value) in local.child_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: def_pvs_air_idx.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                local.is_present,
            );
        }

        /*
         * We look up proof metadata from VerifierPvsAir here to ensure consistency
         * on each row.
         */
        self.pvs_air_consistency_bus.lookup_key(
            builder,
            local.proof_idx,
            PvsAirConsistencyMessage {
                deferral_flag: local.deferral_flag,
                has_verifier_pvs: local.has_verifier_pvs,
            },
            local.is_present,
        );

        /*
         * If this row corresponds to a direct deferral hook circuit child (i.e.
         * has_verifier_pvs == 0), receive the child's cached trace commit and
         * constrain it to an expected constant.
         */
        let expected_def_hook_commit =
            <CommitBytes as Into<[u32; DIGEST_SIZE]>>::into(self.expected_def_hook_commit);
        self.cached_commit_bus.receive(
            builder,
            local.proof_idx,
            CachedCommitBusMessage {
                air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                cached_commit: expected_def_hook_commit.map(AB::Expr::from_u32),
            },
            local.is_present * not(local.has_verifier_pvs),
        );

        /*
         * Finally, we constrain the public values to be consistent with the
         * child's. If there is one row then the pvs are simply passed through.
         * If there are two, then initial_acc_hash and final_acc_hash are
         * combined and depth is incremented by 1.
         */
        let &DeferralPvs::<_> {
            initial_acc_hash,
            final_acc_hash,
            depth,
        } = builder.public_values().borrow();

        // constrain that pvs are passed through if there is one row
        let mut when_one_row = builder.when(has_one_row);
        when_one_row.assert_eq(local.child_pvs.depth, depth);

        assert_array_eq(
            &mut when_one_row,
            local.child_pvs.initial_acc_hash,
            initial_acc_hash,
        );
        assert_array_eq(
            &mut when_one_row,
            local.child_pvs.final_acc_hash,
            final_acc_hash,
        );

        // constrain that pvs are updated properly if there are two rows
        let row_delta = next.row_idx - local.row_idx;
        let single_present_is_left = not(local.single_present_is_right);
        let single_present_is_local =
            row_delta.clone() * (row_delta + AB::Expr::ONE) * AB::F::TWO.inverse();

        let left_init_child = from_fn(|i| {
            single_present_is_left.clone() * local.child_pvs.initial_acc_hash[i]
                + local.single_present_is_right * next.child_pvs.initial_acc_hash[i]
        });
        let right_init_child = from_fn(|i| {
            local.single_present_is_right * local.child_pvs.initial_acc_hash[i]
                + single_present_is_left.clone() * next.child_pvs.initial_acc_hash[i]
        });

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(left_init_child, right_init_child),
                output: initial_acc_hash.map(Into::into),
            },
            single_present_is_local.clone(),
        );

        let left_final_child = from_fn(|i| {
            single_present_is_left.clone() * local.child_pvs.final_acc_hash[i]
                + local.single_present_is_right * next.child_pvs.final_acc_hash[i]
        });
        let right_final_child = from_fn(|i| {
            local.single_present_is_right * local.child_pvs.final_acc_hash[i]
                + single_present_is_left.clone() * next.child_pvs.final_acc_hash[i]
        });

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(left_final_child, right_final_child),
                output: final_acc_hash.map(Into::into),
            },
            single_present_is_local,
        );

        builder
            .when(has_two_rows)
            .assert_one(depth.into() - local.child_pvs.depth);
    }
}
