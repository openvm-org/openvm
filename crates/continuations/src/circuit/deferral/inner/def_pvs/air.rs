use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::utils::{assert_array_eq, not};
use openvm_recursion_circuit::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, PublicValuesBus, PublicValuesBusMessage,
    },
    prelude::DIGEST_SIZE,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    circuit::deferral::{
        inner::bus::{
            DefPvsConsistencyBus, DefPvsConsistencyMessage, InputOrMerkleCommitBus,
            InputOrMerkleCommitMessage,
        },
        DeferralAggregationPvs, DeferralCircuitPvs, DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralAggPvsCols<F> {
    pub proof_idx: F,
    pub is_present: F,
    pub has_verifier_pvs: F,

    pub merkle_commit: [F; DIGEST_SIZE],
    pub child_pvs: DeferralCircuitPvs<F>,
}

pub struct DeferralAggPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub poseidon2_bus: Poseidon2CompressBus,
    pub input_or_merkle_commit_bus: InputOrMerkleCommitBus,
    pub def_pvs_consistency_bus: DefPvsConsistencyBus,
}

impl<F> BaseAir<F> for DeferralAggPvsAir {
    fn width(&self) -> usize {
        DeferralAggPvsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralAggPvsAir {
    fn num_public_values(&self) -> usize {
        DeferralAggregationPvs::<u8>::width()
    }
}
impl<F> PartitionedBaseAir<F> for DeferralAggPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for DeferralAggPvsAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &DeferralAggPvsCols<AB::Var> = (*local).borrow();
        let next: &DeferralAggPvsCols<AB::Var> = (*next).borrow();

        /*
         * This AIR may have 1 or 2 rows. The first row is always present, while the second row
         * may be padding (is_present = 0). It is assumed these rows are adjacent leaves in the
         * deferral aggregation tree; otherwise the final root-layer Merkle check will fail.
         */
        builder.assert_bool(local.is_present);
        builder.when_first_row().assert_one(local.is_present);

        builder.assert_bool(local.proof_idx);
        builder.when_first_row().assert_zero(local.proof_idx);
        builder
            .when_transition()
            .assert_one(next.proof_idx - local.proof_idx);

        builder.assert_bool(local.has_verifier_pvs);
        builder.assert_eq(local.has_verifier_pvs, next.has_verifier_pvs);

        /*
         * We need to receive public values here to ensure the values read are correct.
         * output_commit is received from ProofShapeModule, and the (possibly cached-folded)
         * input_commit/merkle_commit is received from InputCommitAir. output_commit is only
         * present for leaf children.
         *
         * A row may be non-present (is_present = 0). For those rows, child-specific public
         * values are intentionally not constrained here; consistency is enforced by the
         * final Merkle root checks at the deferral hook layer.
         */
        let is_leaf = not(local.has_verifier_pvs);
        let is_internal = local.has_verifier_pvs;
        let air_idx = AB::Expr::from_usize(DEF_CIRCUIT_PVS_AIR_ID);

        for (pv_idx, value) in local.child_pvs.output_commit.iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx: air_idx.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx + DIGEST_SIZE),
                    value: (*value).into(),
                },
                is_leaf.clone() * local.is_present,
            );
        }

        self.input_or_merkle_commit_bus.receive(
            builder,
            local.proof_idx,
            InputOrMerkleCommitMessage {
                has_verifier_pvs: local.has_verifier_pvs.into(),
                commit: from_fn(|i| {
                    is_leaf.clone() * local.child_pvs.input_commit[i]
                        + is_internal * local.merkle_commit[i]
                }),
            },
            local.is_present,
        );

        /*
         * On the leaf layer we need to constrain that merkle_commit is the Poseidon2 compression
         * of input_commit and output_commit.
         */
        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.child_pvs.input_commit,
                    local.child_pvs.output_commit,
                ),
                output: local.merkle_commit,
            },
            is_leaf.clone() * local.is_present,
        );

        /*
         * We want to ensure consistency between AIRs that process public values, and we do so
         * using the def_pvs_consistency_bus.
         */
        self.def_pvs_consistency_bus.receive(
            builder,
            local.proof_idx,
            DefPvsConsistencyMessage {
                has_verifier_pvs: local.has_verifier_pvs,
            },
            local.is_present,
        );

        /*
         * On internal layers we receive num_def_circuit_proofs from PublicValuesBus. To save
         * columns, we repurpose the unused local.child_pvs.input_commit[0].
         */
        let local_num_def_circuit_proofs = local.child_pvs.input_commit[0];

        builder
            .when(not(local.is_present))
            .when(is_internal)
            .assert_zero(local_num_def_circuit_proofs);

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: AB::Expr::from_usize(DEF_AGG_PVS_AIR_ID),
                pv_idx: AB::Expr::from_usize(DIGEST_SIZE),
                value: local_num_def_circuit_proofs.into(),
            },
            is_internal * local.is_present,
        );

        /*
         * If there is only one row, then this proof is a wrapper and its public values should
         * match those of its child's. Otherwise, constrain that merkle_commit is the hash of
         * the child proofs'. We also constrain that num_def_circuit_proofs is the sum of the
         * present children's.
         */
        let &DeferralAggregationPvs::<_> {
            merkle_commit,
            num_def_circuit_proofs,
        } = builder.public_values().borrow();

        let is_first = not(local.proof_idx);
        let is_first_of_two_rows = is_first.clone() * next.proof_idx;

        let local_num_proofs =
            local.is_present * is_leaf.clone() + is_internal * local_num_def_circuit_proofs;
        let next_num_proofs =
            next.is_present * is_leaf + is_internal * next.child_pvs.input_commit[0];

        let mut when_one_row = builder.when(is_first * not(next.proof_idx));
        when_one_row.assert_eq(local_num_proofs.clone(), num_def_circuit_proofs);
        assert_array_eq(&mut when_one_row, local.merkle_commit, merkle_commit);

        builder
            .when_transition()
            .assert_eq(local_num_proofs + next_num_proofs, num_def_circuit_proofs);

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.merkle_commit, next.merkle_commit)
                    .map(Into::into),
                output: merkle_commit.map(Into::into),
            },
            is_first_of_two_rows,
        );
    }
}
