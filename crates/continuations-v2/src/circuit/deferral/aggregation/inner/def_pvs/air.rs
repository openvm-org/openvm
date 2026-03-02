use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::utils::{assert_array_eq, not};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, PublicValuesBus, PublicValuesBusMessage,
    },
    prelude::DIGEST_SIZE,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    circuit::deferral::{
        aggregation::inner::bus::{
            InputOrMerkleCommitBus, InputOrMerkleCommitMessage, PvAirConsistencyBus,
            PvAirConsistencyMessage,
        },
        DeferralAggregationPvs, DeferralCircuitPvs, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralPvsCols<F> {
    pub proof_idx: F,
    pub is_present: F,
    pub has_verifier_pvs: F,

    pub merkle_commit: [F; DIGEST_SIZE],
    pub child_pvs: DeferralCircuitPvs<F>,
}

pub struct DeferralPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub poseidon2_bus: Poseidon2CompressBus,
    pub input_or_merkle_commit_bus: InputOrMerkleCommitBus,
    pub pv_air_consistency_bus: PvAirConsistencyBus,
}

impl<F> BaseAir<F> for DeferralPvsAir {
    fn width(&self) -> usize {
        DeferralPvsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralPvsAir {
    fn num_public_values(&self) -> usize {
        DeferralAggregationPvs::<u8>::width()
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

        /*
         * We need to receive public values here to ensure the values read are correct.
         * output_commit is received from ProofShapeModule, and the (possibly cached-folded)
         * input_commit/merkle_commit is received from InputCommitAir. output_commit is only
         * present for leaf children.
         *
         * A row may be non-present (is_present = 0). For those rows, child-specific public
         * values are intentionally not constrained here; consistency is enforced by the
         * final Merkle root checks at the deferral root layer.
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
            is_leaf * local.is_present,
        );

        /*
         * We want to ensure consistency between AIRs that process public values, and we do so
         * using the pv_air_consistency_bus.
         */
        self.pv_air_consistency_bus.receive(
            builder,
            local.proof_idx,
            PvAirConsistencyMessage {
                has_verifier_pvs: local.has_verifier_pvs,
            },
            local.is_present,
        );

        /*
         * If there is only one row, then this proof is a wrapper and its public values should
         * match those of its child's. Otherwise, constrain that merkle_commit is the hash of
         * the child proofs'.
         */
        let &DeferralAggregationPvs::<_> { merkle_commit } = builder.public_values().borrow();

        let is_first = not(local.proof_idx);
        let is_first_of_two_rows = is_first.clone() * next.proof_idx;

        assert_array_eq(
            &mut builder.when(is_first * not(next.proof_idx)),
            local.merkle_commit,
            merkle_commit,
        );

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
