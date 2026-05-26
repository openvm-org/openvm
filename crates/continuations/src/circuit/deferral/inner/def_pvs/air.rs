use std::{array::from_fn, borrow::Borrow};

use itertools::Itertools;
use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{assert_array_eq, not},
    ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_recursion_circuit::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, PublicValuesBus, PublicValuesBusMessage,
    },
    prelude::{DIGEST_SIZE, F},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::poseidon2_compress_with_capacity;
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
        MAX_DEF_AGG_MERKLE_DEPTH,
    },
    utils::digests_to_poseidon2_input,
    CommitBytes,
};

const ENCODER_MAX_DEGREE: u32 = 2;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DeferralAggPvsCols<F> {
    pub proof_idx: F,
    pub is_present: F,
    pub has_verifier_pvs: F,

    pub merkle_commit: [F; DIGEST_SIZE],
    pub child_pvs: DeferralCircuitPvs<F>,
}

#[derive(ColumnsAir)]
#[columns_via(DeferralAggPvsCols<u8>)]
pub struct DeferralAggPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub poseidon2_bus: Poseidon2CompressBus,
    pub input_or_merkle_commit_bus: InputOrMerkleCommitBus,
    pub def_pvs_consistency_bus: DefPvsConsistencyBus,

    pub encoder: Encoder,
    pub zero_hashes: [CommitBytes; MAX_DEF_AGG_MERKLE_DEPTH + 1],
}

impl DeferralAggPvsAir {
    pub(super) fn depth_encoder() -> Encoder {
        Encoder::new(MAX_DEF_AGG_MERKLE_DEPTH + 1, ENCODER_MAX_DEGREE, false)
    }

    pub fn new(
        public_values_bus: PublicValuesBus,
        poseidon2_bus: Poseidon2CompressBus,
        input_or_merkle_commit_bus: InputOrMerkleCommitBus,
        def_pvs_consistency_bus: DefPvsConsistencyBus,
    ) -> Self {
        let encoder = Self::depth_encoder();
        assert!(encoder.width() < DIGEST_SIZE);
        let mut zero_hash = [F::ZERO; DIGEST_SIZE];
        Self {
            public_values_bus,
            poseidon2_bus,
            input_or_merkle_commit_bus,
            def_pvs_consistency_bus,
            encoder,
            zero_hashes: from_fn(|_| {
                zero_hash = poseidon2_compress_with_capacity(zero_hash, zero_hash).0;
                zero_hash.into()
            }),
        }
    }
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
        let local_merkle_depth = local.child_pvs.input_commit[1];

        let mut when_dummy_internal = builder.when(not(local.is_present) * is_internal);
        when_dummy_internal.assert_zero(local_num_def_circuit_proofs);
        when_dummy_internal.assert_zero(local_merkle_depth);

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

        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: AB::Expr::from_usize(DEF_AGG_PVS_AIR_ID),
                pv_idx: AB::Expr::from_usize(DIGEST_SIZE + 1),
                value: local_merkle_depth.into(),
            },
            is_internal * local.is_present,
        );

        /*
         * If there is only one row, then this proof is a wrapper and its public values should
         * match those of its child's. Otherwise, constrain that num_def_circuit_proofs is the
         * sum of the present children's and that merkle_depth is the child merkle_depth + 1.
         * If both children are present, we constrain that their merkle_depth values are equal.
         */
        let &DeferralAggregationPvs::<_> {
            merkle_commit,
            num_def_circuit_proofs,
            merkle_depth,
        } = builder.public_values().borrow();

        let is_first = not(local.proof_idx);
        let is_first_of_two_rows = is_first.clone() * next.proof_idx;

        let local_num_proofs =
            local.is_present * is_leaf.clone() + is_internal * local_num_def_circuit_proofs;
        let next_num_proofs =
            next.is_present * is_leaf + is_internal * next.child_pvs.input_commit[0];
        let local_merkle_depth = is_internal * local_merkle_depth;
        let next_merkle_depth = is_internal * next.child_pvs.input_commit[1];

        let mut when_one_row = builder.when(is_first * not(next.proof_idx));
        when_one_row.assert_eq(local_num_proofs.clone(), num_def_circuit_proofs);
        when_one_row.assert_eq(local_merkle_depth.clone(), merkle_depth);
        assert_array_eq(&mut when_one_row, local.merkle_commit, merkle_commit);

        builder
            .when_transition()
            .assert_eq(local_num_proofs + next_num_proofs, num_def_circuit_proofs);
        builder
            .when_transition()
            .assert_eq(local_merkle_depth.clone() + AB::Expr::ONE, merkle_depth);
        builder
            .when_transition()
            .when(next.is_present)
            .assert_eq(local_merkle_depth, next_merkle_depth);

        /*
         * If there are two rows, then the second (next) is either present or not. If present
         * the parent merkle_hash should be the compression of both children's merkle_commit.
         * If not the second merkle_commit contains encoder flags for the child merkle depth,
         * which is used to select the correct zero hash depth.
         */
        let flags = next
            .merkle_commit
            .into_iter()
            .take(self.encoder.width())
            .collect_vec();
        self.encoder
            .eval(&mut builder.when(not(next.is_present)), &flags);

        let mut zero_hash = [AB::Expr::ZERO; DIGEST_SIZE];
        for i in 0..=MAX_DEF_AGG_MERKLE_DEPTH {
            let is_current = self.encoder.get_flag_expr::<AB>(i, &flags);
            builder
                .when(not(next.is_present))
                .when(is_current.clone())
                .assert_eq(merkle_depth, AB::Expr::from_usize(i + 1));

            let current_zero_hash: [AB::F; DIGEST_SIZE] = self.zero_hashes[i].into();
            for j in 0..DIGEST_SIZE {
                zero_hash[j] += is_current.clone() * current_zero_hash[j];
            }
        }

        let right_child = from_fn(|i| {
            next.is_present * next.merkle_commit[i] + not(next.is_present) * zero_hash[i].clone()
        });

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.merkle_commit.map(Into::into), right_child),
                output: merkle_commit.map(Into::into),
            },
            is_first_of_two_rows,
        );
    }
}
