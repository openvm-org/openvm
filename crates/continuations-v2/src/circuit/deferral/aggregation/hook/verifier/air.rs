use std::{array::from_fn, borrow::Borrow};

use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PublicValuesBus, PublicValuesBusMessage,
    },
    utils::assert_zeros,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{DeferralPvs, CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID};

use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::{
            aggregation::hook::bus::{
                DefVkCommitBus, DefVkCommitMessage, OnionResultBus, OnionResultMessage,
            },
            DeferralAggregationPvs, DeferralVerifierPvs, DEF_AGG_PVS_AIR_ID,
        },
        subair::{MerkleRootBus, MerkleRootMessage},
        CONSTRAINT_EVAL_CACHED_INDEX,
    },
    utils::{digests_to_poseidon2_input, pad_slice_to_poseidon2_input, zero_hash},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralRootPvsCols<F> {
    pub verifier_pvs: DeferralVerifierPvs<F>,
    pub def_pvs: DeferralAggregationPvs<F>,

    pub intermediate_vk_commit: [F; DIGEST_SIZE],
    pub def_vk_commit: [F; DIGEST_SIZE],

    pub input_onion: [F; DIGEST_SIZE],
    pub output_onion: [F; DIGEST_SIZE],

    pub def_vk_commit_padded: [F; DIGEST_SIZE],
    pub input_onion_padded: [F; DIGEST_SIZE],
    pub output_onion_padded: [F; DIGEST_SIZE],
}

pub struct DeferralRootPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,

    pub def_vk_commit_bus: DefVkCommitBus,
    pub merkle_root_bus: MerkleRootBus,
    pub onion_res_bus: OnionResultBus,

    pub expected_internal_recursive_dag_commit: CommitBytes,
    pub zero_hash: CommitBytes,
}

impl DeferralRootPvsAir {
    pub fn new(
        public_values_bus: PublicValuesBus,
        cached_commit_bus: CachedCommitBus,
        poseidon2_compress_bus: Poseidon2CompressBus,
        def_vk_commit_bus: DefVkCommitBus,
        merkle_root_bus: MerkleRootBus,
        onion_res_bus: OnionResultBus,
        expected_internal_recursive_dag_commit: CommitBytes,
    ) -> Self {
        let zero_hash = zero_hash(1).into();
        Self {
            public_values_bus,
            cached_commit_bus,
            poseidon2_compress_bus,
            def_vk_commit_bus,
            merkle_root_bus,
            onion_res_bus,
            expected_internal_recursive_dag_commit,
            zero_hash,
        }
    }
}

impl<F: Field> BaseAir<F> for DeferralRootPvsAir {
    fn width(&self) -> usize {
        DeferralRootPvsCols::<u8>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DeferralRootPvsAir {
    fn num_public_values(&self) -> usize {
        DeferralPvs::<u8>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for DeferralRootPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for DeferralRootPvsAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have one element");
        let local: &DeferralRootPvsCols<AB::Var> = (*local).borrow();

        /*
         * Check that the final proof is computed by the internal recursive prover, i.e.
         * that internal_flag is 2 and recursion flag is 1 or 2.
         */
        builder.assert_eq(local.verifier_pvs.internal_flag, AB::F::TWO);
        builder.assert_bool(local.verifier_pvs.recursion_flag - AB::F::ONE);

        /*
         * We need to receive public values from ProofShapeModule to ensure the values read
         * here are correct. All verifier public values will be at VERIFIER_PVS_AIR_ID,
         * while the deferral public values will be at DEF_AGG_PVS_AIR_ID.
         */
        let verifier_pvs_id = AB::Expr::from_usize(VERIFIER_PVS_AIR_ID);
        let def_pvs_id = AB::Expr::from_usize(DEF_AGG_PVS_AIR_ID);

        for (pv_idx, value) in local.verifier_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                AB::F::ONE,
            );
        }

        for (pv_idx, value) in local.def_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: def_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                AB::F::ONE,
            );
        }

        /*
         * We also need to receive the cached commit from ProofShapeModule, which is either for
         * the internal-for-leaf (i.e. if recursion_flag == 1) or internal-recursive layer. In
         * the former case we constrain verifier_pvs.internal_recursive_dag_commit to be unset
         * (i.e. all 0), and in the latter we constrain it to be equal to a pre-generated
         * constant as it should be the same regardless of def_vk (provided internal system
         * params are the same).
         */
        let cached_commit = from_fn(|i| {
            local.verifier_pvs.internal_for_leaf_dag_commit[i]
                * (AB::Expr::TWO - local.verifier_pvs.recursion_flag)
                + local.verifier_pvs.internal_recursive_dag_commit[i]
                    * (local.verifier_pvs.recursion_flag - AB::F::ONE)
        });
        self.cached_commit_bus.receive(
            builder,
            AB::F::ZERO,
            CachedCommitBusMessage {
                air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                cached_commit,
            },
            AB::F::ONE,
        );

        assert_zeros(
            &mut builder.when_ne(local.verifier_pvs.recursion_flag, AB::F::TWO),
            local.verifier_pvs.internal_recursive_dag_commit,
        );

        assert_array_eq(
            &mut builder.when_ne(local.verifier_pvs.recursion_flag, AB::F::ONE),
            local.verifier_pvs.internal_recursive_dag_commit,
            <CommitBytes as Into<[u32; DIGEST_SIZE]>>::into(
                self.expected_internal_recursive_dag_commit,
            )
            .map(AB::F::from_u32),
        );

        /*
         * Commit def_vk_commit should be the compression of def_dag_commit, leaf_dag_commit, and
         * internal_for_leaf_dag_commit. We constrain this here and send def_vk_commit to its
         * bus.
         */
        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.verifier_pvs.def_dag_commit,
                    local.verifier_pvs.leaf_dag_commit,
                ),
                output: local.intermediate_vk_commit,
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.intermediate_vk_commit,
                    local.verifier_pvs.internal_for_leaf_dag_commit,
                ),
                output: local.def_vk_commit,
            },
            AB::F::ONE,
        );

        self.def_vk_commit_bus.send(
            builder,
            DefVkCommitMessage {
                def_vk_commit: local.def_vk_commit,
            },
            AB::F::ONE,
        );

        /*
         * The input_onion and output_onion are computed by decommitting def_pvs.merkle_commit
         * and performing two onion hashes. Both steps are done in different AIRs, but we
         * must receive these commits to constrain consistency.
         */
        self.merkle_root_bus.receive(
            builder,
            MerkleRootMessage {
                merkle_root: local.def_pvs.merkle_commit.map(Into::into),
                idx: AB::Expr::ZERO,
            },
            AB::F::ONE,
        );

        self.onion_res_bus.receive(
            builder,
            OnionResultMessage {
                input_onion: local.input_onion,
                output_onion: local.output_onion,
            },
            AB::F::ONE,
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. initial_acc_hash should be the compression of a padded
         * def_vk_commit, and final_acc_hash should be the compression of the input and
         * output onions. Note that the Merkle root computation hashes each leaf with the
         * zero digest prior.
         */
        let &DeferralPvs::<_> {
            initial_acc_hash,
            final_acc_hash,
            depth,
        } = builder.public_values().borrow();

        builder.assert_one(depth);

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pad_slice_to_poseidon2_input(
                    &local.def_vk_commit.map(Into::into),
                    AB::Expr::ZERO,
                ),
                output: local.def_vk_commit_padded.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pad_slice_to_poseidon2_input(
                    &local.input_onion.map(Into::into),
                    AB::Expr::ZERO,
                ),
                output: local.input_onion_padded.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pad_slice_to_poseidon2_input(
                    &local.output_onion.map(Into::into),
                    AB::Expr::ZERO,
                ),
                output: local.output_onion_padded.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.def_vk_commit_padded.map(Into::into),
                    <CommitBytes as Into<[u32; DIGEST_SIZE]>>::into(self.zero_hash)
                        .map(AB::Expr::from_u32),
                ),
                output: initial_acc_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.input_onion_padded.map(Into::into),
                    local.output_onion_padded.map(Into::into),
                ),
                output: final_acc_hash.map(Into::into),
            },
            AB::F::ONE,
        );
    }
}
