use std::{array::from_fn, borrow::Borrow};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::SubAir;
use openvm_recursion_circuit::bus::{
    CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
    PreHashBus, PreHashMessage, PublicValuesBus, PublicValuesBusMessage,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use verify_stark::pvs::{
    DagCommit, DeferralPvs, VerifierBasePvs, CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID,
};

use crate::{
    bn254::{CommitBytes, DagCommitBytes},
    circuit::{
        deferral::{
            aggregation::hook::bus::{
                DefVkCommitBus, DefVkCommitMessage, OnionResultBus, OnionResultMessage,
            },
            DeferralAggregationPvs, DEF_AGG_PVS_AIR_ID,
        },
        root::NUM_DIGESTS_IN_VK_COMMIT,
        subair::{HashSliceCtx, HashSliceSubAir, MerkleRootBus, MerkleRootMessage},
        utils::{assert_dag_commit_eq, assert_dag_commit_unset, vk_commit_components},
        CONSTRAINT_EVAL_CACHED_INDEX,
    },
    utils::{digests_to_poseidon2_input, pad_slice_to_poseidon2_input, zero_hash},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralHookPvsCols<F> {
    pub verifier_pvs: VerifierBasePvs<F>,
    pub def_pvs: DeferralAggregationPvs<F>,

    pub intermediate_vk_states: [[F; POSEIDON2_WIDTH]; NUM_DIGESTS_IN_VK_COMMIT - 1],
    pub def_vk_commit: [F; DIGEST_SIZE],

    pub input_onion: [F; DIGEST_SIZE],
    pub output_onion: [F; DIGEST_SIZE],

    pub def_vk_commit_padded: [F; DIGEST_SIZE],
    pub input_onion_padded: [F; DIGEST_SIZE],
    pub output_onion_padded: [F; DIGEST_SIZE],
}

pub struct DeferralHookPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub pre_hash_bus: PreHashBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub hash_slice_subair: HashSliceSubAir,

    pub def_vk_commit_bus: DefVkCommitBus,
    pub merkle_root_bus: MerkleRootBus,
    pub onion_res_bus: OnionResultBus,

    pub expected_internal_recursive_dag_commit: DagCommitBytes,
    pub zero_hash: CommitBytes,
}

impl DeferralHookPvsAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        public_values_bus: PublicValuesBus,
        cached_commit_bus: CachedCommitBus,
        pre_hash_bus: PreHashBus,
        poseidon2_compress_bus: Poseidon2CompressBus,
        hash_slice_subair: HashSliceSubAir,
        def_vk_commit_bus: DefVkCommitBus,
        merkle_root_bus: MerkleRootBus,
        onion_res_bus: OnionResultBus,
        expected_internal_recursive_dag_commit: DagCommitBytes,
    ) -> Self {
        let zero_hash = zero_hash(1).into();
        Self {
            public_values_bus,
            cached_commit_bus,
            pre_hash_bus,
            poseidon2_compress_bus,
            hash_slice_subair,
            def_vk_commit_bus,
            merkle_root_bus,
            onion_res_bus,
            expected_internal_recursive_dag_commit,
            zero_hash,
        }
    }
}

impl<F: Field> BaseAir<F> for DeferralHookPvsAir {
    fn width(&self) -> usize {
        DeferralHookPvsCols::<u8>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for DeferralHookPvsAir {
    fn num_public_values(&self) -> usize {
        DeferralPvs::<u8>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for DeferralHookPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for DeferralHookPvsAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have one element");
        let local: &DeferralHookPvsCols<AB::Var> = (*local).borrow();

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
            local
                .verifier_pvs
                .internal_for_leaf_dag_commit
                .cached_commit[i]
                * (AB::Expr::TWO - local.verifier_pvs.recursion_flag)
                + local
                    .verifier_pvs
                    .internal_recursive_dag_commit
                    .cached_commit[i]
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

        self.pre_hash_bus.receive(
            builder,
            AB::F::ZERO,
            PreHashMessage::<AB::F> {
                vk_pre_hash: self.expected_internal_recursive_dag_commit.pre_hash.into(),
            },
            AB::F::ONE,
        );

        assert_dag_commit_unset(
            &mut builder.when_ne(local.verifier_pvs.recursion_flag, AB::F::TWO),
            local.verifier_pvs.internal_recursive_dag_commit,
        );

        assert_dag_commit_eq(
            &mut builder.when_ne(local.verifier_pvs.recursion_flag, AB::F::ONE),
            local.verifier_pvs.internal_recursive_dag_commit,
            DagCommit::<AB::Expr>::from(self.expected_internal_recursive_dag_commit),
        );

        /*
         * Commit def_vk_commit is hash_slice of the 6 vk_commit_components (cached_commit and
         * vk_pre_hash for each of def_dag_commit (called app_dag_commit in the struct),
         * leaf_dag_commit, and internal_for_leaf_dag_commit).
         * We constrain this here and send def_vk_commit to its bus.
         */
        let vk_commit_components: Vec<_> = vk_commit_components(&local.verifier_pvs)
            .into_iter()
            .map(|c| c.map(Into::into))
            .collect();
        self.hash_slice_subair.eval(
            builder,
            HashSliceCtx {
                elements: vk_commit_components.as_slice(),
                intermediate: local
                    .intermediate_vk_states
                    .map(|v| v.map(Into::into))
                    .as_slice(),
                result: &local.def_vk_commit.map(Into::into),
                enabled: &AB::Expr::ONE,
            },
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
                    self.zero_hash.into(),
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
