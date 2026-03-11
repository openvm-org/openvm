use std::{array::from_fn, borrow::Borrow};

use openvm_circuit::arch::{ExitCode, POSEIDON2_WIDTH};
use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PreHashBus, PreHashMessage, PublicValuesBus, PublicValuesBusMessage,
    },
    utils::assert_zeros,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use openvm_verify_stark_host::pvs::{
    DagCommit, DeferralPvs, VerifierBasePvs, VerifierDefPvs, VmPvs, CONSTRAINT_EVAL_AIR_ID,
    DEF_PVS_AIR_ID, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    bn254::{CommitBytes, DagCommitBytes},
    circuit::{
        root::{
            bus::{
                DeferralAccPathBus, DeferralAccPathMessage, DeferralMerkleRootsBus,
                DeferralMerkleRootsMessage, MemoryMerkleCommitBus, MemoryMerkleCommitMessage,
            },
            RootVerifierPvs, NUM_DIGESTS_IN_VK_COMMIT,
        },
        subair::{HashSliceCtx, HashSliceSubAir},
        utils::{assert_dag_commit_eq, assert_dag_commit_unset, vk_commit_components},
        CONSTRAINT_EVAL_CACHED_INDEX,
    },
    utils::{digests_to_poseidon2_input, pad_slice_to_poseidon2_input},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootVerifierPvsCols<F> {
    pub child_verifier_pvs: VerifierBasePvs<F>,
    pub child_vm_pvs: VmPvs<F>,

    pub program_commit_hash: [F; DIGEST_SIZE],
    pub initial_root_hash: [F; DIGEST_SIZE],
    pub initial_pc_hash: [F; DIGEST_SIZE],
    pub intermediate_exe_commit: [F; DIGEST_SIZE],

    pub intermediate_vk_states: [[F; POSEIDON2_WIDTH]; NUM_DIGESTS_IN_VK_COMMIT - 1],
}

pub struct RootVerifierPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub pre_hash_bus: PreHashBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub memory_merkle_commit_bus: MemoryMerkleCommitBus,
    pub def_acc_paths_bus: DeferralAccPathBus,
    pub def_merkle_roots_bus: DeferralMerkleRootsBus,

    pub hash_slice_subair: HashSliceSubAir,

    pub expected_internal_recursive_dag_commit: DagCommitBytes,
    pub expected_def_hook_commit: Option<CommitBytes>,
}

impl<F: Field> BaseAir<F> for RootVerifierPvsAir {
    fn width(&self) -> usize {
        RootVerifierPvsCols::<u8>::width()
            + if self.expected_def_hook_commit.is_some() {
                RootDefVerifierCols::<u8>::width()
            } else {
                0
            }
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for RootVerifierPvsAir {
    fn num_public_values(&self) -> usize {
        RootVerifierPvs::<u8>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for RootVerifierPvsAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for RootVerifierPvsAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let base_cols_width = RootVerifierPvsCols::<AB::Var>::width();
        let (base_local, rec_local) = local.split_at(base_cols_width);
        let local: &RootVerifierPvsCols<AB::Var> = (*base_local).borrow();

        if let Some(def_hook_commit) = self.expected_def_hook_commit {
            let def: &RootDefVerifierCols<AB::Var> = (*rec_local).borrow();
            self.eval_deferrals(builder, local, def, def_hook_commit);
        }

        /*
         * RootVerifierPvs only exposes a reduced/derived subset of the child public values,
         * so we must constrain required child public values here. We start with is_terminate
         * and exit code.
         */
        let success = AB::F::from_u32(ExitCode::Success as u32);
        builder.assert_eq(local.child_vm_pvs.exit_code, success);
        builder.assert_one(local.child_vm_pvs.is_terminate);

        /*
         * Check that the final proof is computed by the internal recursive prover, i.e.
         * that internal_flag is 2 and recursion flag is 1 or 2.
         */
        builder.assert_eq(local.child_verifier_pvs.internal_flag, AB::F::TWO);
        builder.assert_bool(local.child_verifier_pvs.recursion_flag - AB::F::ONE);

        /*
         * UserPvsCommitAir constrains that the Merkle root of the original app user public
         * values is some user_pvs_commit. We also need to constrain that those values were
         * part of the final memory state - we do this in UserPvsInMemoryAir, which has to
         * receive final_root in order to verify a Merkle proof from user_pvs_commit to it.
         */
        self.memory_merkle_commit_bus.send(
            builder,
            MemoryMerkleCommitMessage {
                merkle_root: local.child_vm_pvs.final_root,
            },
            AB::F::ONE,
        );

        /*
         * We still need to receive public values from ProofShapeModule to ensure the values
         * being read here are correct. Vm and verifier public values are on separate AIRs.
         */
        let vm_pvs_id = AB::Expr::from_usize(VM_PVS_AIR_ID);
        let verifier_pvs_id = AB::Expr::from_usize(VERIFIER_PVS_AIR_ID);

        for (pv_idx, value) in local.child_vm_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: vm_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                AB::F::ONE,
            );
        }

        for (pv_idx, value) in local.child_verifier_pvs.as_slice().iter().enumerate() {
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

        /*
         * We also need to receive the cached commit from ProofShapeModule, which is either for
         * the internal-for-leaf (i.e. if recursion_flag == 1) or internal-recursive layer. In
         * the former case we constrain child_pvs.internal_recursive_dag_commit to be unset (i.e.
         * all 0), and in the latter we constrain it to be equal to a pre-generated constant as
         * it should be the same regardless of app_vk (provided internal system params are the
         * same).
         */
        let cached_commit = from_fn(|i| {
            local
                .child_verifier_pvs
                .internal_for_leaf_dag_commit
                .cached_commit[i]
                * (AB::Expr::TWO - local.child_verifier_pvs.recursion_flag)
                + local
                    .child_verifier_pvs
                    .internal_recursive_dag_commit
                    .cached_commit[i]
                    * (local.child_verifier_pvs.recursion_flag - AB::F::ONE)
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
            &mut builder.when_ne(local.child_verifier_pvs.recursion_flag, AB::F::TWO),
            local.child_verifier_pvs.internal_recursive_dag_commit,
        );

        assert_dag_commit_eq(
            &mut builder.when_ne(local.child_verifier_pvs.recursion_flag, AB::F::ONE),
            local.child_verifier_pvs.internal_recursive_dag_commit,
            DagCommit::<AB::Expr>::from(self.expected_internal_recursive_dag_commit),
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. The app_vk_commit is constrained to be hash_slice of the 6
         * vk_commit_components (cached_commit and vk_pre_hash for each of app, leaf, and
         * internal-for-leaf DAG commits).
         */
        let &RootVerifierPvs::<_> {
            app_exe_commit,
            app_vk_commit,
        } = builder.public_values().borrow();

        let vk_commit_components: Vec<_> = vk_commit_components(&local.child_verifier_pvs)
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
                result: &app_vk_commit.map(Into::into),
                enabled: &AB::Expr::ONE,
            },
        );

        /*
         * The app_exe_commit is a commit to the app program, initial memory state, and initial
         * PC. Child public values program_commit, initial_root, and initial_pc are individually
         * hashed and then permuted together to produce app_exe_commit.
         */
        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pad_slice_to_poseidon2_input(
                    &local.child_vm_pvs.program_commit.map(Into::into),
                    AB::Expr::ZERO,
                ),
                output: local.program_commit_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pad_slice_to_poseidon2_input(
                    &local.child_vm_pvs.initial_root.map(Into::into),
                    AB::Expr::ZERO,
                ),
                output: local.initial_root_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: pad_slice_to_poseidon2_input(
                    &[local.child_vm_pvs.initial_pc.into()],
                    AB::Expr::ZERO,
                ),
                output: local.initial_pc_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.program_commit_hash,
                    local.initial_root_hash,
                ),
                output: local.intermediate_exe_commit,
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.intermediate_exe_commit,
                    local.initial_pc_hash,
                )
                .map(Into::into),
                output: app_exe_commit.map(Into::into),
            },
            AB::F::ONE,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootDefVerifierCols<F> {
    pub child_def_verifier_pvs: VerifierDefPvs<F>,
    pub child_def_pvs: DeferralPvs<F>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootVerifierCombinedPvs<F> {
    pub base: RootVerifierPvsCols<F>,
    pub def: RootDefVerifierCols<F>,
}

impl RootVerifierPvsAir {
    fn eval_deferrals<AB>(
        &self,
        builder: &mut AB,
        base: &RootVerifierPvsCols<AB::Var>,
        def: &RootDefVerifierCols<AB::Var>,
        expected_def_hook_commit: CommitBytes,
    ) where
        AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues,
    {
        let verifier_pvs = def.child_def_verifier_pvs;
        let def_pvs = def.child_def_pvs;

        /*
         * We need to receive the additional public values here.
         */
        let def_pvs_idx = AB::Expr::from_usize(DEF_PVS_AIR_ID);
        let verifier_pvs_id = AB::Expr::from_usize(VERIFIER_PVS_AIR_ID);
        let base_verifier_pvs_width = VerifierBasePvs::<u8>::width();

        for (pv_idx, value) in verifier_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx + base_verifier_pvs_width),
                    value: (*value).into(),
                },
                AB::F::ONE,
            );
        }

        for (pv_idx, value) in def_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: def_pvs_idx.clone(),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: (*value).into(),
                },
                AB::F::ONE,
            );
        }

        /*
         * The final internal-recursive proof's deferral_flag must be either 0 or 2, and
         * in the latter case the def_hook_commit must match the expected one.
         */
        builder
            .assert_zero(verifier_pvs.deferral_flag * (verifier_pvs.deferral_flag - AB::Expr::TWO));
        assert_array_eq::<_, _, AB::Expr, _>(
            &mut builder.when(verifier_pvs.deferral_flag),
            verifier_pvs.def_hook_vk_commit,
            expected_def_hook_commit.into(),
        );

        /*
         * If deferral_flag is 2, there must be a Merkle path from initial_acc_hash to
         * initial_root and from final_acc_hash to final_root. Else, we constrain that
         * they are 0 and that the deferral address space remained unchanged.
         */
        let mut when_no_deferral = builder.when_ne(verifier_pvs.deferral_flag, AB::Expr::TWO);
        when_no_deferral.assert_zero(def_pvs.depth);
        assert_zeros(&mut when_no_deferral, def_pvs.initial_acc_hash);
        assert_zeros(&mut when_no_deferral, def_pvs.final_acc_hash);

        self.def_acc_paths_bus.send(
            builder,
            DeferralAccPathMessage {
                initial_acc_hash: def_pvs.initial_acc_hash.map(Into::into),
                final_acc_hash: def_pvs.final_acc_hash.map(Into::into),
                depth: def_pvs.depth.into(),
                is_unset: (AB::Expr::TWO - verifier_pvs.deferral_flag) * AB::F::TWO.inverse(),
            },
            AB::F::ONE,
        );

        self.def_merkle_roots_bus.send(
            builder,
            DeferralMerkleRootsMessage {
                initial_root: base.child_vm_pvs.initial_root,
                final_root: base.child_vm_pvs.final_root,
            },
            AB::F::ONE,
        );
    }
}
