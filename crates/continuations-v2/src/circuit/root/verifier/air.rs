use std::{array::from_fn, borrow::Borrow};

use openvm_circuit::arch::ExitCode;
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
use verify_stark::pvs::{
    VerifierBasePvs, VmPvs, CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID,
};

use crate::{
    bn254::CommitBytes,
    circuit::{
        root::{
            bus::{MemoryMerkleCommitBus, MemoryMerkleCommitMessage},
            digests_to_poseidon2_input, pad_slice_to_poseidon2_input, RootVerifierPvs,
        },
        CONSTRAINT_EVAL_CACHED_INDEX,
    },
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

    pub intermediate_vk_commit: [F; DIGEST_SIZE],
}

pub struct RootVerifierPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub memory_merkle_commit_bus: MemoryMerkleCommitBus,

    pub expected_internal_recursive_dag_commit: CommitBytes,
}

impl<F: Field> BaseAir<F> for RootVerifierPvsAir {
    fn width(&self) -> usize {
        RootVerifierPvsCols::<u8>::width()
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
        let local: &RootVerifierPvsCols<AB::Var> = (*local).borrow();

        /*
         * RootVerifierPvs only exposes a reduced/derived subset of NonRootVerifierPvs,
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
            local.child_verifier_pvs.internal_for_leaf_dag_commit[i]
                * (AB::Expr::TWO - local.child_verifier_pvs.recursion_flag)
                + local.child_verifier_pvs.internal_recursive_dag_commit[i]
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

        assert_zeros(
            &mut builder.when_ne(local.child_verifier_pvs.recursion_flag, AB::F::TWO),
            local.child_verifier_pvs.internal_recursive_dag_commit,
        );

        assert_array_eq(
            &mut builder.when_ne(local.child_verifier_pvs.recursion_flag, AB::F::ONE),
            local.child_verifier_pvs.internal_recursive_dag_commit,
            <CommitBytes as Into<[u32; DIGEST_SIZE]>>::into(
                self.expected_internal_recursive_dag_commit,
            )
            .map(AB::F::from_u32),
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. The app_vk_commit is constrained to be the hashed combination of the
         * child's app, leaf, and internal-for-leaf DAG commits.
         */
        let &RootVerifierPvs::<_> {
            app_exe_commit,
            app_vk_commit,
        } = builder.public_values().borrow();

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.child_verifier_pvs.app_dag_commit,
                    local.child_verifier_pvs.leaf_dag_commit,
                ),
                output: local.intermediate_vk_commit,
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(
                    local.intermediate_vk_commit.map(Into::into),
                    local
                        .child_verifier_pvs
                        .internal_for_leaf_dag_commit
                        .map(Into::into),
                ),
                output: app_vk_commit.map(Into::into),
            },
            AB::F::ONE,
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
