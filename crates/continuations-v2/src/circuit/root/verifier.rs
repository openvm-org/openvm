use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::arch::ExitCode;
use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use recursion_circuit::bus::{
    CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
    PublicValuesBus, PublicValuesBusMessage,
};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    poseidon2::sponge::{poseidon2_compress, poseidon2_hash_slice},
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{NonRootVerifierPvs, VERIFIER_PVS_AIR_ID};

use crate::circuit::{
    CONSTRAINT_EVAL_AIR_ID, CONSTRAINT_EVAL_CACHED_INDEX,
    root::{
        RootVerifierPvs,
        bus::{
            MemoryMerkleCommitBus, MemoryMerkleCommitMessage, UserPvsCommitBus,
            UserPvsCommitMessage,
        },
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootVerifierPvsCols<F> {
    pub child_pvs: NonRootVerifierPvs<F>,

    pub program_commit_hash: [F; DIGEST_SIZE],
    pub initial_root_hash: [F; DIGEST_SIZE],
    pub initial_pc_hash: [F; DIGEST_SIZE],
    pub intermediate_exe_commit: [F; DIGEST_SIZE],
}

pub struct RootVerifierPvsAir<F> {
    pub public_values_bus: PublicValuesBus,
    pub cached_commit_bus: CachedCommitBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,

    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub memory_merkle_commit_bus: MemoryMerkleCommitBus,

    pub expected_internal_recursive_vk_commit: [F; DIGEST_SIZE],
}

impl<F: Field> BaseAir<F> for RootVerifierPvsAir<F> {
    fn width(&self) -> usize {
        RootVerifierPvsCols::<u8>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for RootVerifierPvsAir<F> {
    fn num_public_values(&self) -> usize {
        RootVerifierPvs::<u8>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for RootVerifierPvsAir<F> {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for RootVerifierPvsAir<AB::F>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let local: &RootVerifierPvsCols<AB::Var> = (*local).borrow();

        /*
         * Since RootVerifierPvs is a true subset of NonRootVerifierPvs, we must constrain
         * the child public values not in RootVerifierPvs here. We start with is_terminate
         * exit code.
         */
        let success = AB::F::from_u32(ExitCode::Success as u32);
        builder.assert_eq(local.child_pvs.exit_code, success);
        builder.assert_one(local.child_pvs.is_terminate);

        /*
         * Check that the final proof is computed by the internal recursive prover, i.e.
         * that internal_flag is 2 and recursion flag is 1 or 2.
         */
        builder.assert_eq(local.child_pvs.internal_flag, AB::F::TWO);
        builder.assert_bool(local.child_pvs.recursion_flag - AB::F::ONE);

        /*
         * UserPvsCommitAir's public values are the original app user public values, and
         * thus it needs to constrain that user_pv_commit is the values' merkle root. We
         * also need to constrain that the revealed public values were part of the final
         * memory state - we do this in UserPvsInMemoryAir, which expects to receive both
         * user_pv_commit and final_root.
         */
        self.user_pvs_commit_bus.send(
            builder,
            UserPvsCommitMessage {
                user_pvs_commit: local.child_pvs.user_pv_commit,
            },
            AB::F::TWO,
        );

        self.memory_merkle_commit_bus.send(
            builder,
            MemoryMerkleCommitMessage {
                merkle_root: local.child_pvs.final_root,
            },
            AB::F::ONE,
        );

        /*
         * We still need to receive public values from ProofShapeModule to ensure the values
         * being read here are correct. All public values will be at VERIFIER_PVS_AIR_ID.
         */
        let verifier_pvs_id = AB::Expr::from_usize(VERIFIER_PVS_AIR_ID);
        let connector_pvs_offset = AB::Expr::from_usize(2 * DIGEST_SIZE);

        self.public_values_bus.receive(
            builder,
            AB::F::ZERO,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.clone(),
                pv_idx: connector_pvs_offset.clone(),
                value: local.child_pvs.initial_pc.into(),
            },
            AB::F::ONE,
        );

        self.public_values_bus.receive(
            builder,
            AB::F::ZERO,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.clone(),
                pv_idx: connector_pvs_offset.clone() + AB::F::ONE,
                value: local.child_pvs.final_pc.into(),
            },
            AB::F::ONE,
        );

        self.public_values_bus.receive(
            builder,
            AB::F::ZERO,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.clone(),
                pv_idx: connector_pvs_offset.clone() + AB::F::TWO,
                value: local.child_pvs.exit_code.into(),
            },
            AB::F::ONE,
        );

        self.public_values_bus.receive(
            builder,
            AB::F::ZERO,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.clone(),
                pv_idx: connector_pvs_offset.clone() + AB::F::from_u8(3),
                value: local.child_pvs.is_terminate.into(),
            },
            AB::F::ONE,
        );

        let merkle_pvs_offset = connector_pvs_offset + AB::Expr::from_u8(4);
        let verifier_pvs_offset = merkle_pvs_offset.clone() + AB::F::from_usize(2 * DIGEST_SIZE);
        let recursive_pvs_offset =
            verifier_pvs_offset.clone() + AB::F::from_usize(1 + 3 * DIGEST_SIZE);

        self.public_values_bus.receive(
            builder,
            AB::F::ZERO,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.clone(),
                pv_idx: verifier_pvs_offset.clone(),
                value: local.child_pvs.internal_flag.into(),
            },
            AB::F::ONE,
        );

        self.public_values_bus.receive(
            builder,
            AB::F::ZERO,
            PublicValuesBusMessage {
                air_idx: verifier_pvs_id.clone(),
                pv_idx: recursive_pvs_offset.clone(),
                value: local.child_pvs.recursion_flag.into(),
            },
            AB::F::ONE,
        );

        for didx in 0..DIGEST_SIZE {
            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(didx),
                    value: local.child_pvs.user_pv_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: AB::Expr::from_usize(didx + DIGEST_SIZE),
                    value: local.child_pvs.program_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: merkle_pvs_offset.clone() + AB::F::from_usize(didx),
                    value: local.child_pvs.initial_root[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: merkle_pvs_offset.clone() + AB::F::from_usize(didx + DIGEST_SIZE),
                    value: local.child_pvs.final_root[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: verifier_pvs_offset.clone() + AB::F::from_usize(didx + 1),
                    value: local.child_pvs.app_vk_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: verifier_pvs_offset.clone() + AB::F::from_usize(didx + DIGEST_SIZE + 1),
                    value: local.child_pvs.leaf_vk_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: verifier_pvs_offset.clone()
                        + AB::F::from_usize(didx + 2 * DIGEST_SIZE + 1),
                    value: local.child_pvs.internal_for_leaf_vk_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: recursive_pvs_offset.clone() + AB::F::from_usize(didx + 1),
                    value: local.child_pvs.internal_recursive_vk_commit[didx].into(),
                },
                AB::F::ONE,
            );
        }

        /*
         * We also need to receive the cached commit from ProofShapeModule, which is for the
         * internal-recursive layer. Note that given some internal system params config for
         * any app_vk the internal-recursive commit will be constant, and thus we cosntrain
         * it to be that constant.
         */
        self.cached_commit_bus.receive(
            builder,
            AB::F::ZERO,
            CachedCommitBusMessage {
                air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                cached_commit: local.child_pvs.internal_recursive_vk_commit.map(Into::into),
            },
            AB::F::ONE,
        );

        assert_array_eq(
            builder,
            local.child_pvs.internal_recursive_vk_commit,
            self.expected_internal_recursive_vk_commit,
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. The app_vk_commit, leaf_vk_commit, and internal_for_leaf_vk_commit
         * are simply constrained to be equal to the child's.
         */
        let &RootVerifierPvs::<_> {
            app_exe_commit,
            app_vk_commit,
            leaf_vk_commit,
            internal_for_leaf_vk_commit,
        } = builder.public_values().borrow();

        assert_array_eq(builder, local.child_pvs.app_vk_commit, app_vk_commit);
        assert_array_eq(builder, local.child_pvs.leaf_vk_commit, leaf_vk_commit);
        assert_array_eq(
            builder,
            local.child_pvs.internal_for_leaf_vk_commit,
            internal_for_leaf_vk_commit,
        );

        /*
         * The app_exe_commit is a commit to the app program, initial memory state, and initial
         * PC. Child public values program_commit, initial_root, and initial_pc are inidivudally
         * hashed and then permuted together to produce app_exe_commit.
         */
        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: from_fn(|i| {
                    if i < DIGEST_SIZE {
                        local.child_pvs.program_commit[i].into()
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                output: local.program_commit_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: from_fn(|i| {
                    if i < DIGEST_SIZE {
                        local.child_pvs.initial_root[i].into()
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                output: local.initial_root_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: from_fn(|i| {
                    if i == 0 {
                        local.child_pvs.initial_pc.into()
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                output: local.initial_pc_hash.map(Into::into),
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: from_fn(|i| {
                    if i < DIGEST_SIZE {
                        local.program_commit_hash[i]
                    } else {
                        local.initial_root_hash[i - DIGEST_SIZE]
                    }
                }),
                output: local.intermediate_exe_commit,
            },
            AB::F::ONE,
        );

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: from_fn(|i| {
                    if i < DIGEST_SIZE {
                        local.intermediate_exe_commit[i].into()
                    } else {
                        local.initial_pc_hash[i - DIGEST_SIZE].into()
                    }
                }),
                output: app_exe_commit.map(Into::into),
            },
            AB::F::ONE,
        );
    }
}

pub fn generate_proving_ctx(proof: Proof) -> AirProvingContextV2<CpuBackendV2> {
    let width = RootVerifierPvsCols::<u8>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut RootVerifierPvsCols<F> = trace.as_mut_slice().borrow_mut();
    let child_pvs: &NonRootVerifierPvs<F> =
        proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();

    cols.child_pvs = *child_pvs;

    cols.program_commit_hash = poseidon2_hash_slice(&child_pvs.program_commit);
    cols.initial_root_hash = poseidon2_hash_slice(&child_pvs.initial_root);
    cols.initial_pc_hash = poseidon2_hash_slice(&[child_pvs.initial_pc]);
    cols.intermediate_exe_commit =
        poseidon2_compress(cols.program_commit_hash, cols.initial_root_hash);

    let mut public_values = vec![F::ZERO; RootVerifierPvs::<u8>::width()];
    let root_pvs: &mut RootVerifierPvs<F> = public_values.as_mut_slice().borrow_mut();

    root_pvs.app_exe_commit =
        poseidon2_compress(cols.intermediate_exe_commit, cols.initial_pc_hash);
    root_pvs.app_vk_commit = child_pvs.app_vk_commit;
    root_pvs.leaf_vk_commit = child_pvs.leaf_vk_commit;
    root_pvs.internal_for_leaf_vk_commit = child_pvs.internal_for_leaf_vk_commit;

    AirProvingContextV2 {
        cached_mains: vec![],
        common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
        public_values,
    }
}
