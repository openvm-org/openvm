use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::arch::{ExitCode, POSEIDON2_WIDTH};
use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_poseidon2_air::Permutation;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, poseidon2_perm, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PublicValuesBus, PublicValuesBusMessage,
    },
    utils::assert_zeros,
};
use stark_recursion_circuit_derive::AlignedBorrow;
use verify_stark::pvs::{NonRootVerifierPvs, CONSTRAINT_EVAL_AIR_ID, VERIFIER_PVS_AIR_ID};

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
    pub child_pvs: NonRootVerifierPvs<F>,

    pub program_commit_hash: [F; DIGEST_SIZE],
    pub initial_root_hash: [F; DIGEST_SIZE],
    pub initial_pc_hash: [F; DIGEST_SIZE],
    pub intermediate_exe_commit: [F; DIGEST_SIZE],
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
         * constrains that the Merkle root of those public values is some user_pvs_commit.
         * We also need to constrain that the revealed public values were part of the final
         * memory state - we do this in UserPvsInMemoryAir, which has to receive final_root
         * in order to verify a Merkle proof from user_pvs_commit to it.
         */
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
        let connector_pvs_offset = AB::Expr::from_usize(DIGEST_SIZE);

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
                    value: local.child_pvs.app_dag_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: verifier_pvs_offset.clone() + AB::F::from_usize(didx + DIGEST_SIZE + 1),
                    value: local.child_pvs.leaf_dag_commit[didx].into(),
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
                    value: local.child_pvs.internal_for_leaf_dag_commit[didx].into(),
                },
                AB::F::ONE,
            );

            self.public_values_bus.receive(
                builder,
                AB::F::ZERO,
                PublicValuesBusMessage {
                    air_idx: verifier_pvs_id.clone(),
                    pv_idx: recursive_pvs_offset.clone() + AB::F::from_usize(didx + 1),
                    value: local.child_pvs.internal_recursive_dag_commit[didx].into(),
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
            local.child_pvs.internal_for_leaf_dag_commit[i]
                * (AB::Expr::TWO - local.child_pvs.recursion_flag)
                + local.child_pvs.internal_recursive_dag_commit[i]
                    * (local.child_pvs.recursion_flag - AB::F::ONE)
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
            &mut builder.when_ne(local.child_pvs.recursion_flag, AB::F::TWO),
            local.child_pvs.internal_recursive_dag_commit,
        );

        assert_array_eq(
            &mut builder.when_ne(local.child_pvs.recursion_flag, AB::F::ONE),
            local.child_pvs.internal_recursive_dag_commit,
            <CommitBytes as Into<[u32; DIGEST_SIZE]>>::into(
                self.expected_internal_recursive_dag_commit,
            )
            .map(AB::F::from_u32),
        );

        /*
         * Finally, we need to constrain that the public values this AIR produces are consistent
         * with the child's. The app_dag_commit, leaf_dag_commit, and
         * internal_for_leaf_dag_commit are simply constrained to be equal to the
         * child's.
         */
        let &RootVerifierPvs::<_> {
            app_exe_commit,
            app_dag_commit,
            leaf_dag_commit,
            internal_for_leaf_dag_commit,
        } = builder.public_values().borrow();

        assert_array_eq(builder, local.child_pvs.app_dag_commit, app_dag_commit);
        assert_array_eq(builder, local.child_pvs.leaf_dag_commit, leaf_dag_commit);
        assert_array_eq(
            builder,
            local.child_pvs.internal_for_leaf_dag_commit,
            internal_for_leaf_dag_commit,
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
                    &local.child_pvs.program_commit.map(Into::into),
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
                    &local.child_pvs.initial_root.map(Into::into),
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
                    &[local.child_pvs.initial_pc.into()],
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

pub fn generate_proving_ctx(
    proof: &Proof<BabyBearPoseidon2Config>,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    let width = RootVerifierPvsCols::<u8>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut RootVerifierPvsCols<F> = trace.as_mut_slice().borrow_mut();
    let child_pvs: &NonRootVerifierPvs<F> =
        proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();

    cols.child_pvs = *child_pvs;

    let padded_program_commit = pad_slice_to_poseidon2_input(&child_pvs.program_commit, F::ZERO);
    let padded_initial_root = pad_slice_to_poseidon2_input(&child_pvs.initial_root, F::ZERO);
    let padded_initial_pc = pad_slice_to_poseidon2_input(&[child_pvs.initial_pc], F::ZERO);

    let perm = poseidon2_perm();
    cols.program_commit_hash = perm.permute(padded_program_commit)[..DIGEST_SIZE]
        .try_into()
        .unwrap();
    cols.initial_root_hash = perm.permute(padded_initial_root)[..DIGEST_SIZE]
        .try_into()
        .unwrap();
    cols.initial_pc_hash = perm.permute(padded_initial_pc)[..DIGEST_SIZE]
        .try_into()
        .unwrap();

    let mut poseidon2_compress_inputs = Vec::with_capacity(5);
    poseidon2_compress_inputs.extend_from_slice(&[
        padded_program_commit,
        padded_initial_root,
        padded_initial_pc,
    ]);

    cols.intermediate_exe_commit =
        poseidon2_compress_with_capacity(cols.program_commit_hash, cols.initial_root_hash).0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        cols.program_commit_hash,
        cols.initial_root_hash,
    ));

    let mut public_values = vec![F::ZERO; RootVerifierPvs::<u8>::width()];
    let root_pvs: &mut RootVerifierPvs<F> = public_values.as_mut_slice().borrow_mut();

    root_pvs.app_exe_commit =
        poseidon2_compress_with_capacity(cols.intermediate_exe_commit, cols.initial_pc_hash).0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        cols.intermediate_exe_commit,
        cols.initial_pc_hash,
    ));

    root_pvs.app_dag_commit = child_pvs.app_dag_commit;
    root_pvs.leaf_dag_commit = child_pvs.leaf_dag_commit;
    root_pvs.internal_for_leaf_dag_commit = child_pvs.internal_for_leaf_dag_commit;

    (
        AirProvingContext {
            cached_mains: vec![],
            common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
            public_values,
        },
        poseidon2_compress_inputs,
    )
}
