use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_poseidon2_air::Permutation;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, poseidon2_perm, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{NonRootVerifierPvs, VERIFIER_PVS_AIR_ID};

use crate::circuit::root::{
    digests_to_poseidon2_input, pad_slice_to_poseidon2_input, verifier::air::RootVerifierPvsCols,
    RootVerifierPvs,
};

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

    cols.intermediate_vk_commit =
        poseidon2_compress_with_capacity(child_pvs.app_dag_commit, child_pvs.leaf_dag_commit).0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        child_pvs.app_dag_commit,
        child_pvs.leaf_dag_commit,
    ));

    let mut public_values = vec![F::ZERO; RootVerifierPvs::<u8>::width()];
    let root_pvs: &mut RootVerifierPvs<F> = public_values.as_mut_slice().borrow_mut();

    root_pvs.app_exe_commit =
        poseidon2_compress_with_capacity(cols.intermediate_exe_commit, cols.initial_pc_hash).0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        cols.intermediate_exe_commit,
        cols.initial_pc_hash,
    ));

    root_pvs.app_vk_commit = poseidon2_compress_with_capacity(
        cols.intermediate_vk_commit,
        child_pvs.internal_for_leaf_dag_commit,
    )
    .0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        cols.intermediate_vk_commit,
        child_pvs.internal_for_leaf_dag_commit,
    ));

    (
        AirProvingContext {
            cached_mains: vec![],
            common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
            public_values,
        },
        poseidon2_compress_inputs,
    )
}
