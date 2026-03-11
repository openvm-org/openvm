use std::borrow::{Borrow, BorrowMut};

use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::Permutation;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, poseidon2_perm, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::{
    DeferralPvs, VerifierBasePvs, VerifierDefPvs, VmPvs, DEF_PVS_AIR_ID, VERIFIER_PVS_AIR_ID,
    VM_PVS_AIR_ID,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    circuit::{
        root::{
            verifier::air::{RootDefVerifierCols, RootVerifierPvsCols},
            RootVerifierPvs,
        },
        subair::hash_slice_trace,
        SingleAirTraceData,
    },
    utils::pad_slice_to_poseidon2_input,
};

pub fn generate_proving_ctx<SC: StarkProtocolConfig<F = F>>(
    proof: &Proof<BabyBearPoseidon2Config>,
    deferral_enabled: bool,
) -> SingleAirTraceData<CpuBackend<SC>> {
    let base_width = RootVerifierPvsCols::<u8>::width();
    let def_width = RootDefVerifierCols::<u8>::width();
    let width = base_width + if deferral_enabled { def_width } else { 0 };
    let mut trace = vec![F::ZERO; width];

    let (base_cols_slice, def_cols_slice) = trace.as_mut_slice().split_at_mut(base_width);
    let cols: &mut RootVerifierPvsCols<F> = base_cols_slice.borrow_mut();

    let (base_pvs_slice, def_pvs_slice) = proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());
    let child_verifier_pvs: &VerifierBasePvs<F> = base_pvs_slice.borrow();
    let child_vm_pvs: &VmPvs<F> = proof.public_values[VM_PVS_AIR_ID].as_slice().borrow();

    cols.child_verifier_pvs = *child_verifier_pvs;
    cols.child_vm_pvs = *child_vm_pvs;

    let padded_program_commit = pad_slice_to_poseidon2_input(&child_vm_pvs.program_commit, F::ZERO);
    let padded_initial_root = pad_slice_to_poseidon2_input(&child_vm_pvs.initial_root, F::ZERO);
    let padded_initial_pc = pad_slice_to_poseidon2_input(&[child_vm_pvs.initial_pc], F::ZERO);

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
    let mut poseidon2_permute_inputs = Vec::new();

    poseidon2_compress_inputs.extend_from_slice(&[
        padded_program_commit,
        padded_initial_root,
        padded_initial_pc,
    ]);

    cols.intermediate_exe_commit =
        poseidon2_compress_with_capacity(cols.program_commit_hash, cols.initial_root_hash).0;
    poseidon2_compress_inputs.push(crate::utils::digests_to_poseidon2_input(
        cols.program_commit_hash,
        cols.initial_root_hash,
    ));

    let vk_elements = [
        child_verifier_pvs.app_dag_commit.cached_commit,
        child_verifier_pvs.app_dag_commit.vk_pre_hash,
        child_verifier_pvs.leaf_dag_commit.cached_commit,
        child_verifier_pvs.leaf_dag_commit.vk_pre_hash,
        child_verifier_pvs
            .internal_for_leaf_dag_commit
            .cached_commit,
        child_verifier_pvs.internal_for_leaf_dag_commit.vk_pre_hash,
    ];
    let (intermediate_vk_states, app_vk_commit) = hash_slice_trace(
        &vk_elements,
        Some(&mut poseidon2_permute_inputs),
        Some(&mut poseidon2_compress_inputs),
    );
    cols.intermediate_vk_states = intermediate_vk_states.try_into().unwrap();

    let mut public_values = vec![F::ZERO; RootVerifierPvs::<u8>::width()];
    let root_pvs: &mut RootVerifierPvs<F> = public_values.as_mut_slice().borrow_mut();

    root_pvs.app_exe_commit =
        poseidon2_compress_with_capacity(cols.intermediate_exe_commit, cols.initial_pc_hash).0;
    poseidon2_compress_inputs.push(crate::utils::digests_to_poseidon2_input(
        cols.intermediate_exe_commit,
        cols.initial_pc_hash,
    ));

    root_pvs.app_vk_commit = app_vk_commit;

    if deferral_enabled {
        let def_verifier_pvs: &VerifierDefPvs<F> = def_pvs_slice.borrow();
        let def_pvs: &DeferralPvs<F> = proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow();
        let def_cols: &mut RootDefVerifierCols<F> = def_cols_slice.borrow_mut();
        def_cols.child_def_verifier_pvs = *def_verifier_pvs;
        def_cols.child_def_pvs = *def_pvs;
    }

    SingleAirTraceData {
        air_proving_ctx: AirProvingContext {
            cached_mains: vec![],
            common_main: RowMajorMatrix::new(trace, width),
            public_values,
        },
        poseidon2_compress_inputs,
        poseidon2_permute_inputs,
    }
}
