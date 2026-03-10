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
use verify_stark::pvs::{
    DeferralPvs, VerifierBasePvs, VerifierDefPvs, VmPvs, DEF_PVS_AIR_ID, VERIFIER_PVS_AIR_ID,
    VM_PVS_AIR_ID,
};

use crate::{
    circuit::{
        deferral::{
            verify::verifier::{DeferredVerifyPvsCols, RecursiveDeferredVerifyCols},
            DeferralCircuitPvs,
        },
        root::NUM_DIGESTS_IN_VK_COMMIT,
        subair::hash_slice_trace,
    },
    utils::{digests_to_poseidon2_input, pad_slice_to_poseidon2_input, poseidon2_input_to_digests},
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeferredVerifyPvsRecord<F> {
    pub program_commit_hash: [F; DIGEST_SIZE],
    pub initial_root_hash: [F; DIGEST_SIZE],
    pub initial_pc_hash: [F; DIGEST_SIZE],
    pub intermediate_exe_commit: [F; DIGEST_SIZE],
    pub intermediate_vk_states: [[F; POSEIDON2_WIDTH]; NUM_DIGESTS_IN_VK_COMMIT - 1],
    pub app_exe_commit: [F; DIGEST_SIZE],
    pub app_vk_commit: [F; DIGEST_SIZE],
}

pub fn generate_record(
    proof: &Proof<BabyBearPoseidon2Config>,
) -> (
    DeferredVerifyPvsRecord<F>,
    Vec<[F; POSEIDON2_WIDTH]>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    let (base_pvs_slice, _) = proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());
    let child_verifier_pvs: &VerifierBasePvs<F> = base_pvs_slice.borrow();
    let child_vm_pvs: &VmPvs<F> = proof.public_values[VM_PVS_AIR_ID].as_slice().borrow();

    let padded_program_commit = pad_slice_to_poseidon2_input(&child_vm_pvs.program_commit, F::ZERO);
    let padded_initial_root = pad_slice_to_poseidon2_input(&child_vm_pvs.initial_root, F::ZERO);
    let padded_initial_pc = pad_slice_to_poseidon2_input(&[child_vm_pvs.initial_pc], F::ZERO);

    let perm = poseidon2_perm();
    let program_commit_hash = perm.permute(padded_program_commit)[..DIGEST_SIZE]
        .try_into()
        .unwrap();
    let initial_root_hash = perm.permute(padded_initial_root)[..DIGEST_SIZE]
        .try_into()
        .unwrap();
    let initial_pc_hash = perm.permute(padded_initial_pc)[..DIGEST_SIZE]
        .try_into()
        .unwrap();

    let mut poseidon2_compress_inputs = Vec::with_capacity(5);
    let mut poseidon2_permute_inputs = Vec::new();
    poseidon2_compress_inputs.extend_from_slice(&[
        padded_program_commit,
        padded_initial_root,
        padded_initial_pc,
    ]);

    let intermediate_exe_commit =
        poseidon2_compress_with_capacity(program_commit_hash, initial_root_hash).0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        program_commit_hash,
        initial_root_hash,
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
    let (intermediate_vk_states_vec, app_vk_commit) = hash_slice_trace(
        &vk_elements,
        Some(&mut poseidon2_permute_inputs),
        Some(&mut poseidon2_compress_inputs),
    );
    let intermediate_vk_states = intermediate_vk_states_vec.try_into().unwrap();

    let app_exe_commit =
        poseidon2_compress_with_capacity(intermediate_exe_commit, initial_pc_hash).0;
    poseidon2_compress_inputs.push(digests_to_poseidon2_input(
        intermediate_exe_commit,
        initial_pc_hash,
    ));

    (
        DeferredVerifyPvsRecord {
            program_commit_hash,
            initial_root_hash,
            initial_pc_hash,
            intermediate_exe_commit,
            intermediate_vk_states,
            app_exe_commit,
            app_vk_commit,
        },
        poseidon2_compress_inputs,
        poseidon2_permute_inputs,
    )
}

pub fn generate_proving_ctx(
    proof: &Proof<BabyBearPoseidon2Config>,
    record: DeferredVerifyPvsRecord<F>,
    final_transcript_state: [F; POSEIDON2_WIDTH],
    output_commit: [F; DIGEST_SIZE],
    deferral_enabled: bool,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let base_width = DeferredVerifyPvsCols::<u8>::width();
    let rec_width = RecursiveDeferredVerifyCols::<u8>::width();
    let width = base_width + if deferral_enabled { rec_width } else { 0 };

    let mut trace = vec![F::ZERO; width];
    let (base_cols_slice, def_cols_slice) = trace.as_mut_slice().split_at_mut(base_width);
    let cols: &mut DeferredVerifyPvsCols<F> = base_cols_slice.borrow_mut();

    let (base_pvs_slice, def_pvs_slice) = proof.public_values[VERIFIER_PVS_AIR_ID]
        .as_slice()
        .split_at(VerifierBasePvs::<u8>::width());
    let child_verifier_pvs: &VerifierBasePvs<F> = base_pvs_slice.borrow();
    let child_vm_pvs: &VmPvs<F> = proof.public_values[VM_PVS_AIR_ID].as_slice().borrow();

    cols.child_verifier_pvs = *child_verifier_pvs;
    cols.child_vm_pvs = *child_vm_pvs;
    cols.program_commit_hash = record.program_commit_hash;
    cols.initial_root_hash = record.initial_root_hash;
    cols.initial_pc_hash = record.initial_pc_hash;
    cols.intermediate_exe_commit = record.intermediate_exe_commit;
    cols.intermediate_vk_states = record.intermediate_vk_states;
    cols.app_exe_commit = record.app_exe_commit;
    cols.app_vk_commit = record.app_vk_commit;
    cols.final_transcript_state = final_transcript_state;

    if deferral_enabled {
        let rec_cols: &mut RecursiveDeferredVerifyCols<_> = def_cols_slice.borrow_mut();
        let def_verifier_pvs: &VerifierDefPvs<_> = def_pvs_slice.borrow();
        let def_pvs: &DeferralPvs<_> = proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow();
        rec_cols.child_def_verifier_pvs = *def_verifier_pvs;
        rec_cols.child_def_pvs = *def_pvs;
    }

    let mut public_values = vec![F::ZERO; DeferralCircuitPvs::<u8>::width()];
    let deferral_pvs: &mut DeferralCircuitPvs<F> = public_values.as_mut_slice().borrow_mut();

    // Note final_transcript_state is computed by the verifier sub-circuit,
    // and is thus added to the list of Poseidon2 compress inputs there
    let (left, right) = poseidon2_input_to_digests(final_transcript_state);
    deferral_pvs.input_commit = poseidon2_compress_with_capacity(left, right).0;
    deferral_pvs.output_commit = output_commit;

    AirProvingContext {
        cached_mains: vec![],
        common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
        public_values,
    }
}
