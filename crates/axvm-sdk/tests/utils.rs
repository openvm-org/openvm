use ax_stark_sdk::{ax_stark_backend::config::StarkGenericConfig, config::FriParameters};
use axvm_circuit::{prover::types::VmProvingKey, system::program::trace::AxVmCommittedExe};
use axvm_sdk::{config::AggConfig, keygen::AggProvingKey};
use serde::Serialize;

pub fn assert_agg_config_eq(a: &AggConfig, b: &AggConfig) {
    assert_eq!(a.max_num_user_public_values, b.max_num_user_public_values);
    assert_eq!(a.compiler_options.word_size, b.compiler_options.word_size);
    assert_fri_params_eq(&a.leaf_fri_params, &b.leaf_fri_params);
    assert_fri_params_eq(&a.internal_fri_params, &b.internal_fri_params);
    assert_fri_params_eq(&a.root_fri_params, &b.root_fri_params);
    assert_serialize_eq(a, b);
}

pub fn assert_agg_pk_eq(a: &AggProvingKey, b: &AggProvingKey) {
    assert_vm_pk_eq(&a.leaf_vm_pk, &b.leaf_vm_pk);
    assert_vm_pk_eq(&a.internal_vm_pk, &b.internal_vm_pk);
    assert_vm_pk_eq(&a.root_verifier_pk.vm_pk, &b.root_verifier_pk.vm_pk);
    assert_committed_exe_eq(&a.internal_committed_exe, &b.internal_committed_exe);
    assert_committed_exe_eq(
        &a.root_verifier_pk.root_committed_exe,
        &b.root_verifier_pk.root_committed_exe,
    );
    assert_eq!(
        a.root_verifier_pk.air_heights,
        b.root_verifier_pk.air_heights,
    );
    assert_serialize_eq(a, b);
}

fn assert_fri_params_eq(a: &FriParameters, b: &FriParameters) {
    assert_eq!(a.log_blowup, b.log_blowup);
    assert_eq!(a.num_queries, b.num_queries);
    assert_eq!(a.proof_of_work_bits, b.proof_of_work_bits);
}

fn assert_vm_pk_eq<SC: StarkGenericConfig, VC>(a: &VmProvingKey<SC, VC>, b: &VmProvingKey<SC, VC>) {
    assert_fri_params_eq(&a.fri_params, &b.fri_params);
    assert_eq!(a.vm_pk.max_constraint_degree, b.vm_pk.max_constraint_degree);
    assert_eq!(a.vm_pk.per_air.len(), b.vm_pk.per_air.len());
    for (a_pk, b_pk) in a.vm_pk.per_air.iter().zip(b.vm_pk.per_air.iter()) {
        assert_eq!(a_pk.air_name, b_pk.air_name);
    }
}

fn assert_committed_exe_eq<SC: StarkGenericConfig>(
    a: &AxVmCommittedExe<SC>,
    b: &AxVmCommittedExe<SC>,
) {
    for (a_inst, b_inst) in a
        .exe
        .program
        .instructions()
        .iter()
        .zip(b.exe.program.instructions().iter())
    {
        assert_eq!(a_inst, b_inst);
    }
    assert_eq!(a.exe.pc_start, b.exe.pc_start);
    assert_eq!(a.exe.init_memory, b.exe.init_memory);
    assert_eq!(a.committed_program.raw_data, b.committed_program.raw_data);
}

fn assert_serialize_eq<T: Serialize>(a: &T, b: &T) {
    let a_bytes = bincode::serialize(a).unwrap();
    let b_bytes = bincode::serialize(b).unwrap();
    assert_eq!(a_bytes, b_bytes);
}
