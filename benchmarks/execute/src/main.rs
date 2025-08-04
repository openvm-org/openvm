use eyre::Result;
use openvm_circuit::arch::{instructions::exe::VmExe, ContinuationVmProof, VirtualMachine};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::prover::hal::DeviceDataTransporter,
    p3_baby_bear::BabyBear,
};

use openvm_continuations::{
    verifier::common::types::VmVerifierPvs, verifier::leaf::types::LeafVmVerifierInput, SC,
};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_sdk::config::{DEFAULT_LEAF_LOG_BLOWUP, SBOX_SIZE};

fn main() -> Result<()> {
    let app_proof_bytes = include_bytes!("../../fixtures/kitchen-sink.app.proof");
    let app_proof: ContinuationVmProof<SC> = bitcode::deserialize(app_proof_bytes)?;

    let leaf_exe_bytes = include_bytes!("../../fixtures/kitchen-sink.leaf.exe");
    let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(leaf_exe_bytes)?;

    let leaf_pk_bytes = include_bytes!("../../fixtures/kitchen-sink.leaf.pk");
    let leaf_pk = bitcode::deserialize(leaf_pk_bytes)?;

    let leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proof, 2);
    let leaf_input = leaf_inputs.first().expect("No leaf input available");

    let config = NativeConfig::aggregation(
        VmVerifierPvs::<u8>::width(),
        SBOX_SIZE.min(FriParameters::standard_fast().max_constraint_degree()),
    );
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_LEAF_LOG_BLOWUP);
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let d_pk = engine.device().transport_pk_to_device(&leaf_pk);
    let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk)?;
    let input_stream = leaf_input.write_to_stream();
    let state = vm.create_initial_state(&leaf_exe, input_stream);

    let _out = vm
        .execute_preflight(&leaf_exe, state, None, NATIVE_MAX_TRACE_HEIGHTS)
        .expect("Failed to execute preflight");

    println!("Successfully executed preflight");

    Ok(())
}
