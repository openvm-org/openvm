use std::sync::Arc;

use continuations_v2::aggregation::{
    AggregationCircuit, MAX_NUM_PROOFS, NonRootAggregationCircuit,
};
use eyre::Result;
use openvm_circuit::{
    arch::{MemoryConfig, VirtualMachine, VmBuilder, VmCircuitConfig, instructions::exe::VmExe},
    system::{memory::dimensions::MemoryDimensions, program::trace::VmCommittedExe},
};
use recursion_circuit::system::VerifierSubCircuit;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, Digest, F, SC, StarkEngineV2, SystemParams,
    poseidon2::sponge::DuplexSponge, prover::DeviceDataTransporterV2,
};

/// Baseline artifacts for a specific VM and fixed executable that are used to verify a final
/// (i.e. internal-recursive) VM STARK proof
pub struct VerificationBaseline {
    /// Commit to the app exe (i.e. hash of the program commit, initial memory merkle root,
    /// and initial program counter)
    pub app_exe_commit: Digest,
    /// VM memory metadata used to verify the user public values merkle proof
    pub memory_dimensions: MemoryDimensions,
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpresionAir, which is
    /// derived from the app_vk
    pub leaf_commit: Digest,
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpresionAir,
    /// which derived from the leaf_vk
    pub internal_for_leaf_commit: Digest,
    /// Cached trace commit of the internal-recursive verifier circuit's SymbolicExpresionAir,
    /// which derived from the internal_for_leaf_vk
    pub internal_recursive_commit: Digest,
}

// TODO[INT-5314]: move to SDK (this should be part of prover)
pub fn generate_baseline<VB: VmBuilder<BabyBearPoseidon2CpuEngineV2> + Default>(
    vm_config: VB::VmConfig,
    exe: Arc<VmExe<F>>,
    memory_config: &MemoryConfig,
    app_params: SystemParams,
    leaf_params: SystemParams,
    internal_params: SystemParams,
) -> Result<VerificationBaseline> {
    let app_engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(app_params);
    let (app_pk, app_vk) =
        app_engine.keygen(&vm_config.create_airs()?.into_airs().collect::<Vec<_>>());
    let d_pk = app_engine.device().transport_pk_to_device(&app_pk);
    let vm = VirtualMachine::new(app_engine, VB::default(), vm_config, d_pk)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let app_commit = VmCommittedExe::<SC>::compute_exe_commit(
        &cached_program_trace.commitment.into(),
        &exe,
        memory_config,
    )
    .into();
    let app_vk = Arc::new(app_vk);

    let leaf_engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(leaf_params);
    let leaf_circuit = NonRootAggregationCircuit::new(Arc::new(
        VerifierSubCircuit::<MAX_NUM_PROOFS>::new_with_set_continuations(app_vk.clone(), true),
    ));
    let leaf_commit = leaf_circuit
        .verifier_circuit
        .commit_child_vk(&leaf_engine, &app_vk);
    let leaf_vk = Arc::new(leaf_engine.keygen(&leaf_circuit.airs()).1);

    let internal_engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(internal_params);
    let internal_for_leaf_circuit = NonRootAggregationCircuit::new(Arc::new(
        VerifierSubCircuit::<MAX_NUM_PROOFS>::new_with_set_continuations(leaf_vk.clone(), true),
    ));
    let internal_for_leaf_commit = internal_for_leaf_circuit
        .verifier_circuit
        .commit_child_vk(&internal_engine, &leaf_vk);
    let internal_for_leaf_vk =
        Arc::new(internal_engine.keygen(&internal_for_leaf_circuit.airs()).1);

    let internal_recursive_circuit = NonRootAggregationCircuit::new(Arc::new(
        VerifierSubCircuit::<MAX_NUM_PROOFS>::new_with_set_continuations(
            internal_for_leaf_vk.clone(),
            true,
        ),
    ));
    let internal_recursive_commit = internal_recursive_circuit
        .verifier_circuit
        .commit_child_vk(&internal_engine, &internal_for_leaf_vk);

    Ok(VerificationBaseline {
        app_exe_commit: app_commit,
        leaf_commit: leaf_commit.commitment,
        internal_for_leaf_commit: internal_for_leaf_commit.commitment,
        internal_recursive_commit: internal_recursive_commit.commitment,
        memory_dimensions: memory_config.memory_dimensions(),
    })
}
