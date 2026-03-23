use std::sync::Arc;

use openvm_circuit::arch::{
    instructions::{instruction::Instruction, program::Program, LocalOpcode, SystemOpcode},
    Executor, MeteredExecutor, PreflightExecutor, VmBuilder, VmExecutionConfig,
};
use openvm_continuations::RootSC;
use openvm_stark_backend::{p3_field::PrimeField32, proof::Proof, StarkEngine, Val};

use crate::{
    prover::{
        vm::types::VmProvingKey, AggProver, DeferralPathProver, EvmProver, RootProver,
    },
    StdIn, F, SC,
};

/// Generate a dummy root proof for keygen purposes.
///
/// Runs a trivial TERMINATE-only program through the full EVM prover pipeline
/// (app → aggregation → root) and returns the resulting root proof.
pub fn generate_dummy_root_proof<E, VB>(
    vm_builder: VB,
    app_vm_pk: &VmProvingKey<VB::VmConfig>,
    agg_prover: Arc<AggProver>,
    def_path_prover: Option<Arc<DeferralPathProver>>,
    root_prover: Arc<RootProver>,
) -> Proof<RootSC>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    Val<SC>: PrimeField32,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
{
    let dummy_program = Program::<F>::from_instructions(&[Instruction::from_isize(
        SystemOpcode::TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    )]);
    let dummy_exe = Arc::new(openvm_circuit::arch::instructions::exe::VmExe::new(
        dummy_program,
    ));

    let mut evm_prover = EvmProver::<E, _>::new(
        vm_builder,
        app_vm_pk,
        dummy_exe,
        agg_prover,
        def_path_prover,
        root_prover,
    )
    .expect("Failed to create dummy EVM prover");

    evm_prover
        .prove(StdIn::default(), &[])
        .expect("Failed to generate dummy root proof")
}
