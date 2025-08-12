use openvm_circuit::{
    arch::{VirtualMachine, VirtualMachineError, VmBuilder, VmInstance},
    system::program::trace::VmCommittedExe,
};
use openvm_stark_backend::prover::hal::DeviceDataTransporter;
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::prover::vm::types::VmProvingKey;

pub mod types;

pub fn new_local_prover<E, VB>(
    vm_builder: VB,
    vm_pk: &VmProvingKey<E::SC, VB::VmConfig>,
    committed_exe: &VmCommittedExe<E::SC>,
) -> Result<VmInstance<E, VB>, VirtualMachineError>
where
    E: StarkFriEngine,
    VB: VmBuilder<E>,
{
    let engine = E::new(vm_pk.fri_params);
    let d_pk = engine.device().transport_pk_to_device(&vm_pk.vm_pk);
    let vm = VirtualMachine::new(engine, vm_builder, vm_pk.vm_config.clone(), d_pk)?;
    let cached_program_trace = vm.transport_committed_exe_to_device(committed_exe);
    // TODO[jpw]: remove this clone
    let instance = VmInstance::new(vm, committed_exe.exe.clone(), cached_program_trace)?;
    Ok(instance)
}
