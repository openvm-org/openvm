use std::sync::Arc;

use openvm_circuit::arch::{
    instructions::exe::VmExe, VirtualMachine, VirtualMachineError, VmBuilder, VmInstance,
};
use openvm_stark_backend::{config::Val, prover::hal::DeviceDataTransporter};
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::prover::vm::types::VmProvingKey;

pub mod types;

pub fn new_local_prover<E, VB>(
    vm_builder: VB,
    vm_pk: &VmProvingKey<E::SC, VB::VmConfig>,
    exe: Arc<VmExe<Val<E::SC>>>,
) -> Result<VmInstance<E, VB>, VirtualMachineError>
where
    E: StarkFriEngine,
    VB: VmBuilder<E>,
{
    let engine = E::new(vm_pk.fri_params);
    let d_pk = engine.device().transport_pk_to_device(&vm_pk.vm_pk);
    let vm = VirtualMachine::new(engine, vm_builder, vm_pk.vm_config.clone(), d_pk)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let instance = VmInstance::new(vm, exe, cached_program_trace)?;
    Ok(instance)
}
