use std::sync::Arc;

use openvm_circuit::arch::{
    VirtualMachine, VirtualMachineError, VmBuilder, VmInstance, instructions::exe::VmExe,
};
use openvm_stark_backend::config::Val;
use stark_backend_v2::{StarkWhirEngine, prover::DeviceDataTransporterV2};

use crate::prover::vm::types::VmProvingKey;

pub mod types;

pub fn new_local_prover<E, VB>(
    vm_builder: VB,
    vm_pk: &VmProvingKey<VB::VmConfig>,
    exe: Arc<VmExe<Val<E::SC>>>,
) -> Result<VmInstance<E, VB>, VirtualMachineError>
where
    E: StarkWhirEngine,
    VB: VmBuilder<E>,
{
    let engine = E::new(vm_pk.get_params());
    let d_pk = engine.device().transport_pk_to_device(&vm_pk.vm_pk);
    let vm = VirtualMachine::new(engine, vm_builder, vm_pk.vm_config.clone(), d_pk)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let instance = VmInstance::new(vm, exe, cached_program_trace)?;
    Ok(instance)
}
