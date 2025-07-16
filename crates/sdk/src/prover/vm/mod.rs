use openvm_circuit::{
    arch::{VirtualMachine, VirtualMachineError, VmLocalProver, VmProverConfig},
    system::program::trace::VmCommittedExe,
};
use openvm_stark_backend::prover::hal::DeviceDataTransporter;
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::prover::vm::types::VmProvingKey;

pub mod types;

pub fn new_local_prover<E, VC>(
    app_vm_pk: &VmProvingKey<E::SC, VC>,
    app_committed_exe: &VmCommittedExe<E::SC>,
) -> Result<VmLocalProver<E, VC>, VirtualMachineError>
where
    E: StarkFriEngine,
    VC: VmProverConfig<E>,
{
    let engine = E::new(app_vm_pk.fri_params);
    let d_pk = engine.device().transport_pk_to_device(&app_vm_pk.vm_pk);
    let vm = VirtualMachine::new(engine, app_vm_pk.vm_config.clone(), d_pk)?;
    let cached_program_trace = vm.transport_committed_exe_to_device(&app_committed_exe);
    // TODO[jpw]: remove this clone
    Ok(VmLocalProver::new(
        vm,
        app_committed_exe.exe.clone(),
        cached_program_trace,
    ))
}
