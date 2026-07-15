use openvm_instructions::exe::{SparseMemoryImage, VmExe};

use crate::{
    arch::{Streams, SystemConfig, VmState},
    system::memory::online::GuestMemory,
};

/// State needed to initialize an rvr execution.
#[derive(Clone, Debug)]
pub struct RvrInitialImage {
    pc_start: u32,
    init_memory: SparseMemoryImage,
}

impl<F> From<&VmExe<F>> for RvrInitialImage {
    fn from(exe: &VmExe<F>) -> Self {
        Self {
            pc_start: exe.pc_start,
            init_memory: exe.init_memory.clone(),
        }
    }
}

impl RvrInitialImage {
    pub(crate) fn create_vm_state(
        &self,
        system_config: &SystemConfig,
        inputs: impl Into<Streams>,
    ) -> VmState<GuestMemory> {
        VmState::initial(system_config, &self.init_memory, self.pc_start, inputs)
    }
}
