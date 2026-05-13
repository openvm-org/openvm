use std::sync::Arc;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::ExtensionRegistry;

use super::{bridge::map_rvr_execute_error, execute::execute, RvrCompiled};
use crate::{
    arch::{ExecutionError, Streams, SystemConfig, VmState},
    system::memory::online::GuestMemory,
};

pub struct RvrPureInstance<F: PrimeField32> {
    pub(crate) system_config: SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) compiled: RvrCompiled,
    pub(crate) extensions: ExtensionRegistry<F>,
}

impl<F> RvrPureInstance<F>
where
    F: PrimeField32,
{
    pub fn execute(
        &self,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let vm_state = VmState::initial(
            &self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        );
        self.execute_from_state(vm_state, num_insns)
    }

    pub fn execute_from_state(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        #[allow(unused_variables)]
        let result = tracing::info_span!("execute_e1")
            .in_scope(|| execute(&self.compiled, &self.extensions, &mut vm_state, num_insns))
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed();
            let insns = result.state.instret;
            tracing::info!("instructions_executed={insns}");
            metrics::counter!("execute_e1_insns").absolute(insns);
            metrics::gauge!("execute_e1_insn_mi/s").set(insns as f64 / elapsed.as_micros() as f64);
        }
        Ok(vm_state)
    }
}
