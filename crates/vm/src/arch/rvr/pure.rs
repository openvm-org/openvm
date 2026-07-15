use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_lift::ExtensionRegistry;

use super::{
    bridge::map_rvr_execute_error, compile::CompileError, execute::execute, state::PureState,
    RvrCompiled,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{ExecutionError, Streams, SystemConfig, VmState},
    system::memory::online::GuestMemory,
};

/// `suspended` is `false` for unlimited runs.
pub struct RvrPureResult {
    pub state: PureState,
    pub suspended: bool,
}

pub struct RvrPureInstance<'a, F: PrimeField32> {
    pub(crate) system_config: &'a SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) compiled: RvrCompiled,
    pub(crate) extensions: ExtensionRegistry<F>,
}

static_assertions::assert_impl_all!(RvrPureInstance<'static, p3_baby_bear::BabyBear>: Send, Sync);

impl<'a, F> RvrPureInstance<'a, F>
where
    F: PrimeField32,
{
    pub fn create_initial_vm_state(
        &self,
        inputs: impl Into<Streams<F>>,
    ) -> VmState<F, GuestMemory> {
        VmState::initial(
            self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        )
    }

    pub fn execute(
        &self,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_from_state(vm_state, num_insns)
    }

    pub fn execute_from_state(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Pure);
        #[allow(unused_variables)]
        let result = tracing::info_span!("execute_pure")
            .in_scope(|| execute(&self.compiled, &self.extensions, &mut vm_state, num_insns))
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            let insns = result.state.instret;
            metrics.record(insns);
        }
        Ok(vm_state)
    }

    /// Persist the compiled shared library into `dir`. Returns the path to
    /// the copied artifact. No compatibility validation is performed here.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        let dest_lib = self.compiled.lib_file_name_with_suffix("pure")?;
        self.compiled.save_artifact(&dir.join(dest_lib))
    }

    /// Persist generated C sources for inspection.
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.compiled.save_generated_sources(dir)
    }
}
