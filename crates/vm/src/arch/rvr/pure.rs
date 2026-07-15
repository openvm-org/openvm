use std::path::{Path, PathBuf};

use rvr_openvm_lift::RvrRuntimeExtension;

use super::{
    bridge::map_rvr_execute_error, compile::CompileError, execute::execute, state::PureState,
    RvrCompiled, RvrInitialImage,
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

pub struct RvrPureInstance<'a> {
    pub(crate) system_config: &'a SystemConfig,
    pub(crate) initial_image: RvrInitialImage,
    pub(crate) compiled: RvrCompiled,
    pub(crate) runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
}

static_assertions::assert_impl_all!(RvrPureInstance<'static>: Send, Sync);

impl RvrPureInstance<'_> {
    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.initial_image
            .create_vm_state(self.system_config, inputs)
    }

    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
        num_insns: Option<u64>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_from_state(vm_state, num_insns)
    }

    pub fn execute_from_state(
        &self,
        mut vm_state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Pure);
        #[allow(unused_variables)]
        let result = tracing::info_span!("execute_pure")
            .in_scope(|| {
                execute(
                    &self.compiled,
                    &self.runtime_hooks,
                    &mut vm_state,
                    num_insns,
                )
            })
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
