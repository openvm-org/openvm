use std::path::{Path, PathBuf};

use rvr_openvm_lift::RvrRuntimeExtension;
use rvr_state::ExecutionStatus;

use super::{
    bridge::map_rvr_execute_error,
    compile::CompileError,
    execute::{
        execute_pure, execute_pure_with_instret_limit, execute_pure_with_instret_tracking,
        TrackedExecutionResult,
    },
    GuestProfileConfig, RvrCompiled, RvrInitialImage,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{ExecutionError, ExecutionOutcome, Streams, SystemConfig, VmState},
    system::memory::online::GuestMemory,
};

struct RvrPureInstanceInner<'a> {
    system_config: &'a SystemConfig,
    initial_image: RvrInitialImage,
    compiled: RvrCompiled,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
}

impl<'a> RvrPureInstanceInner<'a> {
    fn new(
        system_config: &'a SystemConfig,
        initial_image: RvrInitialImage,
        compiled: RvrCompiled,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    ) -> Self {
        Self {
            system_config,
            initial_image,
            compiled,
            runtime_hooks,
        }
    }

    fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.initial_image
            .create_vm_state(self.system_config, inputs)
    }

    fn execute_pure_from_state(
        &self,
        mut vm_state: VmState<GuestMemory>,
        profile: Option<&GuestProfileConfig>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        tracing::info_span!("execute_pure")
            .in_scope(|| execute_pure(&self.compiled, &self.runtime_hooks, &mut vm_state, profile))
            .map_err(map_rvr_execute_error)?;
        Ok(vm_state)
    }

    fn execute_tracked_from_state(
        &self,
        mut vm_state: VmState<GuestMemory>,
    ) -> Result<(VmState<GuestMemory>, u64), ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Pure);
        let retired = tracing::info_span!("execute_pure")
            .in_scope(|| {
                execute_pure_with_instret_tracking(
                    &self.compiled,
                    &self.runtime_hooks,
                    &mut vm_state,
                )
            })
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        metrics.record(retired);
        Ok((vm_state, retired))
    }

    fn execute_tracked_from_state_for(
        &self,
        mut vm_state: VmState<GuestMemory>,
        num_insns: u64,
    ) -> Result<(VmState<GuestMemory>, TrackedExecutionResult), ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Pure);
        let result = tracing::info_span!("execute_pure")
            .in_scope(|| {
                execute_pure_with_instret_limit(
                    &self.compiled,
                    &self.runtime_hooks,
                    &mut vm_state,
                    num_insns,
                )
            })
            .map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        metrics.record(result.retired);
        Ok((vm_state, result))
    }

    fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        let dest_lib = self.compiled.artifact_file_name()?;
        self.compiled.save_artifact(&dir.join(dest_lib))
    }

    fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.compiled.save_generated_sources(dir)
    }
}

/// Compiled pure RVR execution without instruction-retirement tracking.
pub struct RvrPureInstance<'a> {
    inner: RvrPureInstanceInner<'a>,
}

/// Compiled pure RVR execution with instruction-retirement tracking.
pub struct RvrPureWithInstretTrackingInstance<'a> {
    inner: RvrPureInstanceInner<'a>,
}

static_assertions::assert_impl_all!(RvrPureInstance<'static>: Send, Sync);
static_assertions::assert_impl_all!(RvrPureWithInstretTrackingInstance<'static>: Send, Sync);

/// State and retired-instruction count produced by tracked pure execution.
pub struct RvrTrackedExecution {
    /// VM state after this execution call.
    pub state: VmState<GuestMemory>,
    /// Instructions retired during this execution call, not across prior resumptions.
    pub retired: u64,
}

/// Result of bounded pure execution with instret tracking.
pub type RvrTrackedExecutionOutcome = ExecutionOutcome<RvrTrackedExecution>;

impl<'a> RvrPureInstance<'a> {
    pub(crate) fn new(
        system_config: &'a SystemConfig,
        initial_image: RvrInitialImage,
        compiled: RvrCompiled,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    ) -> Self {
        Self {
            inner: RvrPureInstanceInner::new(system_config, initial_image, compiled, runtime_hooks),
        }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner.create_initial_vm_state(inputs)
    }

    /// Whether this artifact can be used for guest sampling.
    pub const fn is_profile_compatible(&self) -> bool {
        self.inner.compiled.is_profile_compatible()
    }

    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_from_state(vm_state)
    }

    pub fn execute_from_state(
        &self,
        vm_state: VmState<GuestMemory>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        self.inner.execute_pure_from_state(vm_state, None)
    }

    pub fn execute_profiled(
        &self,
        inputs: impl Into<Streams>,
        profile: &GuestProfileConfig,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_from_state_profiled(vm_state, profile)
    }

    pub fn execute_from_state_profiled(
        &self,
        vm_state: VmState<GuestMemory>,
        profile: &GuestProfileConfig,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        self.inner.execute_pure_from_state(vm_state, Some(profile))
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.inner.save(dir)
    }

    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.inner.save_generated_sources(dir)
    }
}

impl<'a> RvrPureWithInstretTrackingInstance<'a> {
    pub(crate) fn new(
        system_config: &'a SystemConfig,
        initial_image: RvrInitialImage,
        compiled: RvrCompiled,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    ) -> Self {
        Self {
            inner: RvrPureInstanceInner::new(system_config, initial_image, compiled, runtime_hooks),
        }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner.create_initial_vm_state(inputs)
    }

    /// Whether this artifact can be used for guest sampling.
    pub const fn is_profile_compatible(&self) -> bool {
        self.inner.compiled.is_profile_compatible()
    }

    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
    ) -> Result<RvrTrackedExecution, ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_from_state(vm_state)
    }

    pub fn execute_from_state(
        &self,
        vm_state: VmState<GuestMemory>,
    ) -> Result<RvrTrackedExecution, ExecutionError> {
        let (state, retired) = self.inner.execute_tracked_from_state(vm_state)?;
        Ok(RvrTrackedExecution { state, retired })
    }

    /// Execute for at most `num_insns`, stopping at a basic-block boundary.
    pub fn execute_for(
        &self,
        inputs: impl Into<Streams>,
        num_insns: u64,
    ) -> Result<RvrTrackedExecutionOutcome, ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_from_state_for(vm_state, num_insns)
    }

    /// Continue for at most `num_insns`, stopping at a basic-block boundary.
    pub fn execute_from_state_for(
        &self,
        vm_state: VmState<GuestMemory>,
        num_insns: u64,
    ) -> Result<RvrTrackedExecutionOutcome, ExecutionError> {
        let (state, result) = self
            .inner
            .execute_tracked_from_state_for(vm_state, num_insns)?;
        let execution = RvrTrackedExecution {
            state,
            retired: result.retired,
        };
        Ok(match result.status {
            ExecutionStatus::Terminated => RvrTrackedExecutionOutcome::Terminated(execution),
            ExecutionStatus::Suspended => RvrTrackedExecutionOutcome::Suspended(execution),
            _ => unreachable!("successful tracked execution must terminate or suspend"),
        })
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.inner.save(dir)
    }

    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.inner.save_generated_sources(dir)
    }
}
