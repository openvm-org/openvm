#[cfg(feature = "rvr")]
use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(feature = "rvr")]
use eyre::{Context, Result};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{execution_mode::ExecutionCtx, InterpretedInstance};
#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::{MeteredCtxConfig, SegmentationConfig},
    rvr::{
        CompileError, GuestProfileConfig, RvrPureInstance, RvrPureWithInstretTrackingInstance,
        RvrTrackedExecution, RvrTrackedExecutionOutcome,
    },
};
use openvm_circuit::{
    arch::{
        execution_mode::{MeteredCostCtx, MeteredCtx},
        ExecutionError, Streams, VmState,
    },
    system::memory::online::GuestMemory,
};
#[cfg(feature = "rvr")]
use serde::{Deserialize, Serialize};

cfg_if::cfg_if! {
    if #[cfg(feature = "rvr")] {
        use openvm_circuit::arch::rvr::{
            RvrMeteredCostInstance, RvrMeteredInstance,
        };
        type PureInstance<'a> = RvrPureInstance<'a>;
        pub type MeteredInstance<'a> = RvrMeteredInstance<'a>;
        pub type MeteredCostInstance<'a> = RvrMeteredCostInstance<'a>;
    } else {
        type PureInstance<'a> = InterpretedInstance<'a, ExecutionCtx>;
        pub type MeteredInstance<'a> = InterpretedInstance<'a, MeteredCtx>;
        pub type MeteredCostInstance<'a> = InterpretedInstance<'a, MeteredCostCtx>;
    }
}

/// Compiled unlimited pure execution.
///
/// With RVR, this artifact has no instruction-retirement tracking or suspension path.
pub struct CompiledExePure<'a> {
    instance: PureInstance<'a>,
}

/// Compiled pure RVR execution with instruction-retirement tracking.
#[cfg(feature = "rvr")]
pub struct CompiledExePureWithInstretTracking<'a> {
    instance: RvrPureWithInstretTrackingInstance<'a>,
}

impl<'a> CompiledExePure<'a> {
    pub(crate) fn new(instance: PureInstance<'a>) -> Self {
        Self { instance }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.instance.create_initial_vm_state(inputs)
    }

    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        self.instance.execute(inputs)
    }

    pub fn execute_from_state(
        &self,
        state: VmState<GuestMemory>,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        self.instance.execute_from_state(state)
    }

    #[cfg(feature = "rvr")]
    /// Whether this artifact can be passed to the profiled execution APIs.
    pub const fn is_profile_compatible(&self) -> bool {
        self.instance.is_profile_compatible()
    }

    #[cfg(feature = "rvr")]
    pub fn execute_profiled(
        &self,
        inputs: impl Into<Streams>,
        profile: &GuestProfileConfig,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        self.instance.execute_profiled(inputs, profile)
    }

    #[cfg(feature = "rvr")]
    pub fn execute_from_state_profiled(
        &self,
        state: VmState<GuestMemory>,
        profile: &GuestProfileConfig,
    ) -> Result<VmState<GuestMemory>, ExecutionError> {
        self.instance.execute_from_state_profiled(state, profile)
    }

    #[cfg(feature = "rvr")]
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.instance.save(dir)
    }

    #[cfg(feature = "rvr")]
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.instance.save_generated_sources(dir)
    }
}

#[cfg(feature = "rvr")]
impl<'a> CompiledExePureWithInstretTracking<'a> {
    pub(crate) fn new(instance: RvrPureWithInstretTrackingInstance<'a>) -> Self {
        Self { instance }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.instance.create_initial_vm_state(inputs)
    }

    /// Execute for at most `num_insns`, stopping before a basic block that
    /// would exceed the limit.
    pub fn execute_for(
        &self,
        inputs: impl Into<Streams>,
        num_insns: u64,
    ) -> Result<RvrTrackedExecutionOutcome, ExecutionError> {
        self.instance.execute_for(inputs, num_insns)
    }

    /// Continue from `state` for at most `num_insns`, stopping before a basic
    /// block that would exceed the limit.
    pub fn execute_from_state_for(
        &self,
        state: VmState<GuestMemory>,
        num_insns: u64,
    ) -> Result<RvrTrackedExecutionOutcome, ExecutionError> {
        self.instance.execute_from_state_for(state, num_insns)
    }

    /// Execute until successful termination and return the instructions retired by this call.
    pub fn execute(
        &self,
        inputs: impl Into<Streams>,
    ) -> Result<RvrTrackedExecution, ExecutionError> {
        self.instance.execute(inputs)
    }

    /// Continue until successful termination and return the instructions retired by this call.
    pub fn execute_from_state(
        &self,
        state: VmState<GuestMemory>,
    ) -> Result<RvrTrackedExecution, ExecutionError> {
        self.instance.execute_from_state(state)
    }

    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.instance.save(dir)
    }

    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.instance.save_generated_sources(dir)
    }
}

/// Bundles a [`MeteredInstance`] with a precomputed [`MeteredCtx`] so each execution
/// just clones the ctx instead of rebuilding from the proving key.
pub struct CompiledExeMetered<'a> {
    pub instance: MeteredInstance<'a>,
    pub ctx: MeteredCtx,
    #[cfg(feature = "rvr")]
    pub executor_idx_to_air_idx: Vec<usize>,
}

pub struct CompiledExeMeteredCost<'a> {
    pub instance: MeteredCostInstance<'a>,
    pub ctx: MeteredCostCtx,
}

#[cfg(feature = "rvr")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteredArtifactMetadata {
    pub metered_ctx_config: MeteredCtxConfig,
    pub segmentation_config: SegmentationConfig,
    pub executor_idx_to_air_idx: Vec<usize>,
    pub profile_compatible: bool,
}

#[cfg(feature = "rvr")]
pub fn metered_artifact_metadata_path(lib_path: &Path) -> PathBuf {
    lib_path.with_extension("json")
}

#[cfg(feature = "rvr")]
pub fn load_metered_artifact_metadata(lib_path: &Path) -> Result<MeteredArtifactMetadata> {
    let path = metered_artifact_metadata_path(lib_path);
    let data = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("failed to parse {}", path.display()))
}

#[cfg(feature = "rvr")]
impl CompiledExeMetered<'_> {
    /// Whether this artifact can be passed to the profiled execution APIs.
    pub const fn is_profile_compatible(&self) -> bool {
        self.instance.is_profile_compatible()
    }

    /// Persist the compiled shared library and static metering metadata into `dir`.
    /// Returns the path of the copied `.so`/`.dylib`.
    pub fn save(&self, dir: &Path) -> Result<PathBuf> {
        let lib_path = self.instance.save(dir)?;
        let metadata = MeteredArtifactMetadata {
            metered_ctx_config: self.ctx.config.clone(),
            segmentation_config: self.ctx.segmentation_ctx.config().clone(),
            executor_idx_to_air_idx: self.executor_idx_to_air_idx.clone(),
            profile_compatible: self.is_profile_compatible(),
        };
        let metadata_path = metered_artifact_metadata_path(&lib_path);
        let data = serde_json::to_vec_pretty(&metadata)?;
        fs::write(&metadata_path, data)
            .with_context(|| format!("failed to write {}", metadata_path.display()))?;
        Ok(lib_path)
    }

    /// Persist generated C sources for inspection.
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), CompileError> {
        self.instance.save_generated_sources(dir)
    }
}

#[cfg(feature = "rvr")]
impl CompiledExeMeteredCost<'_> {
    /// Whether this artifact can be passed to the profiled execution APIs.
    pub const fn is_profile_compatible(&self) -> bool {
        self.instance.is_profile_compatible()
    }

    /// Persist the compiled shared library into `dir`. Returns the path of
    /// the copied `.so`/`.dylib`. The `MeteredCostCtx` is not persisted — it
    /// is rebuilt on load via
    /// [`Sdk::load_compiled_metered_cost`](crate::Sdk::load_compiled_metered_cost).
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.instance.save(dir)
    }
}
