#[cfg(feature = "rvr")]
use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(feature = "rvr")]
use eyre::{Context, Result};
use openvm_circuit::arch::execution_mode::{MeteredCostCtx, MeteredCtx};
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::{execution_mode::ExecutionCtx, InterpretedInstance};
#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    execution_mode::{MeteredCtxConfig, SegmentationConfig},
    rvr::CompileError,
};
#[cfg(feature = "rvr")]
use serde::{Deserialize, Serialize};

use crate::F;

cfg_if::cfg_if! {
    if #[cfg(feature = "rvr")] {
        use openvm_circuit::arch::rvr::{
            RvrMeteredCostInstance, RvrMeteredInstance, RvrPureInstance,
        };
        pub type CompiledExePure<'a, F> = RvrPureInstance<'a, F>;
        pub type MeteredInstance<'a, F> = RvrMeteredInstance<'a, F>;
        pub type MeteredCostInstance<'a, F> = RvrMeteredCostInstance<'a, F>;
    } else if #[cfg(feature = "aot")] {
        use openvm_circuit::arch::AotInstance;
        pub type CompiledExePure<'a, F> = AotInstance<'a, F, ExecutionCtx>;
        pub type MeteredInstance<'a, F> = AotInstance<'a, F, MeteredCtx>;
        // AOT has no dedicated metered-cost backend; fall back to the interpreter.
        pub type MeteredCostInstance<'a, F> = InterpretedInstance<'a, F, MeteredCostCtx>;
    } else {
        pub type CompiledExePure<'a, F> = InterpretedInstance<'a, F, ExecutionCtx>;
        pub type MeteredInstance<'a, F> = InterpretedInstance<'a, F, MeteredCtx>;
        pub type MeteredCostInstance<'a, F> = InterpretedInstance<'a, F, MeteredCostCtx>;
    }
}

/// Bundles a [`MeteredInstance`] with a precomputed [`MeteredCtx`] so each execution
/// just clones the ctx instead of rebuilding from the proving key.
pub struct CompiledExeMetered<'a> {
    pub instance: MeteredInstance<'a, F>,
    pub ctx: MeteredCtx,
    #[cfg(feature = "rvr")]
    pub executor_idx_to_air_idx: Vec<usize>,
}

pub struct CompiledExeMeteredCost<'a> {
    pub instance: MeteredCostInstance<'a, F>,
    pub ctx: MeteredCostCtx,
}

#[cfg(feature = "rvr")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeteredArtifactMetadata {
    pub metered_ctx_config: MeteredCtxConfig,
    pub segmentation_config: SegmentationConfig,
    pub executor_idx_to_air_idx: Vec<usize>,
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
    /// Persist the compiled shared library and static metering metadata into `dir`.
    /// Returns the path of the copied `.so`/`.dylib`.
    pub fn save(&self, dir: &Path) -> Result<PathBuf> {
        let lib_path = self.instance.save(dir)?;
        let metadata = MeteredArtifactMetadata {
            metered_ctx_config: self.ctx.config.clone(),
            segmentation_config: self.ctx.segmentation_ctx.config().clone(),
            executor_idx_to_air_idx: self.executor_idx_to_air_idx.clone(),
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
    /// Persist the compiled shared library into `dir`. Returns the path of
    /// the copied `.so`/`.dylib`. The `MeteredCostCtx` is not persisted — it
    /// is rebuilt on load via
    /// [`Sdk::load_compiled_metered_cost`](crate::Sdk::load_compiled_metered_cost).
    pub fn save(&self, dir: &Path) -> Result<PathBuf, CompileError> {
        self.instance.save(dir)
    }
}
