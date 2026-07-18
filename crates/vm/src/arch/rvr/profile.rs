use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const MAX_SAMPLE_HZ: u32 = 20_000;
const DEFAULT_MAX_SAMPLES: usize = 1 << 18;

/// On-disk representation produced by RVR guest sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestProfileFormat {
    /// Ordered samples with timing, module, and stack-quality metadata.
    Raw,
}

/// Current version of the ordered RVR sampling format.
pub const RAW_GUEST_PROFILE_VERSION: u32 = 3;

/// A native module observed while the guest was executing.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawNativeModule {
    /// Stable display name, usually the module's file name.
    pub name: String,
    /// On-host path used only for immediate local symbolication.
    pub path: String,
    /// Whether this is the generated OpenVM execution artifact.
    pub generated: bool,
}

/// Exact interrupted native instruction pointer.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawNativeFrame {
    /// Index into [`RawGuestProfile::native_modules`], when `dladdr` resolved it.
    pub module_index: Option<u32>,
    /// Module-relative PC when resolved, otherwise the absolute interrupted PC.
    pub pc: u64,
}

/// One RVR sample. Guest PCs are ordered root-to-leaf and contain caller
/// return addresses; `host_pc` is the interrupted instruction pointer,
/// relative to the compiled native artifact.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawGuestProfileSample {
    pub wall_time_ns: u64,
    pub cpu_time_ns: u64,
    /// Exact native leaf captured from the signal ucontext.
    pub native_leaf: Option<RawNativeFrame>,
    /// Exact guest instruction stored immediately before entering host code.
    pub guest_callsite_pc: Option<u64>,
    /// Guest caller return addresses in root-to-leaf order.
    pub guest_return_pcs: Vec<u64>,
    /// Whether the guest frame walk reached the fixed maximum depth.
    pub stack_truncated: bool,
}

/// Versioned, ordered RVR sampling output consumed by execution-profile tools.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawGuestProfile {
    pub version: u32,
    pub requested_sample_hz: u32,
    pub owner_tid: i32,
    pub start_unix_time_ns: u64,
    pub start_wall_time_ns: u64,
    pub end_wall_time_ns: u64,
    pub start_cpu_time_ns: u64,
    pub end_cpu_time_ns: u64,
    pub delivered_samples: u64,
    pub dropped_samples: u64,
    pub timer_overruns: u64,
    pub timer_arm_failures: u64,
    pub clock_failures: u64,
    pub native_modules: Vec<RawNativeModule>,
    pub samples: Vec<RawGuestProfileSample>,
}

/// Explicit configuration for one RVR guest profiling execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuestProfileConfig {
    output: PathBuf,
    sample_hz: u32,
    format: GuestProfileFormat,
    native_artifact_output: Option<PathBuf>,
    max_samples: usize,
}

impl GuestProfileConfig {
    pub fn new(
        output: impl Into<PathBuf>,
        sample_hz: u32,
        format: GuestProfileFormat,
    ) -> Result<Self, String> {
        if !(1..=MAX_SAMPLE_HZ).contains(&sample_hz) {
            return Err(format!(
                "guest profile sampling rate must be in 1..={MAX_SAMPLE_HZ}, got {sample_hz}"
            ));
        }
        Ok(Self {
            output: output.into(),
            sample_hz,
            format,
            native_artifact_output: None,
            max_samples: DEFAULT_MAX_SAMPLES,
        })
    }

    pub fn raw(output: impl Into<PathBuf>, sample_hz: u32) -> Result<Self, String> {
        Self::new(output, sample_hz, GuestProfileFormat::Raw)
    }

    /// Capture ordered samples and preserve the exact native artifact needed
    /// to resolve interrupted host PCs after execution.
    pub fn raw_with_native_artifact(
        output: impl Into<PathBuf>,
        native_artifact_output: impl Into<PathBuf>,
        sample_hz: u32,
    ) -> Result<Self, String> {
        let mut config = Self::raw(output, sample_hz)?;
        config.native_artifact_output = Some(native_artifact_output.into());
        Ok(config)
    }

    pub fn output(&self) -> &Path {
        &self.output
    }

    pub fn sample_hz(&self) -> u32 {
        self.sample_hz
    }

    pub fn format(&self) -> GuestProfileFormat {
        self.format
    }

    pub(crate) fn native_artifact_output(&self) -> Option<&Path> {
        self.native_artifact_output.as_deref()
    }

    /// Set the maximum number of complete samples retained in memory.
    /// Additional samples are counted as dropped rather than silently replacing
    /// earlier timeline data.
    pub fn with_max_samples(mut self, max_samples: usize) -> Result<Self, String> {
        if max_samples == 0 {
            return Err("guest profile max_samples must be nonzero".to_string());
        }
        self.max_samples = max_samples;
        Ok(self)
    }

    pub(crate) fn max_samples(&self) -> usize {
        self.max_samples
    }
}

#[cfg(test)]
mod tests {
    use super::{GuestProfileConfig, MAX_SAMPLE_HZ};

    #[test]
    fn validates_sampling_rate_at_both_boundaries() {
        assert!(GuestProfileConfig::raw("profile.raw", 0).is_err());
        assert!(GuestProfileConfig::raw("profile.raw", 1).is_ok());
        assert!(GuestProfileConfig::raw("profile.raw", MAX_SAMPLE_HZ).is_ok());
        assert!(GuestProfileConfig::raw("profile.raw", MAX_SAMPLE_HZ + 1).is_err());
    }
}
