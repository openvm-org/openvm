use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const MAX_SAMPLE_HZ: u32 = 1_000_000;

/// On-disk representation produced by RVR guest sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestProfileFormat {
    /// Collapsed stacks with sample counts, suitable for flamegraph tools.
    Folded,
    /// One ordered, semicolon-separated stack per sample.
    Raw,
}

/// Current version of the ordered RVR sampling format.
pub const RAW_GUEST_PROFILE_VERSION: u32 = 2;

/// One RVR sample. Guest PCs are ordered root-to-leaf and contain caller
/// return addresses; `host_pc` is the interrupted instruction pointer,
/// relative to the compiled native artifact.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawGuestProfileSample {
    pub wall_time_ns: u64,
    pub cpu_time_ns: u64,
    pub host_pc: Option<u64>,
    pub guest_pcs: Vec<u64>,
}

/// Versioned, ordered RVR sampling output consumed by execution-profile tools.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawGuestProfile {
    pub version: u32,
    pub samples: Vec<RawGuestProfileSample>,
}

/// Explicit configuration for one RVR guest profiling execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuestProfileConfig {
    output: PathBuf,
    sample_hz: u32,
    format: GuestProfileFormat,
    native_artifact_output: Option<PathBuf>,
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

    pub fn folded(output: impl Into<PathBuf>, sample_hz: u32) -> Result<Self, String> {
        Self::new(output, sample_hz, GuestProfileFormat::Folded)
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
