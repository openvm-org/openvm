use std::path::{Path, PathBuf};

const MAX_SAMPLE_HZ: u32 = 1_000_000;

/// On-disk representation produced by RVR guest sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestProfileFormat {
    /// Collapsed stacks with sample counts, suitable for flamegraph tools.
    Folded,
    /// One ordered, semicolon-separated stack per sample.
    Raw,
}

/// Explicit configuration for one RVR guest profiling execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuestProfileConfig {
    output: PathBuf,
    sample_hz: u32,
    format: GuestProfileFormat,
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
        })
    }

    pub fn raw(output: impl Into<PathBuf>, sample_hz: u32) -> Result<Self, String> {
        Self::new(output, sample_hz, GuestProfileFormat::Raw)
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
