use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

pub const MAX_GUEST_PROFILE_SAMPLE_HZ: u32 = 20_000;
const DEFAULT_MAX_SAMPLES: usize = 1 << 18;

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
    /// Guest PC snapshot. It is exact at profiled host-call boundaries; the
    /// converter discards it when the native leaf is generated guest code.
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
#[derive(Clone, Debug)]
pub struct GuestProfileConfig {
    output: PathBuf,
    sample_hz: u32,
    native_artifact_output: Option<PathBuf>,
    session: Option<Arc<Mutex<GuestProfileSession>>>,
    max_samples: usize,
}

#[derive(Debug, Default)]
struct GuestProfileSession {
    profile: Option<RawGuestProfile>,
    native_artifact_saved: bool,
}

impl GuestProfileConfig {
    pub fn new(output: impl Into<PathBuf>, sample_hz: u32) -> Result<Self, String> {
        if !(1..=MAX_GUEST_PROFILE_SAMPLE_HZ).contains(&sample_hz) {
            return Err(format!(
                "guest profile sampling rate must be in 1..={MAX_GUEST_PROFILE_SAMPLE_HZ}, got {sample_hz}"
            ));
        }
        Ok(Self {
            output: output.into(),
            sample_hz,
            native_artifact_output: None,
            session: None,
            max_samples: DEFAULT_MAX_SAMPLES,
        })
    }

    pub fn raw(output: impl Into<PathBuf>, sample_hz: u32) -> Result<Self, String> {
        Self::new(output, sample_hz)
    }

    /// Capture ordered samples and preserve the exact native artifact needed
    /// to resolve interrupted host PCs after execution.
    pub fn raw_with_native_artifact(
        output: impl Into<PathBuf>,
        native_artifact_output: impl Into<PathBuf>,
        sample_hz: u32,
    ) -> Result<Self, String> {
        let output = output.into();
        let native_artifact_output = native_artifact_output.into();
        if paths_alias(&output, &native_artifact_output)? {
            return Err(
                "guest profile output and native artifact output must be different paths"
                    .to_string(),
            );
        }
        let mut config = Self::raw(output, sample_hz)?;
        config.native_artifact_output = Some(native_artifact_output);
        Ok(config)
    }

    /// Capture a logical session that can span several sequential executions.
    pub fn raw_session(output: impl Into<PathBuf>, sample_hz: u32) -> Result<Self, String> {
        let mut config = Self::raw(output, sample_hz)?;
        config.session = Some(Arc::default());
        Ok(config)
    }

    /// Capture a logical session and preserve its generated native artifact.
    pub fn raw_session_with_native_artifact(
        output: impl Into<PathBuf>,
        native_artifact_output: impl Into<PathBuf>,
        sample_hz: u32,
    ) -> Result<Self, String> {
        let mut config = Self::raw_with_native_artifact(output, native_artifact_output, sample_hz)?;
        config.session = Some(Arc::default());
        Ok(config)
    }

    pub fn output(&self) -> &Path {
        &self.output
    }

    pub fn sample_hz(&self) -> u32 {
        self.sample_hz
    }

    pub(crate) fn native_artifact_output(&self) -> Option<&Path> {
        self.native_artifact_output.as_deref()
    }

    pub(crate) fn is_session(&self) -> bool {
        self.session.is_some()
    }

    pub(crate) fn staging_session(&self) -> Self {
        Self {
            output: self.output.clone(),
            sample_hz: self.sample_hz,
            native_artifact_output: None,
            session: Some(Arc::default()),
            max_samples: self.max_samples,
        }
    }

    pub(crate) fn take_session_profile(&self) -> Result<RawGuestProfile, String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "guest profile is not configured as a session".to_string())?;
        let mut session = session
            .lock()
            .map_err(|_| "guest profile session lock is poisoned".to_string())?;
        session
            .profile
            .take()
            .ok_or_else(|| "guest profile session contains no executions".to_string())
    }

    pub(crate) fn session_needs_native_artifact(&self) -> Result<bool, String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "guest profile is not configured as a session".to_string())?;
        let session = session
            .lock()
            .map_err(|_| "guest profile session lock is poisoned".to_string())?;
        Ok(!session.native_artifact_saved)
    }

    pub(crate) fn validate_session_profile(&self, profile: &RawGuestProfile) -> Result<(), String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "guest profile is not configured as a session".to_string())?;
        let session = session
            .lock()
            .map_err(|_| "guest profile session lock is poisoned".to_string())?;
        if let Some(existing) = &session.profile {
            validate_session_append(existing, profile, self.max_samples)?;
        }
        Ok(())
    }

    pub(crate) fn append_session(
        &self,
        profile: RawGuestProfile,
        native_artifact_saved: bool,
    ) -> Result<(), String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "guest profile is not configured as a session".to_string())?;
        let mut session = session
            .lock()
            .map_err(|_| "guest profile session lock is poisoned".to_string())?;
        if let Some(existing) = &session.profile {
            validate_session_append(existing, &profile, self.max_samples)?;
        }
        session.profile = Some(match session.profile.take() {
            Some(existing) => merge_compatible_raw_profiles(existing, profile),
            None => profile,
        });
        session.native_artifact_saved |= native_artifact_saved;
        Ok(())
    }

    /// Write a completed session to its configured raw output path.
    pub fn finish_session(&self) -> Result<(), String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "guest profile is not configured as a session".to_string())?;
        let session = session
            .lock()
            .map_err(|_| "guest profile session lock is poisoned".to_string())?;
        let profile = session
            .profile
            .as_ref()
            .ok_or_else(|| "guest profile session contains no executions".to_string())?;
        let output = serde_json::to_vec(profile)
            .map_err(|error| format!("failed to serialize raw guest profile: {error}"))?;
        std::fs::write(&self.output, output).map_err(|error| {
            format!(
                "failed to write guest profile {}: {error}",
                self.output.display()
            )
        })
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

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn max_samples(&self) -> usize {
        self.max_samples
    }
}

fn paths_alias(left: &Path, right: &Path) -> Result<bool, String> {
    let left_absolute = std::path::absolute(left)
        .map_err(|error| format!("failed to resolve profile output path: {error}"))?;
    let right_absolute = std::path::absolute(right)
        .map_err(|error| format!("failed to resolve native artifact output path: {error}"))?;
    if left_absolute == right_absolute {
        return Ok(true);
    }
    let (Ok(left_metadata), Ok(right_metadata)) =
        (std::fs::metadata(left), std::fs::metadata(right))
    else {
        return Ok(false);
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        Ok(left_metadata.dev() == right_metadata.dev()
            && left_metadata.ino() == right_metadata.ino())
    }
    #[cfg(not(unix))]
    {
        Ok(std::fs::canonicalize(left).ok() == std::fs::canonicalize(right).ok())
    }
}

#[cfg(test)]
fn merge_raw_profiles(
    existing: RawGuestProfile,
    next: RawGuestProfile,
) -> Result<RawGuestProfile, String> {
    validate_raw_profile_compatibility(&existing, &next)?;
    Ok(merge_compatible_raw_profiles(existing, next))
}

fn validate_session_append(
    existing: &RawGuestProfile,
    next: &RawGuestProfile,
    max_samples: usize,
) -> Result<(), String> {
    validate_raw_profile_compatibility(existing, next)?;
    if existing.samples.len().saturating_add(next.samples.len()) > max_samples {
        return Err(format!(
            "profile session exceeds its {max_samples}-sample memory bound"
        ));
    }
    Ok(())
}

fn validate_raw_profile_compatibility(
    existing: &RawGuestProfile,
    next: &RawGuestProfile,
) -> Result<(), String> {
    if existing.version != next.version
        || existing.requested_sample_hz != next.requested_sample_hz
        || existing.owner_tid != next.owner_tid
    {
        return Err(
            "profile session execution is incompatible with earlier executions".to_string(),
        );
    }
    let existing_generated = existing
        .native_modules
        .iter()
        .find(|module| module.generated);
    let next_generated = next.native_modules.iter().find(|module| module.generated);
    if existing_generated.map(|module| (&module.name, &module.path))
        != next_generated.map(|module| (&module.name, &module.path))
    {
        return Err("profile session cannot span different generated artifacts".to_string());
    }
    Ok(())
}

fn merge_compatible_raw_profiles(
    mut existing: RawGuestProfile,
    mut next: RawGuestProfile,
) -> RawGuestProfile {
    let wall_offset = existing.end_wall_time_ns;
    let cpu_offset = existing.end_cpu_time_ns;
    for sample in &mut next.samples {
        sample.wall_time_ns =
            wall_offset.saturating_add(sample.wall_time_ns.saturating_sub(next.start_wall_time_ns));
        sample.cpu_time_ns =
            cpu_offset.saturating_add(sample.cpu_time_ns.saturating_sub(next.start_cpu_time_ns));
    }

    let mut module_indices = Vec::with_capacity(next.native_modules.len());
    for module in next.native_modules.drain(..) {
        let existing_index = existing
            .native_modules
            .iter()
            .position(|candidate| {
                candidate.name == module.name
                    && candidate.path == module.path
                    && candidate.generated == module.generated
            })
            .unwrap_or_else(|| {
                existing.native_modules.push(module);
                existing.native_modules.len() - 1
            });
        module_indices.push(existing_index as u32);
    }
    for sample in &mut next.samples {
        if let Some(frame) = &mut sample.native_leaf {
            if let Some(index) = frame.module_index {
                frame.module_index = module_indices.get(index as usize).copied();
            }
        }
    }

    existing.end_wall_time_ns = wall_offset.saturating_add(
        next.end_wall_time_ns
            .saturating_sub(next.start_wall_time_ns),
    );
    existing.end_cpu_time_ns =
        cpu_offset.saturating_add(next.end_cpu_time_ns.saturating_sub(next.start_cpu_time_ns));
    existing.delivered_samples = existing
        .delivered_samples
        .saturating_add(next.delivered_samples);
    existing.dropped_samples = existing
        .dropped_samples
        .saturating_add(next.dropped_samples);
    existing.timer_overruns = existing.timer_overruns.saturating_add(next.timer_overruns);
    existing.timer_arm_failures = existing
        .timer_arm_failures
        .saturating_add(next.timer_arm_failures);
    existing.clock_failures = existing.clock_failures.saturating_add(next.clock_failures);
    existing.samples.append(&mut next.samples);
    existing
}

#[cfg(test)]
mod tests {
    use super::{
        merge_raw_profiles, GuestProfileConfig, RawGuestProfile, RawGuestProfileSample,
        RawNativeFrame, RawNativeModule, MAX_GUEST_PROFILE_SAMPLE_HZ, RAW_GUEST_PROFILE_VERSION,
    };

    fn raw_profile(module_path: &str, start: u64, end: u64) -> RawGuestProfile {
        RawGuestProfile {
            version: RAW_GUEST_PROFILE_VERSION,
            requested_sample_hz: 1_000,
            owner_tid: 7,
            start_unix_time_ns: 100,
            start_wall_time_ns: start,
            end_wall_time_ns: end,
            start_cpu_time_ns: start,
            end_cpu_time_ns: end,
            delivered_samples: 1,
            dropped_samples: 0,
            timer_overruns: 0,
            timer_arm_failures: 0,
            clock_failures: 0,
            native_modules: vec![RawNativeModule {
                name: "generated.so".to_string(),
                path: module_path.to_string(),
                generated: true,
            }],
            samples: vec![RawGuestProfileSample {
                wall_time_ns: start + 1,
                cpu_time_ns: start + 1,
                native_leaf: Some(RawNativeFrame {
                    module_index: Some(0),
                    pc: start,
                }),
                guest_callsite_pc: None,
                guest_return_pcs: Vec::new(),
                stack_truncated: false,
            }],
        }
    }

    #[test]
    fn validates_sampling_rate_at_both_boundaries() {
        assert!(GuestProfileConfig::raw("profile.raw", 0).is_err());
        assert!(GuestProfileConfig::raw("profile.raw", 1).is_ok());
        assert!(GuestProfileConfig::raw("profile.raw", MAX_GUEST_PROFILE_SAMPLE_HZ).is_ok());
        assert!(GuestProfileConfig::raw("profile.raw", MAX_GUEST_PROFILE_SAMPLE_HZ + 1).is_err());
    }

    #[test]
    fn rejects_aliasing_raw_and_native_artifact_paths() {
        assert!(
            GuestProfileConfig::raw_with_native_artifact("profile.out", "profile.out", 1_000)
                .is_err()
        );
        assert!(GuestProfileConfig::raw_with_native_artifact(
            "profile.out",
            "./profile.out",
            1_000
        )
        .is_err());

        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("profile.out");
        let alias = dir.path().join("artifact.so");
        std::fs::write(&output, []).unwrap();
        std::fs::hard_link(&output, &alias).unwrap();
        assert!(GuestProfileConfig::raw_with_native_artifact(output, alias, 1_000).is_err());
    }

    #[test]
    fn session_merge_removes_inactive_gaps() {
        let merged = merge_raw_profiles(
            raw_profile("/tmp/generated.so", 10, 20),
            raw_profile("/tmp/generated.so", 1_000, 1_020),
        )
        .unwrap();

        assert_eq!(merged.end_wall_time_ns, 40);
        assert_eq!(merged.end_cpu_time_ns, 40);
        assert_eq!(merged.samples[1].wall_time_ns, 21);
        assert_eq!(merged.samples[1].cpu_time_ns, 21);
        assert_eq!(merged.delivered_samples, 2);
        assert_eq!(merged.native_modules.len(), 1);
    }

    #[test]
    fn session_rejects_a_different_generated_artifact() {
        let error = merge_raw_profiles(
            raw_profile("/tmp/first.so", 10, 20),
            raw_profile("/tmp/second.so", 30, 40),
        )
        .unwrap_err();
        assert!(error.contains("different generated artifacts"));
    }

    #[test]
    fn rejected_session_append_preserves_existing_samples() {
        let config = GuestProfileConfig::raw_session("profile.raw", 1_000).unwrap();
        config
            .append_session(raw_profile("generated.so", 10, 20), false)
            .unwrap();
        assert!(config
            .append_session(raw_profile("other.so", 30, 40), false)
            .is_err());
        config
            .append_session(raw_profile("generated.so", 50, 60), false)
            .unwrap();

        let profile = config.take_session_profile().unwrap();
        assert_eq!(profile.delivered_samples, 2);
        assert_eq!(profile.samples.len(), 2);
    }

    #[test]
    fn session_sample_bound_applies_across_executions() {
        let config = GuestProfileConfig::raw_session("profile.raw", 1_000)
            .unwrap()
            .with_max_samples(1)
            .unwrap();
        config
            .append_session(raw_profile("generated.so", 10, 20), false)
            .unwrap();
        let error = config
            .append_session(raw_profile("generated.so", 30, 40), false)
            .unwrap_err();
        assert!(error.contains("1-sample memory bound"));
        assert_eq!(config.take_session_profile().unwrap().samples.len(), 1);
    }

    #[test]
    fn session_does_not_commit_before_its_artifact_is_saved() {
        let config = GuestProfileConfig::raw_session_with_native_artifact(
            "profile.raw",
            "generated.so",
            1_000,
        )
        .unwrap();

        assert!(config.session_needs_native_artifact().unwrap());
        // A caller that fails to save the artifact does not append the profile.
        assert!(config.session_needs_native_artifact().unwrap());

        config
            .append_session(raw_profile("generated.so", 10, 20), true)
            .unwrap();
        assert!(!config.session_needs_native_artifact().unwrap());
    }
}
