use std::{
    ffi::OsString,
    path::Path,
    sync::{Mutex, MutexGuard},
};

use eyre::{bail, eyre, Context};
pub use openvm_prof::firefox::FirefoxProfile as ExecutionProfile;

use crate::SdkError;

const PROFILE_PATH_ENV: &str = "OPENVM_RVR_GUEST_CALL_PROFILE";
const PROFILE_HZ_ENV: &str = "OPENVM_RVR_GUEST_CALL_PROFILE_HZ";
const PROFILE_FORMAT_ENV: &str = "OPENVM_RVR_GUEST_CALL_PROFILE_FORMAT";
static PROFILE_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Temporarily enables the low-overhead sampler owned by the RVR executor.
struct GuestProfileGuard {
    _lock: MutexGuard<'static, ()>,
    previous: [(OsString, Option<OsString>); 3],
}

impl GuestProfileGuard {
    fn start(path: &Path, sample_hz: u32) -> eyre::Result<Self> {
        if sample_hz == 0 || sample_hz > 1_000_000 {
            bail!("execution profile rate must be in 1..=1000000, got {sample_hz}");
        }
        let lock = PROFILE_ENV_LOCK
            .lock()
            .map_err(|_| eyre!("execution profile lock is poisoned"))?;
        let previous = [
            (PROFILE_PATH_ENV.into(), std::env::var_os(PROFILE_PATH_ENV)),
            (PROFILE_HZ_ENV.into(), std::env::var_os(PROFILE_HZ_ENV)),
            (
                PROFILE_FORMAT_ENV.into(),
                std::env::var_os(PROFILE_FORMAT_ENV),
            ),
        ];
        // The process-wide lock prevents concurrent profiling sessions from
        // observing each other's private RVR configuration variables.
        unsafe {
            std::env::set_var(PROFILE_PATH_ENV, path);
            std::env::set_var(PROFILE_HZ_ENV, sample_hz.to_string());
            std::env::set_var(PROFILE_FORMAT_ENV, "raw");
        }
        Ok(Self {
            _lock: lock,
            previous,
        })
    }
}

impl Drop for GuestProfileGuard {
    fn drop(&mut self) {
        for (key, value) in &self.previous {
            // SAFETY: the process-wide lock excludes other profiling sessions
            // while these private variables are installed and restored.
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

/// Profile one SDK execution and return its value together with a reusable
/// Firefox Profiler artifact.
///
/// The closure is intentionally mode-agnostic: callers may wrap [`execute`](crate::GenericSdk::execute),
/// [`execute_metered`](crate::GenericSdk::execute_metered), or
/// [`execute_metered_cost`](crate::GenericSdk::execute_metered_cost) without duplicating sampler,
/// symbolication, compression, or upload logic.
pub fn profile_execution<T>(
    guest_elf_path: &Path,
    sample_hz: u32,
    execute: impl FnOnce() -> Result<T, SdkError>,
) -> Result<(T, ExecutionProfile), SdkError> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err(SdkError::Other(eyre!(
            "RVR execution profiling requires Linux x86_64"
        )));
    }

    let raw_profile =
        tempfile::NamedTempFile::new().context("failed to create temporary RVR profile")?;
    let guard = GuestProfileGuard::start(raw_profile.path(), sample_hz)?;
    let execution_result = execute();
    drop(guard);
    let output = execution_result?;
    let profile =
        ExecutionProfile::from_raw_guest_stacks(raw_profile.path(), guest_elf_path, sample_hz)?;
    Ok((output, profile))
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use super::{GuestProfileGuard, PROFILE_FORMAT_ENV, PROFILE_HZ_ENV, PROFILE_PATH_ENV};

    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn env_test_lock() -> MutexGuard<'static, ()> {
        ENV_TEST_LOCK.lock().unwrap()
    }

    #[test]
    fn guard_restores_existing_profiler_configuration() {
        let _test_lock = env_test_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.raw");
        let previous = [
            (PROFILE_PATH_ENV, std::env::var_os(PROFILE_PATH_ENV)),
            (PROFILE_HZ_ENV, std::env::var_os(PROFILE_HZ_ENV)),
            (PROFILE_FORMAT_ENV, std::env::var_os(PROFILE_FORMAT_ENV)),
        ];
        unsafe {
            std::env::set_var(PROFILE_PATH_ENV, "previous-profile");
            std::env::set_var(PROFILE_HZ_ENV, "17");
            std::env::set_var(PROFILE_FORMAT_ENV, "folded");
        }

        {
            let _guard = GuestProfileGuard::start(&path, 2000).unwrap();
            assert_eq!(
                std::env::var_os(PROFILE_PATH_ENV).as_deref(),
                Some(path.as_os_str())
            );
            assert_eq!(std::env::var(PROFILE_HZ_ENV).unwrap(), "2000");
            assert_eq!(std::env::var(PROFILE_FORMAT_ENV).unwrap(), "raw");
        }
        assert_eq!(std::env::var(PROFILE_PATH_ENV).unwrap(), "previous-profile");
        assert_eq!(std::env::var(PROFILE_HZ_ENV).unwrap(), "17");
        assert_eq!(std::env::var(PROFILE_FORMAT_ENV).unwrap(), "folded");

        for (key, value) in previous {
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}
