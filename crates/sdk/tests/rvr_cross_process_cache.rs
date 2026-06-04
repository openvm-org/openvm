//! Cross-process cache reuse test for the rvr backend.
//!
//! Runs rvr_cache_probe twice per mode against the same cache dir. The first
//! run is a cold compile; the second should be a cache hit. Three assertions
//! verify the fix:
//!
//! 1. The cache dir still contains exactly one artifact after the second run —
//!    no second file was created, so the probe did not recompile.
//! 2. The artifact mtime is unchanged — the existing file was not rewritten.
//! 3. Both runs produce the same public values — the loaded artifact is
//!    functionally correct.

#[cfg(feature = "rvr")]
mod tests {
    use std::path::Path;

    const PROBE: &str = env!("CARGO_BIN_EXE_rvr_cache_probe");

    fn artifacts(dir: &Path) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "so" || ext == "dylib"))
            .collect()
    }

    fn run_probe(mode: &str, cache_dir: &Path) -> Vec<u8> {
        let output = std::process::Command::new(PROBE)
            .arg(mode)
            .arg(cache_dir)
            .output()
            .expect("failed to spawn rvr_cache_probe");
        assert!(
            output.status.success(),
            "probe failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        hex::decode(String::from_utf8_lossy(&output.stdout).trim())
            .expect("probe stdout is not valid hex")
    }

    #[test]
    fn test_rvr_cache_cross_process() {
        for mode in ["pure", "metered", "metered_cost"] {
            let cache_dir = tempfile::tempdir().unwrap();

            let pv_cold = run_probe(mode, cache_dir.path());

            let arts = artifacts(cache_dir.path());
            assert_eq!(arts.len(), 1, "[{mode}] expected one artifact after cold compile");
            let mtime_before = std::fs::metadata(&arts[0]).unwrap().modified().unwrap();

            let pv_warm = run_probe(mode, cache_dir.path());

            let arts_after = artifacts(cache_dir.path());
            assert_eq!(
                arts_after.len(),
                1,
                "[{mode}] second run created a new artifact — fingerprint not stable across processes"
            );
            assert_eq!(
                std::fs::metadata(&arts_after[0]).unwrap().modified().unwrap(),
                mtime_before,
                "[{mode}] artifact was rewritten — second run recompiled instead of reusing cache"
            );
            assert_eq!(pv_cold, pv_warm, "[{mode}] second run produced different public values");
        }
    }
}
