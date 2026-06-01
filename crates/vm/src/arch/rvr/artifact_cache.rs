//! Content-addressed fingerprinting for compiled rvr shared libraries.
//!
//! The fingerprint covers all inputs that affect the compiled `.so`: the
//! generated C project files, toolchain identity, runtime CPU features (for
//! `-march=native`), make invocation arguments, and any extra extension
//! compiler flags. Two compilations that produce the same fingerprint are
//! safe to share a cached artifact.

use std::{fs, io, path::Path};

use sha2::{Digest, Sha256};

/// Compute a hex fingerprint for a prepared rvr C project directory.
///
/// Inputs hashed:
/// - `native_debug_info` flag (affects the `DEBUG=` make variable)
/// - `toolchain.compiler` path (different compiler → different binary)
/// - `toolchain.linker` (affects link flags and output)
/// - `toolchain.make` (build driver identity)
/// - `toolchain.host_os` (affects LTO and other platform flags)
/// - [`host_cpu_features`] (for `-march=native` machine-specificity)
/// - `ext_cflags` (extra compiler flags for extension sources)
/// - `make_args` (OPT, LTO, EXT_LIBS, EXT_SRCS and other make variables)
/// - name + content of every file under `project_dir` in sorted order
///
/// Returns a 32-character lowercase hex string (128 bits of SHA-256), or
/// an `io::Error` if the project directory cannot be read.
pub fn compute_fingerprint(
    project_dir: &Path,
    toolchain: &rvr_openvm::RuntimeToolchain,
    native_debug_info: bool,
    ext_cflags: &[String],
    make_args: &[String],
) -> Result<String, io::Error> {
    let mut h = Sha256::new();
    h.update([native_debug_info as u8]);
    h.update(toolchain.compiler.as_bytes());
    h.update(b"\0");
    h.update(toolchain.linker.as_bytes());
    h.update(b"\0");
    h.update(toolchain.make.as_bytes());
    h.update(b"\0");
    h.update(toolchain.host_os.as_bytes());
    h.update(b"\0");
    h.update(host_cpu_features().as_bytes());
    h.update(b"\0");
    for flag in ext_cflags {
        h.update(flag.as_bytes());
        h.update(b"\0");
    }
    for arg in make_args {
        h.update(arg.as_bytes());
        h.update(b"\0");
    }
    hash_dir_into(project_dir, &mut h)?;
    let result = h.finalize();
    Ok(hex::encode(&result[..16]))
}

/// Return a string identifying the host CPU instruction-set features visible
/// to the compiler for `-march=native`.
///
/// Linux: reads the `flags:` line from `/proc/cpuinfo` (the list of ISA
/// extensions the OS exposes), which captures virtualization-masked flags that
/// the model name alone does not distinguish.
///
/// macOS: reads the CPU brand string via `sysctl`. On Apple Silicon the brand
/// string maps directly to a specific `-mcpu=` target.
///
/// Other platforms: fall back to the Rust target arch string.
pub fn host_cpu_features() -> String {
    #[cfg(target_os = "linux")]
    {
        // The `flags` line lists every ISA extension the OS exposes.
        // Two CPUs with the same model name can differ here under
        // virtualization (hypervisor feature masking).
        fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("flags"))
                    .map(|l| l.to_string())
            })
            .unwrap_or_default()
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        std::env::consts::ARCH.to_string()
    }
}

/// Recursively hash the names and contents of all files under `dir` in
/// deterministic sorted order. Returns an error if any file cannot be read.
fn hash_dir_into(dir: &Path, h: &mut Sha256) -> Result<(), io::Error> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if path.is_dir() {
            h.update(name.as_bytes());
            h.update(b"/\0");
            hash_dir_into(&path, h)?;
        } else {
            h.update(name.as_bytes());
            h.update(b"\0");
            h.update(&fs::read(&path)?);
            h.update(b"\0");
        }
    }
    Ok(())
}
