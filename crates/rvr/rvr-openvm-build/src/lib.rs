#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Default clang command used for RVR native C builds.
pub const DEFAULT_CLANG_COMMAND: &str = "clang-22";

/// Minimum supported LLVM clang major version (c2y, preserve_none on AArch64).
pub const MIN_CLANG_MAJOR: u32 = 19;

/// Keg-only Homebrew LLVM locations (not on PATH by default).
#[cfg(target_os = "macos")]
const HOMEBREW_LLVM_BIN_DIRS: &[&str] = &["/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"];

/// Select the C compiler for RVR native code.
///
/// `RVR_CC` is the explicit override and is returned even if the command name
/// is not clang. Callers must validate the selected command with
/// [`ensure_clang_compiler`]. Otherwise prefer `clang-22`, then a clang-valued
/// `CC`, then plain `clang`. Non-clang `CC` values are ignored so ambient Cargo
/// C compiler settings do not make RVR use GCC by accident.
///
/// Only the first whitespace-delimited token from `RVR_CC` or `CC` is used;
/// pass compiler flags through `CFLAGS`.
pub fn default_compiler_command() -> String {
    if let Some(compiler) = env_command("RVR_CC") {
        return compiler;
    }
    if command_exists(DEFAULT_CLANG_COMMAND) {
        return DEFAULT_CLANG_COMMAND.to_string();
    }
    // Probe Homebrew LLVM before falling back to plain `clang`, which on
    // macOS is Apple clang and fails `ensure_clang_compiler`.
    #[cfg(target_os = "macos")]
    for dir in HOMEBREW_LLVM_BIN_DIRS {
        for name in [DEFAULT_CLANG_COMMAND, "clang"] {
            let candidate = format!("{dir}/{name}");
            if command_exists(&candidate) {
                return candidate;
            }
        }
    }
    if let Some(compiler) = env_command("CC").filter(|compiler| is_clang_command(compiler)) {
        return compiler;
    }
    "clang".to_string()
}

/// Check that `compiler` resolves to clang; returns its `--version` output.
pub fn ensure_clang_compiler(compiler: &str) -> Result<String, String> {
    let output = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|e| {
            format!(
                "required RVR C compiler '{compiler}' could not be executed: {e}; \
                 install clang-22 or set RVR_CC=clang"
            )
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let version = format!("{stdout}\n{stderr}");
    if output.status.success() && version.to_ascii_lowercase().contains("clang") {
        Ok(version)
    } else {
        Err(format!(
            "RVR C compilation requires clang, but selected compiler '{compiler}' is not clang; \
             install clang-22 or set RVR_CC=clang"
        ))
    }
}

/// Check that `compiler` can compile the generated RVR C project: LLVM (not
/// Apple) clang >= [`MIN_CLANG_MAJOR`]. Stricter than [`ensure_clang_compiler`].
pub fn ensure_rvr_clang_compiler(compiler: &str) -> Result<(), String> {
    let version = ensure_clang_compiler(compiler)?;
    // Unparsable versions pass; only reject known-bad configurations.
    let Some((is_apple, major)) = parse_clang_version_output(&version) else {
        return Ok(());
    };
    if is_apple {
        return Err(format!(
            "selected compiler '{compiler}' is Apple clang {major}, whose LTO bitcode is \
             incompatible with the lld linker used for RVR builds; install LLVM clang \
             (e.g. `brew install llvm`) and put clang-22 on PATH, or set \
             RVR_CC=$(brew --prefix llvm)/bin/clang"
        ));
    }
    if major < MIN_CLANG_MAJOR {
        return Err(format!(
            "selected compiler '{compiler}' is clang {major}, but RVR requires \
             clang >= {MIN_CLANG_MAJOR} (c2y, preserve_none); install a newer LLVM \
             (e.g. clang-22) or set RVR_CC to a newer clang"
        ));
    }
    Ok(())
}

/// Parse `clang --version` output into `(is_apple, major_version)`.
/// Returns `None` if no `clang version <N>` marker is found.
fn parse_clang_version_output(version: &str) -> Option<(bool, u32)> {
    let lower = version.to_ascii_lowercase();
    let is_apple = lower.contains("apple clang");
    let rest = lower.split("clang version ").nth(1)?;
    let major = rest
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()?;
    Some((is_apple, major))
}

pub fn command_exists(command: &str) -> bool {
    if command.contains(std::path::MAIN_SEPARATOR) {
        return is_executable_file(Path::new(command));
    }

    let Some(path_var) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&path_var).any(|dir| is_executable_file(&dir.join(command)))
}

pub fn is_clang_command(command: &str) -> bool {
    clang_version_suffix(command).is_some()
}

pub fn clang_version_suffix(command: &str) -> Option<&str> {
    let basename = command_basename(command);
    if basename == "clang" || basename.starts_with("clang-") {
        basename.strip_prefix("clang")
    } else {
        None
    }
}

/// Build a Rust staticlib crate in a private target directory and return the
/// expected archive path.
pub fn build_rust_staticlib(
    manifest_path: &Path,
    target_dir: &Path,
    lib_name: &str,
    crate_name: &str,
) -> PathBuf {
    build_rust_staticlib_with_features(manifest_path, target_dir, lib_name, crate_name, &[])
}

/// Build a Rust staticlib crate with the requested Cargo features in a private
/// target directory and return the expected archive path.
pub fn build_rust_staticlib_with_features(
    manifest_path: &Path,
    target_dir: &Path,
    lib_name: &str,
    crate_name: &str,
    features: &[&str],
) -> PathBuf {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(&cargo);
    command
        .args([
            "build",
            "--release",
            "--config",
            "profile.release.lto=false",
            "--manifest-path",
        ])
        .arg(manifest_path)
        .arg("--target-dir")
        .arg(target_dir);
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn cargo for {crate_name}: {e}"));

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("cargo build for {crate_name} failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    }

    let lib_path = target_dir.join("release").join(lib_name);
    assert!(
        lib_path.exists(),
        "expected staticlib at {} after cargo build",
        lib_path.display()
    );
    lib_path
}

fn env_command(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.split_whitespace().next().map(str::to_string))
}

fn command_basename(command: &str) -> &str {
    Path::new(command)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(command)
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::{clang_version_suffix, is_clang_command, parse_clang_version_output};

    #[test]
    fn detects_clang_command_names() {
        assert!(is_clang_command("clang"));
        assert!(is_clang_command("clang-22"));
        assert!(is_clang_command("/usr/bin/clang-22"));
        assert!(is_clang_command("/opt/homebrew/opt/llvm/bin/clang"));
    }

    #[test]
    fn rejects_non_clang_command_names() {
        assert!(!is_clang_command("gcc"));
        assert!(!is_clang_command("/opt/not-clang-tools/gcc"));
        assert!(!is_clang_command("my-clang-wrapper"));
        assert!(!is_clang_command("clang++"));
    }

    #[test]
    fn extracts_clang_version_suffix() {
        assert_eq!(clang_version_suffix("clang"), Some(""));
        assert_eq!(clang_version_suffix("clang-22"), Some("-22"));
        assert_eq!(clang_version_suffix("/usr/bin/clang-18"), Some("-18"));
        assert_eq!(clang_version_suffix("gcc"), None);
    }

    #[test]
    fn parses_clang_version_output() {
        assert_eq!(
            parse_clang_version_output(
                "Apple clang version 17.0.0 (clang-1700.4.4.1)\nTarget: arm64-apple-darwin25.3.0"
            ),
            Some((true, 17))
        );
        assert_eq!(
            parse_clang_version_output(
                "Homebrew clang version 22.1.5\nTarget: arm64-apple-darwin25.3.0"
            ),
            Some((false, 22))
        );
        assert_eq!(
            parse_clang_version_output("Ubuntu clang version 18.1.3 (1ubuntu1)"),
            Some((false, 18))
        );
        assert_eq!(
            parse_clang_version_output("clang version 19.1.0"),
            Some((false, 19))
        );
        assert_eq!(parse_clang_version_output("gcc (GCC) 13.2.0"), None);
    }
}
