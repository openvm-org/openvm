#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Default clang command used for RVR native C builds.
pub const DEFAULT_CLANG_COMMAND: &str = "clang-22";

/// Select the C compiler for RVR native code.
///
/// `RVR_CC` is the explicit override and is returned even if the command name
/// is not clang. Callers must validate the selected command with
/// [`ensure_clang_compiler`]. Otherwise prefer `clang-22`, then a clang-valued
/// `CC`, then plain `clang`. Non-clang `CC` values are ignored so ambient Cargo
/// C compiler settings do not make RVR use GCC by accident.
pub fn default_compiler_command() -> String {
    if let Some(compiler) = env_command("RVR_CC") {
        return compiler;
    }
    if command_exists(DEFAULT_CLANG_COMMAND) {
        return DEFAULT_CLANG_COMMAND.to_string();
    }
    if let Some(compiler) = env_command("CC").filter(|compiler| is_clang_command(compiler)) {
        return compiler;
    }
    "clang".to_string()
}

/// Check that `compiler` resolves to clang.
pub fn ensure_clang_compiler(compiler: &str) -> Result<(), String> {
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
        Ok(())
    } else {
        Err(format!(
            "RVR C compilation requires clang, but selected compiler '{compiler}' is not clang; \
             install clang-22 or set RVR_CC=clang"
        ))
    }
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

/// Build a Rust staticlib crate in a private target directory and return the
/// expected archive path.
pub fn build_rust_staticlib(
    manifest_path: &Path,
    target_dir: &Path,
    lib_name: &str,
    crate_name: &str,
) -> PathBuf {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .args([
            "build",
            "--release",
            "--config",
            "profile.release.lto=false",
            "--manifest-path",
        ])
        .arg(manifest_path)
        .arg("--target-dir")
        .arg(target_dir)
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

fn is_clang_command(command: &str) -> bool {
    Path::new(command)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "clang" || name.starts_with("clang-"))
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
