use std::{fmt, process::Command};

use rvr_openvm_build::{clang_version_suffix, is_clang_command};
pub use rvr_openvm_build::{
    command_exists, default_compiler_command, ensure_clang_compiler, DEFAULT_CLANG_COMMAND,
};

/// C compiler to use for generated code.
///
/// Accepts clang compiler commands (e.g., "clang", "clang-20").
/// For clang, the linker (lld) version is auto-derived from the compiler
/// command (e.g., "clang-20" → "lld-20"). Use `with_linker()` to override.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiler {
    command: String,
    linker: Option<String>,
}

impl Compiler {
    /// Create a compiler with the given command.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            linker: None,
        }
    }

    /// Default clang compiler.
    #[must_use]
    pub fn clang() -> Self {
        Self::new(DEFAULT_CLANG_COMMAND)
    }

    /// Set explicit linker command (overrides auto-derivation).
    #[must_use]
    pub fn with_linker(mut self, linker: impl Into<String>) -> Self {
        self.linker = Some(linker.into());
        self
    }

    /// Command to invoke.
    #[must_use]
    pub fn command(&self) -> &str {
        &self.command
    }

    /// Check if this is a clang-based compiler (for flag selection).
    #[must_use]
    pub fn is_clang(&self) -> bool {
        is_clang_command(&self.command)
    }

    /// Get the linker to use with `-fuse-ld=`. Returns the linker (explicit
    /// or auto-derived from compiler version) for clang; `None` for gcc.
    #[must_use]
    pub fn linker(&self) -> Option<String> {
        if !self.is_clang() {
            return None;
        }
        if let Some(ref linker) = self.linker {
            return Some(linker.clone());
        }
        Some(format!("lld{}", self.version_suffix()))
    }

    /// Get llvm-addr2line command, auto-derived from compiler version.
    #[must_use]
    pub fn addr2line(&self) -> String {
        format!("llvm-addr2line{}", self.version_suffix())
    }

    /// Extract version suffix from compiler command (e.g., "clang-20" → "-20").
    fn version_suffix(&self) -> &str {
        clang_version_suffix(&self.command).unwrap_or("")
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::clang()
    }
}

pub fn default_compiler() -> Compiler {
    Compiler::new(default_compiler_command())
}

pub fn default_linker() -> Option<String> {
    std::env::var("RVR_LD")
        .ok()
        .or_else(|| std::env::var("LD").ok())
        .filter(|value| !value.trim().is_empty())
}

pub fn default_linker_or_lld() -> String {
    default_linker().unwrap_or_else(|| "lld".to_string())
}

pub fn default_make_command() -> String {
    std::env::var("RVR_MAKE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .and_then(|value| value.split_whitespace().next().map(str::to_string))
        .unwrap_or_else(|| "make".to_string())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeToolchain {
    pub compiler: String,
    pub linker: String,
    pub make: String,
    pub host_os: &'static str,
}

impl RuntimeToolchain {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            compiler: default_compiler_command(),
            linker: default_linker_or_lld(),
            make: default_make_command(),
            host_os: host_os(),
        }
    }

    pub fn check(self) -> Result<Self, RuntimeToolchainError> {
        let mut missing = Vec::new();
        if let Err(message) = ensure_clang_compiler(&self.compiler) {
            missing.push(MissingTool {
                role: "C compiler",
                command: self.compiler.clone(),
                hint: message,
            });
        }
        if !linker_exists(&self.linker) {
            missing.push(MissingTool {
                role: "linker",
                command: self.linker.clone(),
                hint: "install lld or ld.lld, or set RVR_LD/LD".to_string(),
            });
        }
        if !command_exists(&self.make) {
            missing.push(MissingTool {
                role: "build tool",
                command: self.make.clone(),
                hint: "install make or set RVR_MAKE".to_string(),
            });
        }

        if missing.is_empty() {
            Ok(self)
        } else {
            Err(RuntimeToolchainError::MissingTools(MissingTools(missing)))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeToolchainError {
    #[error("{0}")]
    MissingTools(MissingTools),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingTools(pub Vec<MissingTool>);

impl fmt::Display for MissingTools {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_slice() {
            [] => write!(f, "runtime toolchain is available"),
            [tool] => write!(f, "{tool}"),
            tools => {
                write!(f, "runtime toolchain unavailable: ")?;
                for (i, tool) in tools.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{tool}")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingTool {
    pub role: &'static str,
    pub command: String,
    pub hint: String,
}

impl fmt::Display for MissingTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "required {} '{}' unavailable; {}",
            self.role, self.command, self.hint
        )
    }
}

pub fn runtime_toolchain() -> Result<RuntimeToolchain, RuntimeToolchainError> {
    RuntimeToolchain::from_env().check()
}

pub fn default_addr2line_cmd() -> String {
    if let Some(explicit) = std::env::var_os("LLVM_ADDR2LINE") {
        let explicit = explicit.to_string_lossy().into_owned();
        if command_exists(&explicit) {
            return explicit;
        }
    }

    resolve_llvm_tool("llvm-addr2line", Some(default_compiler().addr2line()))
        .unwrap_or_else(|| "llvm-addr2line".to_string())
}

pub fn default_dwarfdump_cmd() -> Option<String> {
    if let Some(explicit) = std::env::var_os("LLVM_DWARFDUMP") {
        let explicit = explicit.to_string_lossy().into_owned();
        if command_exists(&explicit) {
            return Some(explicit);
        }
    }

    resolve_llvm_tool(
        "llvm-dwarfdump",
        derive_llvm_tool_from_compiler("llvm-dwarfdump"),
    )
}

fn resolve_llvm_tool(base_name: &str, preferred: Option<String>) -> Option<String> {
    let mut candidates = Vec::new();
    if let Some(preferred) = preferred {
        candidates.push(preferred);
    }
    candidates.push(base_name.to_string());
    candidates.push(format!("/usr/bin/{base_name}"));
    candidates.push(format!("/opt/homebrew/opt/llvm/bin/{base_name}"));
    candidates.push(format!("/usr/local/opt/llvm/bin/{base_name}"));

    if let Some(versioned) = derive_llvm_tool_from_compiler(base_name) {
        candidates.push(versioned.clone());
        candidates.push(format!("/usr/bin/{versioned}"));
    }

    dedup_preserve_order(candidates)
        .into_iter()
        .find(|candidate| command_exists(candidate))
}

fn derive_llvm_tool_from_compiler(base_name: &str) -> Option<String> {
    let compiler = default_compiler_command();
    let version_suffix = clang_version_suffix(&compiler)?;
    if version_suffix.is_empty() {
        detect_clang_major(&compiler).map(|major| format!("{base_name}-{major}"))
    } else {
        Some(format!("{base_name}{version_suffix}"))
    }
}

fn detect_clang_major(compiler: &str) -> Option<u32> {
    let output = Command::new(compiler).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next()?;
    let version = first_line
        .split_whitespace()
        .find(|token| token.as_bytes().first().is_some_and(u8::is_ascii_digit))?;
    version.split('.').next()?.parse().ok()
}

pub fn linker_exists(linker: &str) -> bool {
    command_exists(linker) || (linker == "lld" && command_exists("ld.lld"))
}

fn host_os() -> &'static str {
    if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "macos") {
        "Darwin"
    } else {
        std::env::consts::OS
    }
}

fn dedup_preserve_order(candidates: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    for candidate in candidates {
        if seen.insert(candidate.clone()) {
            deduped.push(candidate);
        }
    }
    deduped
}
