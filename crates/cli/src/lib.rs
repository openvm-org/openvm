pub mod commands;
pub mod default;
pub mod input;
pub mod util;

use std::process::{Command, Stdio};

use eyre::{Context, Result};

pub const RUSTUP_TOOLCHAIN_NAME: &str = "nightly-2025-02-14";

pub const OPENVM_VERSION_MESSAGE: &str = concat!(
    "v",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("VERGEN_GIT_SHA"),
    ")"
);

#[allow(dead_code)]
trait CommandExecutor {
    fn run(&mut self) -> Result<()>;
}

impl CommandExecutor for Command {
    fn run(&mut self) -> Result<()> {
        self.stderr(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stdin(Stdio::inherit())
            .output()
            .with_context(|| format!("while executing `{:?}`", &self))
            .map(|_| ())
    }
}

#[allow(unreachable_code)]
pub fn is_supported_target() -> bool {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return true;

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    return true;

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    return true;

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return true;

    false
}

pub fn get_target() -> String {
    target_lexicon::HOST.to_string()
}
