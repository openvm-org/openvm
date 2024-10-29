// Initial cargo build commands taken from risc0 under Apache 2.0 license

#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use axvm_platform::memory;
use cargo_metadata::{MetadataCommand, Package};
use config::GuestBuildOptions;

pub use self::config::{DockerOptions, GuestOptions};

mod config;

#[allow(dead_code)]
const RUSTUP_TOOLCHAIN_NAME: &str = "axiom";

/// Returns the given cargo Package from the metadata in the Cargo.toml manifest
/// within the provided `manifest_dir`.
pub fn get_package(manifest_dir: impl AsRef<Path>) -> Package {
    let manifest_path = manifest_dir.as_ref().join("Cargo.toml");
    let manifest_meta = MetadataCommand::new()
        .manifest_path(&manifest_path)
        .no_deps()
        .exec()
        .expect("cargo metadata command failed");
    let mut matching: Vec<Package> = manifest_meta
        .packages
        .into_iter()
        .filter(|pkg| {
            let std_path: &Path = pkg.manifest_path.as_ref();
            std_path == manifest_path
        })
        .collect();
    if matching.is_empty() {
        eprintln!(
            "ERROR: No package found in {}",
            manifest_dir.as_ref().display()
        );
        std::process::exit(-1);
    }
    if matching.len() > 1 {
        eprintln!(
            "ERROR: Multiple packages found in {}",
            manifest_dir.as_ref().display()
        );
        std::process::exit(-1);
    }
    matching.pop().unwrap()
}

/// Determines and returns the build target directory from the Cargo manifest at
/// the given `manifest_path`.
pub fn get_target_dir(manifest_path: impl AsRef<Path>) -> PathBuf {
    MetadataCommand::new()
        .manifest_path(manifest_path.as_ref())
        .no_deps()
        .exec()
        .expect("cargo metadata command failed")
        .target_directory
        .into()
}

/// When called from a build.rs, returns the current package being built.
pub fn current_package() -> Package {
    get_package(env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn is_debug() -> bool {
    get_env_var("AXIOM_BUILD_DEBUG") == "1"
}

fn is_skip_build() -> bool {
    !get_env_var("AXIOM_SKIP_BUILD").is_empty()
}

fn get_env_var(name: &str) -> String {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).unwrap_or_default()
}

/// Returns all target ELF paths associated with the given guest crate.
pub fn guest_methods(
    pkg: &Package,
    target_dir: impl AsRef<Path>,
    guest_features: &[String],
) -> Vec<PathBuf> {
    let profile = if is_debug() { "debug" } else { "release" };
    pkg.targets
        .iter()
        .filter(|target| target.kind.iter().any(|kind| kind == "bin"))
        .filter(|target| {
            target
                .required_features
                .iter()
                .all(|required_feature| guest_features.contains(required_feature))
        })
        .map(|target| {
            target_dir
                .as_ref()
                .join("riscv32im-risc0-zkvm-elf")
                .join(profile)
                .join(&target.name)
                .to_path_buf()
        })
        .collect()
}

/// Build a [Command] with CARGO and RUSTUP_TOOLCHAIN environment variables
/// removed.
fn sanitized_cmd(tool: &str) -> Command {
    let mut cmd = Command::new(tool);
    for (key, _val) in env::vars().filter(|x| x.0.starts_with("CARGO")) {
        cmd.env_remove(key);
    }
    cmd.env_remove("RUSTUP_TOOLCHAIN");
    cmd
}

/// Creates a std::process::Command to execute the given cargo
/// command in an environment suitable for targeting the zkvm guest.
pub fn cargo_command(subcmd: &str, rust_flags: &[&str]) -> Command {
    let rustc = sanitized_cmd("rustup")
        .args(["+nightly", "which", "rustc"]) // TODO: switch +nightly to +axiom
        .output()
        .expect("rustup failed to find nightly toolchain")
        .stdout;

    let rustc = String::from_utf8(rustc).unwrap();
    let rustc = rustc.trim();
    println!("Using rustc: {rustc}");

    let mut cmd = sanitized_cmd("cargo +nightly"); // +nightly is temporary
    let mut args = vec![subcmd, "--target", "riscv32im-risc0-zkvm-elf"];

    if std::env::var("AXIOM_BUILD_LOCKED").is_ok() {
        args.push("--locked");
    }

    // let rust_src = get_env_var("AXIOM_RUST_SRC");
    // if !rust_src.is_empty() {
    // TODO[jpw]: only do this for custom src once we make axiom toolchain
    args.push("-Z");
    args.push("build-std=alloc,core,proc_macro,panic_abort"); // ,std");
    args.push("-Z");
    args.push("build-std-features=compiler-builtins-mem");
    // cmd.env("__CARGO_TESTS_ONLY_SRC_ROOT", rust_src);
    // }

    println!("Building guest package: cargo {}", args.join(" "));

    let encoded_rust_flags = encode_rust_flags(rust_flags);

    cmd.env("RUSTC", rustc)
        .env("CARGO_ENCODED_RUSTFLAGS", encoded_rust_flags)
        .args(args);
    cmd
}

/// Returns a string that can be set as the value of CARGO_ENCODED_RUSTFLAGS when compiling guests
pub(crate) fn encode_rust_flags(rustc_flags: &[&str]) -> String {
    [
        // Append other rust flags
        rustc_flags,
        &[
            // Replace atomic ops with nonatomic versions since the guest is single threaded.
            "-C",
            "passes=loweratomic",
            // Specify where to start loading the program in
            // memory.  The clang linker understands the same
            // command line arguments as the GNU linker does; see
            // https://ftp.gnu.org/old-gnu/Manuals/ld-2.9.1/html_mono/ld.html#SEC3
            // for details.
            "-C",
            &format!("link-arg=-Ttext=0x{:08X}", memory::TEXT_START),
            // Apparently not having an entry point is only a linker warning(!), so
            // error out in this case.
            "-C",
            "link-arg=--fatal-warnings",
            "-C",
            "panic=abort",
        ],
    ]
    .concat()
    .join("\x1f")
}

// HACK: Attempt to bypass the parent cargo output capture and
// send directly to the tty, if available.  This way we get
// progress messages from the inner cargo so the user doesn't
// think it's just hanging.
fn tty_println(msg: &str) {
    let tty_file = env::var("RISC0_GUEST_LOGFILE").unwrap_or_else(|_| "/dev/tty".to_string());

    let mut tty = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(tty_file)
        .ok();

    if let Some(tty) = &mut tty {
        writeln!(tty, "{msg}").unwrap();
    } else {
        eprintln!("{msg}");
    }
}

/// Builds a package that targets the riscv guest into the specified target
/// directory.
pub fn build_guest_package<P>(
    pkg: &Package,
    target_dir: P,
    guest_opts: &GuestBuildOptions,
    runtime_lib: Option<&str>,
) where
    P: AsRef<Path>,
{
    if is_skip_build() {
        return;
    }

    fs::create_dir_all(target_dir.as_ref()).unwrap();

    let runtime_rust_flags = runtime_lib
        .map(|lib| vec![String::from("-C"), format!("link_arg={}", lib)])
        .unwrap_or_default();
    let rust_flags: Vec<_> = [
        runtime_rust_flags
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        guest_opts.rustc_flags.iter().map(|s| s.as_str()).collect(),
    ]
    .concat();

    let mut cmd = cargo_command("build", &rust_flags);

    let features_str = guest_opts.features.join(",");
    if !features_str.is_empty() {
        cmd.args(["--features", &features_str]);
    }

    cmd.args([
        "--manifest-path",
        pkg.manifest_path.as_str(),
        "--target-dir",
        target_dir.as_ref().to_str().unwrap(),
    ]);
    tty_println(&format!("cargo command: {:?}", cmd));

    if !is_debug() {
        cmd.args(["--release"]);
    }

    let mut child = cmd
        .stderr(Stdio::piped())
        .spawn()
        .expect("cargo build failed");
    let stderr = child.stderr.take().unwrap();

    tty_println(&format!(
        "{}: Starting build for riscv32im-risc0-zkvm-elf",
        pkg.name
    ));

    for line in BufReader::new(stderr).lines() {
        tty_println(&format!("{}: {}", pkg.name, line.unwrap()));
    }

    let res = child.wait().expect("Guest 'cargo build' failed");
    if !res.success() {
        std::process::exit(res.code().unwrap());
    }
}

/// Detect rust toolchain of given name
pub fn detect_toolchain(name: &str) {
    let result = Command::new("rustup")
        .args(["toolchain", "list", "--verbose"])
        .stderr(Stdio::inherit())
        .output()
        .unwrap();
    if !result.status.success() {
        eprintln!("Failed to run: 'rustup toolchain list --verbose'");
        std::process::exit(result.status.code().unwrap());
    }

    let stdout = String::from_utf8(result.stdout).unwrap();
    if !stdout.lines().any(|line| line.trim().starts_with(name)) {
        eprintln!("The '{name}' toolchain could not be found.");
        // eprintln!("To install the risc0 toolchain, use rzup.");
        // eprintln!("For example:");
        // eprintln!("  curl -L https://risczero.com/install | bash");
        // eprintln!("  rzup install");
        std::process::exit(-1);
    }
}
