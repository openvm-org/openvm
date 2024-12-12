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

use cargo_metadata::{MetadataCommand, Package};
use openvm_platform::memory;

pub use self::config::GuestOptions;

mod config;

#[allow(dead_code)]
const RUSTUP_TOOLCHAIN_NAME: &str = "nightly-2024-10-30";

/// Returns the given cargo Package from the metadata in the Cargo.toml manifest
/// within the provided `manifest_dir`.
pub fn get_package(manifest_dir: impl AsRef<Path>) -> Package {
    let manifest_path = fs::canonicalize(manifest_dir.as_ref().join("Cargo.toml")).unwrap();
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

/// Returns the target executable directory given `target_dir` and `profile`.
pub fn get_dir_with_profile(target_dir: impl AsRef<Path>, profile: &str) -> PathBuf {
    target_dir
        .as_ref()
        .join("riscv32im-risc0-zkvm-elf")
        .join(profile)
}

/// When called from a build.rs, returns the current package being built.
pub fn current_package() -> Package {
    get_package(env::var("CARGO_MANIFEST_DIR").unwrap())
}

/// Reads the value of the environment variable `OPENVM_BUILD_DEBUG` and returns true if it is set to 1.
pub fn is_debug() -> bool {
    get_env_var("OPENVM_BUILD_DEBUG") == "1"
}

/// Reads the value of the environment variable `OPENVM_SKIP_BUILD` and returns true if it is set to 1.
pub fn is_skip_build() -> bool {
    !get_env_var("OPENVM_SKIP_BUILD").is_empty()
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
        .filter(|target| {
            target
                .kind
                .iter()
                .any(|kind| kind == "bin" || kind == "example")
        })
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
        .args(["+nightly-2024-10-30", "which", "rustc"]) // TODO: switch +nightly to +openvm once we make a toolchain
        .output()
        .expect("rustup failed to find nightly toolchain")
        .stdout;

    let rustc = String::from_utf8(rustc).unwrap();
    let rustc = rustc.trim();
    println!("Using rustc: {rustc}");

    let mut cmd = sanitized_cmd("cargo");
    // TODO[jpw]: remove +nightly
    let mut args = vec![
        "+nightly-2024-10-30",
        subcmd,
        "--target",
        "riscv32im-risc0-zkvm-elf",
    ];

    if std::env::var("OPENVM_BUILD_LOCKED").is_ok() {
        args.push("--locked");
    }

    // let rust_src = get_env_var("OPENVM_RUST_SRC");
    // if !rust_src.is_empty() {
    // TODO[jpw]: only do this for custom src once we make openvm toolchain
    args.push("-Z");
    args.push("build-std=alloc,core,proc_macro,panic_abort,std");
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
            "passes=lower-atomic",
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
    let tty_file = env::var("OPENVM_GUEST_LOGFILE").unwrap_or_else(|_| "/dev/tty".to_string());

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
pub fn build_guest_package(
    pkg: &Package,
    guest_opts: &GuestOptions,
    runtime_lib: Option<&str>,
) -> Result<PathBuf, Option<i32>> {
    if is_skip_build() {
        return Err(None);
    }

    let target_dir = guest_opts
        .target_dir
        .clone()
        .unwrap_or_else(|| get_target_dir(pkg.manifest_path.clone()));

    fs::create_dir_all(&target_dir).unwrap();

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
        target_dir.to_str().unwrap(),
    ]);

    let profile = if let Some(profile) = &guest_opts.profile {
        profile
    } else if is_debug() {
        "dev"
    } else {
        "release"
    };
    cmd.args(["--profile", profile]);

    cmd.args(&guest_opts.options);

    let command_string = format!(
        "{} {}",
        cmd.get_program().to_string_lossy(),
        cmd.get_args()
            .map(|arg| arg.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    );
    tty_println(&format!("cargo command: {command_string}"));

    let mut child = cmd
        .stderr(Stdio::piped())
        .env("CARGO_TERM_COLOR", "always")
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
        Err(res.code())
    } else {
        Ok(get_dir_with_profile(&target_dir, profile))
    }
}

/// A filter for selecting a target from a package.
#[derive(Default)]
pub struct TargetFilter {
    /// A substring of the target name to match.
    pub name_substr: Option<String>,
    /// The kind of target to match.
    pub kind: Option<String>,
}

/// Finds the unique executable target in the given package and target directory,
/// using the given target filter.
pub fn find_unique_executable<P: AsRef<Path>, Q: AsRef<Path>>(
    pkg_dir: P,
    target_dir: Q,
    target_filter: &TargetFilter,
) -> eyre::Result<PathBuf> {
    let pkg = get_package(pkg_dir.as_ref());
    let elf_paths = pkg
        .targets
        .into_iter()
        .filter(move |target| {
            if let Some(name_substr) = &target_filter.name_substr {
                if !target.name.contains(name_substr) {
                    return false;
                }
            }
            if let Some(kind) = &target_filter.kind {
                if !target.kind.iter().any(|k| k == kind) {
                    return false;
                }
            }
            true
        })
        .collect::<Vec<_>>();
    if elf_paths.len() != 1 {
        Err(eyre::eyre!(
            "Expected 1 target, got {}: {:#?}",
            elf_paths.len(),
            elf_paths
        ))
    } else {
        Ok(target_dir.as_ref().join(&elf_paths[0].name))
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
