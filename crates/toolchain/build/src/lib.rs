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

use cargo_metadata::{Metadata, MetadataCommand, Package};
use openvm_platform::memory;

pub use self::config::GuestOptions;

mod config;

/// Default rustc target for the openvm guest. Built-in to the openvm rust
/// fork. Override with `OPENVM_RUSTC_TARGET` to use a different rv64im target.
pub const DEFAULT_RUSTC_TARGET: &str = "riscv64im-unknown-openvm-elf";

/// Default rustup toolchain name. Format: `openvm-<rustup-name>` —
/// `openvm-` prefix plus whatever rustup itself calls the upstream
/// toolchain (`X.Y.Z` for stable, `nightly-YYYY-MM-DD` for nightly,
/// `beta-YYYY-MM-DD` for beta). Defaults to a stable OpenVM release for
/// predictability; published nightly variants can be installed explicitly
/// with `--version openvm-nightly-…`.
/// Override with `OPENVM_RUST_TOOLCHAIN`.
//
// Keep in sync with the default `TAG` in `ci/install-openvm-toolchain.sh`.
pub const DEFAULT_RUSTUP_TOOLCHAIN_NAME: &str = "openvm-1.94.1";

const BUILD_LOCKED_ENV: &str = "OPENVM_BUILD_LOCKED";
const SKIP_BUILD_ENV: &str = "OPENVM_SKIP_BUILD";
const GUEST_LOGFILE_ENV: &str = "OPENVM_GUEST_LOGFILE";
const ALLOWED_CARGO_ENVS: &[&str] = &["CARGO_HOME"];

/// Get the target triple from environment variable or default.
pub fn get_rustc_target() -> String {
    env::var("OPENVM_RUSTC_TARGET").unwrap_or_else(|_| DEFAULT_RUSTC_TARGET.to_string())
}

/// Get the Rust toolchain name from environment variable or default.
pub fn get_rustup_toolchain_name() -> String {
    env::var("OPENVM_RUST_TOOLCHAIN").unwrap_or_else(|_| DEFAULT_RUSTUP_TOOLCHAIN_NAME.to_string())
}

/// Returns `true` for toolchains installed by `cargo openvm toolchain install`.
pub fn is_openvm_toolchain(toolchain: &str) -> bool {
    toolchain.starts_with("openvm-")
}

/// Returns `true` if the toolchain's sysroot contains prebuilt rlibs for
/// `target`.
pub fn target_has_prebuilt_std(rustc_path: &str, target: &str) -> bool {
    Path::new(rustc_path)
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("lib/rustlib").join(target).join("lib").exists())
        .unwrap_or(false)
}

/// Returns the given cargo Package from the metadata in the Cargo.toml manifest
/// within the provided `manifest_dir`.
pub fn get_package(manifest_dir: impl AsRef<Path>) -> Package {
    let manifest_path = manifest_dir
        .as_ref()
        .join("Cargo.toml")
        .canonicalize()
        .unwrap();
    let manifest_meta = get_metadata(&manifest_path);
    let matching = find_matching_packages(&manifest_meta, &manifest_path);

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
    matching.into_iter().next().unwrap()
}

/// Returns all packages from the Cargo.toml manifest at the given `manifest_dir`.
pub fn get_workspace_packages(manifest_dir: impl AsRef<Path>) -> Vec<Package> {
    let manifest_path = manifest_dir
        .as_ref()
        .join("Cargo.toml")
        .canonicalize()
        .unwrap();
    let manifest_meta = get_metadata(&manifest_path);
    get_workspace_member_packages(manifest_meta)
}

/// Returns a single package if the manifest path matches exactly, otherwise returns all
/// workspace packages.
pub fn get_in_scope_packages(manifest_dir: impl AsRef<Path>) -> Vec<Package> {
    let manifest_path = manifest_dir
        .as_ref()
        .join("Cargo.toml")
        .canonicalize()
        .unwrap();
    let manifest_meta = get_metadata(&manifest_path);

    // Check if any package has this exact manifest path
    let matching = find_matching_packages(&manifest_meta, &manifest_path);

    // If we found a package with this exact manifest path, return it
    if !matching.is_empty() {
        return matching;
    }

    // Otherwise return all workspace members
    get_workspace_member_packages(manifest_meta)
}

/// Helper function to get cargo metadata for a manifest path
fn get_metadata(manifest_path: &Path) -> Metadata {
    MetadataCommand::new()
        .manifest_path(manifest_path)
        .no_deps()
        .exec()
        .unwrap_or_else(|e| {
            panic!(
                "cargo metadata command failed for manifest path: {}: {e:?}",
                manifest_path.display()
            )
        })
}

/// Helper function to get workspace members
fn get_workspace_member_packages(manifest_meta: Metadata) -> Vec<Package> {
    manifest_meta
        .packages
        .into_iter()
        .filter(|pkg| manifest_meta.workspace_members.contains(&pkg.id))
        .collect()
}

/// Helper function to find packages matching a manifest path
fn find_matching_packages(manifest_meta: &Metadata, manifest_path: &Path) -> Vec<Package> {
    manifest_meta
        .packages
        .iter()
        .filter(|pkg| {
            let std_path: &Path = pkg.manifest_path.as_ref();
            std_path == manifest_path
        })
        .cloned()
        .collect()
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

/// Returns the workspace root directory from the Cargo manifest at
/// the given `manifest_path`.
pub fn get_workspace_root(manifest_path: impl AsRef<Path>) -> PathBuf {
    MetadataCommand::new()
        .manifest_path(manifest_path.as_ref())
        .no_deps()
        .exec()
        .expect("cargo metadata command failed")
        .workspace_root
        .into()
}

/// Returns the target executable directory given `target_dir` and `profile`.
pub fn get_dir_with_profile(
    target_dir: impl AsRef<Path>,
    profile: &str,
    examples: bool,
) -> PathBuf {
    let mut res = target_dir.as_ref().join(get_rustc_target()).to_path_buf();
    if profile == "dev" || profile == "test" {
        res.push("debug");
    } else if profile == "bench" {
        res.push("release");
    } else {
        res.push(profile);
    }
    if examples {
        res.join("examples")
    } else {
        res
    }
}

/// When called from a build.rs, returns the current package being built.
pub fn current_package() -> Package {
    get_package(env::var("CARGO_MANIFEST_DIR").unwrap())
}

/// Reads the value of the environment variable `OPENVM_SKIP_BUILD` and returns true if it is set to
/// 1.
pub fn is_skip_build() -> bool {
    !get_env_var(SKIP_BUILD_ENV).is_empty()
}

fn get_env_var(name: &str) -> String {
    // Cargo directives are meaningful only while this library is running in a
    // build script. Emitting them from `cargo openvm execute` pollutes the
    // command's machine-readable stdout.
    if env::var_os("OUT_DIR").is_some() {
        println!("cargo:rerun-if-env-changed={name}");
    }
    env::var(name).unwrap_or_default()
}

/// Returns all target ELF paths associated with the given guest crate.
pub fn guest_methods<S: AsRef<str>>(
    pkg: &Package,
    target_dir: impl AsRef<Path>,
    guest_features: &[String],
    profile: &Option<S>,
) -> Vec<PathBuf> {
    let profile = profile.as_ref().map(|s| s.as_ref()).unwrap_or("release");
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
        .flat_map(|target| {
            let path_prefix = target_dir.as_ref().join(get_rustc_target()).join(profile);
            target
                .kind
                .iter()
                .map(|target_kind| {
                    let mut path = path_prefix.clone();
                    if target_kind == "example" {
                        path.push(target_kind);
                    }
                    path.join(&target.name).to_path_buf()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Build a [Command] with CARGO and RUSTUP_TOOLCHAIN environment variables
/// removed.
fn sanitized_cmd(tool: &str) -> Command {
    let mut cmd = Command::new(tool);
    for (key, _val) in env::vars()
        .filter(|x| x.0.starts_with("CARGO") && !ALLOWED_CARGO_ENVS.contains(&x.0.as_str()))
    {
        cmd.env_remove(key);
    }
    cmd.env_remove("RUSTUP_TOOLCHAIN");
    cmd
}

/// Creates a std::process::Command to execute the given cargo
/// command in an environment suitable for targeting the zkvm guest.
pub fn cargo_command(subcmd: &str, rust_flags: &[&str]) -> Command {
    let toolchain = format!("+{}", get_rustup_toolchain_name());
    let target = get_rustc_target();

    let rustc = sanitized_cmd("rustup")
        .args([&toolchain, "which", "rustc"])
        .output()
        .expect("rustup failed to find openvm toolchain")
        .stdout;

    let rustc = String::from_utf8(rustc).unwrap();
    let rustc = rustc.trim();
    eprintln!("Using rustc: {rustc}");

    let mut cmd = sanitized_cmd("cargo");
    let mut args = vec![toolchain.as_str(), subcmd, "--target", target.as_str()];

    if std::env::var(BUILD_LOCKED_ENV).is_ok() {
        args.push("--locked");
    }

    // Upstream tier-3 targets do not ship target rlibs, so build std from rust-src.
    if !target_has_prebuilt_std(rustc, &target) {
        args.extend_from_slice(&[
            "-Z",
            "build-std=core,alloc,panic_abort",
            "-Z",
            "build-std-features=compiler-builtins-mem",
        ]);
    }

    eprintln!("Building guest package: cargo {}", args.join(" "));

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
            // https://docs.rs/getrandom/0.3.2/getrandom/index.html#opt-in-backends
            "--cfg",
            "getrandom_backend=\"custom\"",
            // Set by `cargo openvm build`; guest crates gate code on
            // cfg(any(openvm_intrinsics, target_os = "openvm")) to switch between portable Rust
            // impls and openvm-intrinsic-using impls.
            "--cfg",
            "openvm_intrinsics",
            "--check-cfg=cfg(openvm_intrinsics)",
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
    let tty_file = env::var(GUEST_LOGFILE_ENV).unwrap_or_else(|_| "/dev/tty".to_string());

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
    target_filter: &Option<TargetFilter>,
) -> Result<PathBuf, Option<i32>> {
    let mut new_opts = guest_opts.clone();

    if new_opts.target_dir.is_none() {
        new_opts.target_dir = Some(get_target_dir(&pkg.manifest_path));
    }

    new_opts.options.extend(vec![
        "--manifest-path".into(),
        pkg.manifest_path.to_string(),
    ]);

    if let Some(runtime_lib) = runtime_lib {
        new_opts.rustc_flags.extend(vec![
            String::from("-C"),
            format!("link_arg={}", runtime_lib),
        ]);
    }

    let mut example = false;
    if let Some(target_filter) = target_filter {
        new_opts.options.extend(vec![
            format!("--{}", target_filter.kind),
            target_filter.name.clone(),
        ]);
        example = target_filter.kind == "example";
    }

    let res = build_generic(&new_opts);
    res.map(|path| if example { path.join("examples") } else { path })
}

/// Generic wrapper call to cargo build
pub fn build_generic(guest_opts: &GuestOptions) -> Result<PathBuf, Option<i32>> {
    if is_skip_build() || guest_opts.target_dir.is_none() {
        eprintln!("Skipping build");
        return Err(None);
    }

    // Verify the openvm toolchain is installed (linked via rustup).
    let toolchain_name = get_rustup_toolchain_name();
    if let Err(code) = ensure_openvm_toolchain_linked(&toolchain_name) {
        return Err(Some(code));
    }

    let target_dir = guest_opts.target_dir.as_ref().unwrap();
    fs::create_dir_all(target_dir).unwrap();
    let rust_flags: Vec<_> = guest_opts.rustc_flags.iter().map(|s| s.as_str()).collect();

    let mut cmd = cargo_command("build", &rust_flags);
    cmd.envs(guest_opts.env.iter().map(|(key, value)| (key, value)));

    if !guest_opts.features.is_empty() {
        cmd.args(["--features", guest_opts.features.join(",").as_str()]);
    }
    cmd.args(["--target-dir", target_dir.to_str().unwrap()]);

    let profile = if let Some(profile) = &guest_opts.profile {
        profile
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
        "openvm build: Starting build for {}",
        get_rustc_target()
    ));

    for line in BufReader::new(stderr).lines() {
        tty_println(&format!("openvm build: {}", line.unwrap()));
    }

    let res = child.wait().expect("Guest 'cargo build' failed");
    if !res.success() {
        Err(res.code())
    } else {
        Ok(get_dir_with_profile(target_dir, profile, false))
    }
}

/// A filter for selecting a target from a package.
#[derive(Default)]
pub struct TargetFilter {
    /// The target name to match.
    pub name: String,
    /// The kind of target to match.
    pub kind: String,
}

/// Finds the unique executable target in the given package and target directory,
/// using the given target filter.
pub fn find_unique_executable<P: AsRef<Path>, Q: AsRef<Path>>(
    pkg_dir: P,
    target_dir: Q,
    target_filter: &Option<TargetFilter>,
) -> eyre::Result<PathBuf> {
    let pkg = get_package(pkg_dir.as_ref());
    let elf_paths = pkg
        .targets
        .into_iter()
        .filter(move |target| {
            // always filter out build script target
            if target.is_custom_build() || target.is_lib() {
                return false;
            }
            if let Some(target_filter) = target_filter {
                return target.kind.iter().any(|k| k == &target_filter.kind)
                    && target.name == target_filter.name;
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
        std::process::exit(-1);
    }
}

/// Verify that the configured rustup toolchain can build the guest target.
fn ensure_openvm_toolchain_linked(toolchain: &str) -> Result<(), i32> {
    let is_openvm = is_openvm_toolchain(toolchain);

    let output = Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .map_err(|e| {
            tty_println(&format!("Failed to check toolchains: {e}"));
            e.raw_os_error().unwrap_or(1)
        })?;

    let installed = String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line.trim().starts_with(toolchain));

    if !installed {
        let hint = if is_openvm {
            "cargo openvm toolchain install".to_string()
        } else {
            format!("rustup toolchain install {toolchain}")
        };
        tty_println(&format!(
            "error: rustup toolchain `{toolchain}` is not installed.\n\nInstall it with:\n\n    {hint}\n"
        ));
        return Err(1);
    }

    // Resolve the sysroot to check for either prebuilt target rlibs or rust-src.
    let which_rustc = Command::new("rustup")
        .args([&format!("+{toolchain}"), "which", "rustc"])
        .output()
        .map_err(|e| {
            tty_println(&format!("Failed to resolve {toolchain} rustc: {e}"));
            e.raw_os_error().unwrap_or(1)
        })?;
    if !which_rustc.status.success() {
        return Err(which_rustc.status.code().unwrap_or(1));
    }
    let rustc_path = String::from_utf8_lossy(&which_rustc.stdout)
        .trim()
        .to_string();
    let toolchain_root = Path::new(&rustc_path)
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);

    if let Some(root) = toolchain_root {
        let target = get_rustc_target();

        if target_has_prebuilt_std(&rustc_path, &target) {
            return Ok(());
        }

        if is_openvm {
            tty_println(&format!(
                "error: OpenVM toolchain `{toolchain}` does not contain prebuilt rlibs for target \
                 `{target}`.\n\nFix with:\n\n    cargo openvm toolchain install --force\n",
            ));
            return Err(1);
        }

        let rustc_version = Command::new("rustup")
            .args(["run", toolchain, "rustc", "--version"])
            .output()
            .map_err(|e| {
                tty_println(&format!("Failed to check {toolchain} rustc version: {e}"));
                e.raw_os_error().unwrap_or(1)
            })?;
        if !rustc_version.status.success() {
            return Err(rustc_version.status.code().unwrap_or(1));
        }
        let rustc_version = String::from_utf8_lossy(&rustc_version.stdout);
        if !rustc_version.contains("nightly") {
            tty_println(&format!(
                "error: target `{target}` has no prebuilt rlibs in toolchain `{toolchain}`. \
                 Building it from `rust-src` uses Cargo's `-Z build-std`, which requires a nightly \
                 Rust toolchain.\n\nUse a nightly toolchain, for example:\n\n    rustup toolchain install nightly --component rust-src\n    OPENVM_RUST_TOOLCHAIN=nightly OPENVM_RUSTC_TARGET={target} cargo openvm build\n",
            ));
            return Err(1);
        }

        let rust_src_dir = root.join("lib/rustlib/src/rust/library");
        if !rust_src_dir.exists() {
            let hint = format!(
                "rustup component add rust-src --toolchain {toolchain}\n\
                 (target `{target}` has no prebuilt rlibs in rustup; the build will use -Z build-std)"
            );
            tty_println(&format!(
                "error: toolchain `{toolchain}` ships neither prebuilt rlibs for target `{target}` \
                 nor the `rust-src` component needed for -Z build-std.\n\nFix with:\n\n    {hint}\n",
            ));
            return Err(1);
        }
    }

    Ok(())
}
