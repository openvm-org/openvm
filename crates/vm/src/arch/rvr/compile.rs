//! IR -> CProject -> make -> .so pipeline.

use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{CProject, TracerMode};
use rvr_openvm_lift::{
    build_blocks, convert_vmexe_to_ir_with_debug, scan_init_memory_for_code_pointers,
    ExtensionRegistry,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::debug::GuestDebugMap;

/// A compiled rvr shared library ready for execution.
pub struct RvrCompiled {
    /// The loaded shared library.
    pub lib: libloading::Library,
    /// Temporary directory holding the generated C code and .so.
    temp_dir: Option<tempfile::TempDir>,
}

impl RvrCompiled {
    /// Path to the directory holding generated C sources and build artifacts,
    /// if this library was compiled (rather than loaded from an existing path).
    /// Valid while the returned [`RvrCompiled`] is alive.
    pub fn artifact_dir(&self) -> Option<&Path> {
        self.temp_dir.as_ref().map(|d| d.path())
    }
}

/// Error during compilation.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("IR conversion failed: {0}")]
    Convert(#[from] rvr_openvm_lift::ConvertError),
    #[error("C project write failed: {0}")]
    CProject(#[from] std::io::Error),
    #[error("make failed: {stderr}")]
    Make { stderr: String },
    #[error("toolchain error: {0}")]
    Toolchain(String),
    #[error("library load failed: {0}")]
    LibLoad(String),
}

/// Chip mapping information for hardcoding chip indices into generated code.
#[derive(Clone, Serialize)]
pub struct ChipMapping {
    /// Per-PC chip index. Index i = chip for PC = pc_base + i*4.
    pub pc_to_chip: Vec<u32>,
    /// Per-AIR widths (MeteredCost mode only). When present, the emitter
    /// precomputes `sum(width[chip] * count)` per block so the generated C
    /// increments `cost` by a single constant instead of loading from the
    /// `chip_widths` array at runtime.
    pub chip_widths: Option<Vec<u64>>,
}

/// Options for the compilation pipeline.
pub struct CompileOptions<'a, F: PrimeField32> {
    /// Base name for generated files and library artifact.
    pub base_name: Option<&'a str>,
    pub tracer_mode: TracerMode,
    pub extensions: &'a ExtensionRegistry<F>,
    pub chips: Option<&'a ChipMapping>,
    /// Guest debug map: OpenVM PC -> SourceLoc.
    pub guest_debug_map: Option<&'a GuestDebugMap>,
    /// Compile with `-g -fno-omit-frame-pointer` for profiling.
    pub native_debug_info: bool,
    /// Optional directory for caching compiled `.so` artifacts. The filename within the
    /// directory is derived from a hash of the inputs that affect codegen (exe, `config_bytes`,
    /// tracer mode, chip mapping, target triple, compile options, OpenVM version) plus the
    /// content of every external static library the extensions link against. If a file at the
    /// derived path already exists it is loaded directly; otherwise the compiled library is
    /// copied into the cache before loading.
    pub cache_dir: Option<&'a Path>,
    /// Stable, serialized fingerprint of the [`VmConfig`](crate::arch::VmConfig) that produced
    /// `extensions`. Mixed into the cache key. Required when `cache_dir` is `Some`; ignored
    /// otherwise. The caller is responsible for picking a deterministic serializer (e.g.
    /// `bitcode::serialize`). This is the cache key's only signal of "which extensions and how
    /// they're parameterized" — combined with `OPENVM_VERSION` (also in the key) it captures
    /// every input that affects in-binary codegen.
    pub config_bytes: Option<&'a [u8]>,
}

/// Compile a VmExe into a shared library (pure execution, optional suspension).
///
/// When `cache_dir` is `Some`, `config_bytes` should also be supplied with a deterministic
/// serialization of the `VmConfig` that produced `extensions`; see
/// [`CompileOptions::config_bytes`].
pub fn compile<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    cache_dir: Option<&Path>,
    config_bytes: Option<&[u8]>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::Pure,
            extensions,
            chips: None,
            guest_debug_map: None,
            native_debug_info: false,
            cache_dir,
            config_bytes,
        },
    )
}

/// Compile a VmExe with per-chip metered execution. See [`compile`] for `cache_dir` /
/// `config_bytes` semantics.
pub fn compile_metered<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    chips: &ChipMapping,
    cache_dir: Option<&Path>,
    config_bytes: Option<&[u8]>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::Metered,
            extensions,
            chips: Some(chips),
            guest_debug_map: None,
            native_debug_info: false,
            cache_dir,
            config_bytes,
        },
    )
}

/// Compile a VmExe with metered cost tracer. See [`compile`] for `cache_dir` / `config_bytes`
/// semantics.
pub fn compile_metered_cost<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    chips: &ChipMapping,
    cache_dir: Option<&Path>,
    config_bytes: Option<&[u8]>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::MeteredCost,
            extensions,
            chips: Some(chips),
            guest_debug_map: None,
            native_debug_info: false,
            cache_dir,
            config_bytes,
        },
    )
}

pub fn load_compiled_from_path(lib_path: &Path) -> Result<RvrCompiled, CompileError> {
    let lib = unsafe {
        libloading::Library::new(lib_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", lib_path.display(), e)))?
    };
    Ok(RvrCompiled {
        lib,
        temp_dir: None,
    })
}

fn compile_impl<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: &CompileOptions<'_, F>,
) -> Result<RvrCompiled, CompileError> {
    let base_name = sanitize_base_name(opts.base_name.unwrap_or("openvm"));

    let cache_path = opts.cache_dir.map(|dir| {
        let key = compute_cache_key(exe, opts);
        // `DLL_SUFFIX` is `.so` on Linux, `.dylib` on macOS, `.dll` on Windows.
        dir.join(format!("{base_name}-{key}{}", std::env::consts::DLL_SUFFIX))
    });
    if let Some(path) = &cache_path {
        if path.is_file() {
            tracing::info!("rvr cache hit: {}", path.display());
            return load_compiled_from_path(path);
        }
    }

    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;

    let valid_pcs: std::collections::HashSet<u32> = ir.iter().map(|li| li.pc()).collect();
    let extra_targets = scan_init_memory_for_code_pointers(exe, &valid_pcs);
    let blocks = build_blocks(&ir, &extra_targets);

    let temp_dir = tempfile::tempdir()?;
    let output_dir = temp_dir.path();

    let mut project = CProject::new(output_dir, &base_name, opts.tracer_mode);

    if let Some(chips) = opts.chips {
        project.pc_to_chip = Some(chips.pc_to_chip.clone());
        project.pc_base = exe.program.pc_base;
        project.chip_widths = chips.chip_widths.clone();
    }

    if cfg!(target_os = "macos") {
        project.enable_lto = false;
    }
    project.native_debug_info = opts.native_debug_info;

    let entry_point = exe.pc_start;
    let text_start = exe.program.pc_base;
    project.write_all(&blocks, entry_point, text_start, opts.extensions)?;

    let ext_staticlibs = opts.extensions.staticlib_paths();
    if let Some(path) = ext_staticlibs
        .iter()
        .find(|path| path.to_string_lossy().contains(' '))
    {
        return Err(CompileError::Make {
            stderr: format!(
                "static library path contains spaces and cannot be passed through make EXT_LIBS safely: {}",
                path.display()
            ),
        });
    }
    let ext_sources: Vec<String> = opts
        .extensions
        .c_sources()
        .into_iter()
        .map(|(filename, _)| filename.to_string())
        .chain(
            opts.extensions
                .extra_c_source_paths()
                .into_iter()
                .map(|path| {
                    path.file_name()
                        .expect("extra C source path missing file name")
                        .to_string_lossy()
                        .into_owned()
                }),
        )
        .collect();
    let ext_cflags = opts.extensions.extra_cflags();

    compile_generated_project(
        output_dir,
        &project.make_args_with_extensions(&ext_staticlibs, &ext_sources, &ext_cflags),
    )?;

    let lib_path = find_shared_lib(output_dir)?;
    let load_path = if let Some(cache_path) = cache_path.as_ref() {
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Copy into the cache, then load from the cache. The build temp dir is dropped on
        // function exit and would otherwise take the .so with it.
        fs::copy(&lib_path, cache_path)?;
        cache_path.as_path()
    } else {
        lib_path.as_path()
    };
    let lib = unsafe {
        libloading::Library::new(load_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", load_path.display(), e)))?
    };

    Ok(RvrCompiled {
        lib,
        // When caching, the .so lives at `cache_path`, so drop the build directory.
        temp_dir: cache_path.is_none().then_some(temp_dir),
    })
}

fn sanitize_base_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "openvm".to_string()
    } else {
        out
    }
}

fn compile_generated_project(output_dir: &Path, make_args: &[String]) -> Result<(), CompileError> {
    let stdout_path = output_dir.join("make.stdout.log");
    let stderr_path = output_dir.join("make.stderr.log");
    let stdout_file = File::create(&stdout_path)?;
    let stderr_file = File::create(&stderr_path)?;
    let total_objects = count_outputs(output_dir, "c");
    let jobs = std::thread::available_parallelism()
        .map_or(4, |n| n.get().saturating_sub(2).max(1))
        .to_string();
    let linker = rvr_openvm::default_linker_or_lld();

    eprintln!(
        "[rvr-openvm] Building native library: {total_objects} translation units with make -j{jobs}"
    );

    if !rvr_openvm::linker_exists(&linker) {
        return Err(CompileError::Toolchain(format!(
            "required linker '{linker}' not found in PATH; install lld or set RVR_LD/LD"
        )));
    }

    let mut child = Command::new("make")
        .arg("-C")
        .arg(output_dir)
        .arg("-j")
        .arg(&jobs)
        .arg("-s")
        .arg("shared")
        .args(make_args)
        .env("CC", rvr_openvm::default_compiler_command())
        .env("LINKER", linker)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()?;

    let progress_delay = Duration::from_secs(10);
    let progress_interval = Duration::from_secs(10);
    let started_at = Instant::now();
    let mut last_report_at = started_at;
    let mut reported_linking = false;

    loop {
        let done = count_outputs(output_dir, "o");
        let elapsed = started_at.elapsed();
        if elapsed >= progress_delay && done >= total_objects && !reported_linking {
            eprintln!(
                "[rvr-openvm] Native compile progress: {done}/{total_objects} object files built; linking ({:.0}s elapsed)",
                elapsed.as_secs_f64()
            );
            last_report_at = Instant::now();
            reported_linking = true;
        } else if elapsed >= progress_delay && last_report_at.elapsed() >= progress_interval {
            eprintln!(
                "[rvr-openvm] Native compile progress: {done}/{total_objects} object files built ({:.0}s elapsed)",
                elapsed.as_secs_f64()
            );
            last_report_at = Instant::now();
        }

        if let Some(status) = child.try_wait()? {
            if status.success() {
                return Ok(());
            }

            return Err(CompileError::Make {
                stderr: read_make_failure(&stdout_path, &stderr_path),
            });
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn count_outputs(dir: &Path, ext: &str) -> usize {
    fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some(ext))
        .count()
}

fn read_make_failure(stdout_path: &Path, stderr_path: &Path) -> String {
    let stdout = fs::read_to_string(stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(stderr_path).unwrap_or_default();
    let mut message = String::new();

    if !stderr.trim().is_empty() {
        message.push_str(&stderr);
    }
    if !stdout.trim().is_empty() {
        if !message.is_empty() {
            message.push_str("\n\n");
        }
        message.push_str("stdout:\n");
        message.push_str(&stdout);
    }

    if message.is_empty() {
        "make failed without emitting output".to_string()
    } else {
        message
    }
}

fn find_shared_lib(dir: &Path) -> Result<PathBuf, CompileError> {
    fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("so" | "dylib")
            )
        })
        .ok_or_else(|| CompileError::Make {
            stderr: "No shared library found after make".to_string(),
        })
}

// ── Cache key computation ────────────────────────────────────────────────────

/// Everything that affects in-binary codegen, flattened into one `Serialize` blob. We feed
/// `bitcode::serialize` of this through SHA-256 instead of walking each field by hand.
///
/// What's *not* here: extension static-library file contents — those live on disk and can change
/// without any of these fields changing, so we hash them separately in [`compute_cache_key`].
#[derive(Serialize)]
struct CacheKeyInputs<'a, F> {
    openvm_version: &'static str,
    os: &'static str,
    arch: &'static str,
    tracer_tag: u8,
    native_debug_info: bool,
    exe: &'a VmExe<F>,
    chips: Option<&'a ChipMapping>,
    config_bytes: Option<&'a [u8]>,
}

/// Compute a hex-encoded cache key whose value changes whenever the inputs to codegen change.
fn compute_cache_key<F: PrimeField32>(exe: &VmExe<F>, opts: &CompileOptions<'_, F>) -> String {
    let inputs = CacheKeyInputs {
        openvm_version: env!("CARGO_PKG_VERSION"),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        tracer_tag: match opts.tracer_mode {
            TracerMode::Pure => 0,
            TracerMode::MeteredCost => 1,
            TracerMode::Metered => 2,
        },
        native_debug_info: opts.native_debug_info,
        exe,
        chips: opts.chips,
        config_bytes: opts.config_bytes,
    };

    let serialized = bitcode::serialize(&inputs).expect("CacheKeyInputs is always serializable");
    let mut hasher = Sha256::new();
    hasher.update(&serialized);
    hash_extension_staticlibs(&mut hasher, opts.extensions);

    let digest = hasher.finalize();
    // 16 bytes = 32 hex chars: ample for collision avoidance in a per-user cache.
    hex::encode(&digest[..16])
}

/// Hash the file contents of every external static library and extra-C-source the extensions
/// link against. These are the only inputs the cache key cannot derive from `config_bytes` +
/// `OPENVM_VERSION`, because they're produced by separate builds and can change without bumping
/// the openvm version.
fn hash_extension_staticlibs<F: PrimeField32>(
    hasher: &mut Sha256,
    extensions: &ExtensionRegistry<F>,
) {
    for path in extensions.staticlib_paths() {
        hash_file_for_cache_key(hasher, path);
    }
    for path in &extensions.extra_c_source_paths() {
        hash_file_for_cache_key(hasher, path);
    }
}

/// Hash a file's full byte contents into the cache key. Falls back to hashing just the path
/// string when the file cannot be read — that mirrors the "rebuild on next change" behavior
/// callers would expect if the file is regenerated.
fn hash_file_for_cache_key(hasher: &mut Sha256, path: &Path) {
    let path_bytes = path.as_os_str().as_encoded_bytes();
    hasher.update((path_bytes.len() as u64).to_le_bytes());
    hasher.update(path_bytes);
    match File::open(path).and_then(|mut f| {
        let mut content_hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            content_hasher.update(&buf[..n]);
        }
        Ok::<_, io::Error>(content_hasher.finalize())
    }) {
        Ok(digest) => {
            hasher.update([1u8]);
            hasher.update(digest);
        }
        Err(_) => hasher.update([0u8]),
    }
}

