//! IR -> CProject -> make -> .so pipeline.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::{collections::BTreeSet, ffi::OsStr};

use openvm_circuit::arch::execution_mode::metered::segment_ctx::DEFAULT_SEGMENT_CHECK_INSNS;
use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::debug::GuestDebugMap;
use crate::emit::{CProject, TracerMode};
use crate::toolchain;
use rvr_openvm_lift::{
    build_blocks, convert_vmexe_to_ir_with_debug, scan_init_memory_for_code_pointers,
    ExtensionRegistry,
};

/// A compiled rvr shared library ready for execution.
pub struct RvrCompiled {
    /// The loaded shared library.
    pub lib: libloading::Library,
    /// Temporary directory holding the generated C code and .so.
    _temp_dir: Option<tempfile::TempDir>,
}

/// Error during compilation.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("IR conversion failed: {0}")]
    Convert(#[from] rvr_openvm_lift::ConvertError),
    #[error("C project write failed: {0}")]
    CProject(#[from] std::io::Error),
    #[error("cache key generation failed: {0}")]
    CacheKey(String),
    #[error("make failed: {stderr}")]
    Make { stderr: String },
    #[error("toolchain error: {0}")]
    Toolchain(String),
    #[error("library load failed: {0}")]
    LibLoad(String),
}

/// Chip mapping information for hardcoding chip indices into generated code.
#[derive(Clone)]
pub struct ChipMapping {
    /// Per-PC chip index. Index i = chip for PC = pc_base + i*4.
    pub pc_to_chip: Vec<u32>,
    /// Chip index for HINT_BUFFER/HINT_STOREW (None if not present).
    pub hint_store_chip_idx: Option<u32>,
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
    pub cache_dir: Option<&'a Path>,
    /// Guest debug map: OpenVM PC -> SourceLoc.
    pub guest_debug_map: Option<&'a GuestDebugMap>,
    /// Compile with `-g -fno-omit-frame-pointer` for profiling.
    pub native_debug_info: bool,
}

/// Compile with full options.
pub fn compile_with_options<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: &CompileOptions<'_, F>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(exe, opts)
}

/// Compile a VmExe into a shared library (pure execution, optional suspension).
pub fn compile<F: PrimeField32>(exe: &VmExe<F>) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::Pure,
            extensions: &ExtensionRegistry::new(),
            chips: None,
            cache_dir: None,
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

/// Compile a VmExe with inline metered cost tracer enabled.
pub fn compile_metered_cost<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::MeteredCost,
            extensions: &ExtensionRegistry::new(),
            chips: Some(chips),
            cache_dir: None,
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

/// Compile a VmExe with instruction-limit suspension support (same as `compile`).
pub fn compile_with_limit<F: PrimeField32>(exe: &VmExe<F>) -> Result<RvrCompiled, CompileError> {
    compile(exe)
}

/// Compile a VmExe with per-chip metered execution.
pub fn compile_metered<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::Metered,
            extensions: &ExtensionRegistry::new(),
            chips: Some(chips),
            cache_dir: None,
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

/// Compile a VmExe with both metered cost and instruction-limit suspension.
pub fn compile_metered_cost_with_limit<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
) -> Result<RvrCompiled, CompileError> {
    compile_metered_cost(exe, chips)
}

/// Compile a VmExe into a shared library in a persistent cache directory.
///
/// The provided `cache_dir` acts as a cache root. This helper stores builds in
/// a content-addressed subdirectory keyed by the executable and tracer mode.
pub fn compile_cached<F: PrimeField32 + Serialize>(
    exe: &VmExe<F>,
    cache_dir: &Path,
) -> Result<RvrCompiled, CompileError> {
    compile_cached_with_subdir(exe, cache_dir, TracerMode::Pure, None)
}

/// Compile a metered VmExe into a shared library in a persistent cache directory.
///
/// The provided `cache_dir` acts as a cache root. This helper stores builds in
/// a content-addressed subdirectory keyed by the executable, tracer mode, and
/// chip mapping.
pub fn compile_metered_cached<F: PrimeField32 + Serialize>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
    cache_dir: &Path,
) -> Result<RvrCompiled, CompileError> {
    compile_cached_with_subdir(exe, cache_dir, TracerMode::Metered, Some(chips))
}

fn compile_cached_with_subdir<F: PrimeField32 + Serialize>(
    exe: &VmExe<F>,
    cache_root: &Path,
    tracer_mode: TracerMode,
    chips: Option<&ChipMapping>,
) -> Result<RvrCompiled, CompileError> {
    let extensions = ExtensionRegistry::new();
    let cache_dir = cache_root.join(native_cache_key(
        exe,
        &extensions,
        tracer_mode,
        chips,
        false,
    )?);
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode,
            extensions: &extensions,
            chips,
            cache_dir: Some(&cache_dir),
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

pub fn native_cache_key<F: PrimeField32 + Serialize>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    tracer_mode: TracerMode,
    chips: Option<&ChipMapping>,
    debug_info: bool,
) -> Result<String, CompileError> {
    let mut hasher = Sha256::new();
    hasher.update(native_cache_stamp().as_bytes());
    update_extension_native_cache_stamp(&mut hasher, extensions)?;
    hasher.update(match tracer_mode {
        TracerMode::Pure => b"pure".as_slice(),
        TracerMode::MeteredCost => b"metered-cost".as_slice(),
        TracerMode::Metered => b"metered".as_slice(),
    });
    hasher.update([u8::from(debug_info)]);
    hasher.update(
        bincode::serde::encode_to_vec(exe, bincode::config::standard())
            .map_err(|err| CompileError::CacheKey(err.to_string()))?,
    );
    if let Some(chips) = chips {
        hasher.update(
            bincode::serde::encode_to_vec(&chips.pc_to_chip, bincode::config::standard())
                .map_err(|err| CompileError::CacheKey(err.to_string()))?,
        );
        hasher.update(
            bincode::serde::encode_to_vec(chips.hint_store_chip_idx, bincode::config::standard())
                .map_err(|err| CompileError::CacheKey(err.to_string()))?,
        );
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compile a VmExe with extensions into a shared library.
pub fn compile_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::Pure,
            extensions,
            chips: None,
            cache_dir: None,
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

/// Compile a VmExe with extensions and metered cost tracer.
pub fn compile_metered_cost_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    chips: &ChipMapping,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::MeteredCost,
            extensions,
            chips: Some(chips),
            cache_dir: None,
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

/// Compile a VmExe with extensions and per-chip metered execution.
pub fn compile_metered_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    chips: &ChipMapping,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            tracer_mode: TracerMode::Metered,
            extensions,
            chips: Some(chips),
            cache_dir: None,
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

/// Cache stamp for generated native artifacts.
///
/// This is derived from the current native support/runtime sources so cached
/// libraries are invalidated when the generated ABI or native glue changes.
pub fn native_cache_stamp() -> String {
    const DEBUG_INFO_CACHE_VERSION: &str = "debug-info-v1";

    let mut hasher = Sha256::new();
    for source in [
        include_str!("../c/openvm_io.c"),
        include_str!("../c/openvm_io.h"),
        include_str!("../c/Makefile"),
        include_str!("compile.rs"),
        include_str!("execute.rs"),
        include_str!("emit/context.rs"),
        include_str!("emit/project.rs"),
        include_str!("toolchain.rs"),
        DEBUG_INFO_CACHE_VERSION,
    ] {
        hasher.update(source.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn update_extension_native_cache_stamp<F: PrimeField32>(
    hasher: &mut Sha256,
    extensions: &ExtensionRegistry<F>,
) -> Result<(), CompileError> {
    for (filename, content) in extensions.c_headers() {
        hasher.update(b"c_header");
        hasher.update(filename.as_bytes());
        hasher.update(content.as_bytes());
    }
    for (filename, content) in extensions.c_sources() {
        hasher.update(b"c_source");
        hasher.update(filename.as_bytes());
        hasher.update(content.as_bytes());
    }
    for path in extensions.extra_c_source_paths() {
        hasher.update(b"extra_c_source_path");
        hasher.update(path.as_os_str().as_encoded_bytes());
        hasher.update(fs::read(&path).map_err(|err| {
            CompileError::CacheKey(format!("failed to read {}: {err}", path.display()))
        })?);
    }
    let mut include_dirs = BTreeSet::new();
    for flag in extensions.extra_cflags() {
        hasher.update(b"extra_cflag");
        hasher.update(flag.as_bytes());
        if let Some(include_dir) = flag.strip_prefix("-I") {
            include_dirs.insert(PathBuf::from(include_dir));
        }
    }
    let include_dirs: Vec<PathBuf> = include_dirs
        .iter()
        .filter(|dir| !include_dirs_contain_parent(dir, &include_dirs))
        .cloned()
        .collect();
    for include_dir in include_dirs {
        hash_cache_stamp_path(hasher, &include_dir, &include_dir)?;
    }
    for path in extensions.staticlib_paths() {
        hasher.update(b"staticlib");
        hasher.update(path.as_os_str().as_encoded_bytes());
        hasher.update(fs::read(path).map_err(|err| {
            CompileError::CacheKey(format!("failed to read {}: {err}", path.display()))
        })?);
    }
    Ok(())
}

fn include_dirs_contain_parent(dir: &Path, include_dirs: &BTreeSet<PathBuf>) -> bool {
    include_dirs
        .iter()
        .any(|other| other != dir && dir.starts_with(other))
}

fn hash_cache_stamp_path(
    hasher: &mut Sha256,
    root: &Path,
    path: &Path,
) -> Result<(), CompileError> {
    let metadata = fs::metadata(path).map_err(|err| {
        CompileError::CacheKey(format!("failed to stat {}: {err}", path.display()))
    })?;
    let rel = path.strip_prefix(root).unwrap_or(Path::new(""));

    if metadata.is_dir() {
        hasher.update(b"dir");
        hasher.update(rel.as_os_str().as_encoded_bytes());
        let mut entries = fs::read_dir(path)
            .map_err(|err| {
                CompileError::CacheKey(format!(
                    "failed to read directory {}: {err}",
                    path.display()
                ))
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                CompileError::CacheKey(format!(
                    "failed to iterate directory {}: {err}",
                    path.display()
                ))
            })?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            hash_cache_stamp_path(hasher, root, &entry.path())?;
        }
    } else if matches!(path.extension().and_then(OsStr::to_str), Some("c" | "h")) {
        hasher.update(b"file");
        hasher.update(rel.as_os_str().as_encoded_bytes());
        hasher.update(fs::read(path).map_err(|err| {
            CompileError::CacheKey(format!("failed to read {}: {err}", path.display()))
        })?);
    }

    Ok(())
}

pub fn load_compiled_from_path(lib_path: &Path) -> Result<RvrCompiled, CompileError> {
    let lib = unsafe {
        libloading::Library::new(lib_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", lib_path.display(), e)))?
    };
    Ok(RvrCompiled {
        lib,
        _temp_dir: None,
    })
}

fn compile_impl<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: &CompileOptions<'_, F>,
) -> Result<RvrCompiled, CompileError> {
    let base_name = sanitize_base_name(opts.base_name.unwrap_or("openvm"));

    if let Some(cache_dir) = opts.cache_dir {
        if let Ok(lib_path) = find_shared_lib(cache_dir) {
            if let Some(compiled) = load_cached_library_if_compatible(&lib_path)? {
                return Ok(compiled);
            }
        }

        if cache_dir.exists() {
            fs::remove_dir_all(cache_dir)?;
        }
        fs::create_dir_all(cache_dir)?;
    }

    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;

    let valid_pcs: std::collections::HashSet<u32> = ir.iter().map(|li| li.pc()).collect();
    let extra_targets = scan_init_memory_for_code_pointers(exe, &valid_pcs);
    let blocks = build_blocks(&ir, &extra_targets, DEFAULT_SEGMENT_CHECK_INSNS as u32);

    let temp_dir = if opts.cache_dir.is_none() {
        Some(tempfile::tempdir()?)
    } else {
        None
    };
    let output_dir = opts
        .cache_dir
        .unwrap_or_else(|| temp_dir.as_ref().unwrap().path());

    let memory_bits = openvm_platform::memory::MEM_BITS as u8;
    let mut project = CProject::new(output_dir, &base_name, opts.tracer_mode, memory_bits);

    if let Some(chips) = opts.chips {
        project.pc_to_chip = Some(chips.pc_to_chip.clone());
        project.pc_base = exe.program.pc_base;
        project.hint_store_chip_idx = chips.hint_store_chip_idx.unwrap_or(u32::MAX);
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
    let lib = unsafe {
        libloading::Library::new(&lib_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", lib_path.display(), e)))?
    };

    Ok(RvrCompiled {
        lib,
        _temp_dir: temp_dir,
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

fn load_cached_library_if_compatible(lib_path: &Path) -> Result<Option<RvrCompiled>, CompileError> {
    type ExecuteFn = unsafe extern "C" fn(*mut std::ffi::c_void);
    type RegisterFn = unsafe extern "C" fn(*const std::ffi::c_void);
    // `compile_cached*` use a content-addressed subdirectory rooted at the
    // caller-provided cache path. Raw `CompileOptions.cache_dir` remains a
    // lower-level escape hatch and is assumed to be managed by the caller.
    // This compatibility check only guards against stale libraries with the
    // wrong exported ABI.

    let lib = unsafe {
        libloading::Library::new(lib_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", lib_path.display(), e)))?
    };

    let has_execute = unsafe { lib.get::<ExecuteFn>(b"rv_execute").is_ok() };
    let has_register = unsafe { lib.get::<RegisterFn>(b"register_openvm_callbacks").is_ok() };

    if has_execute && has_register {
        Ok(Some(RvrCompiled {
            lib,
            _temp_dir: None,
        }))
    } else {
        drop(lib);
        Ok(None)
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
    let linker = toolchain::default_linker_or_lld();

    eprintln!(
        "[rvr-openvm] Building native library: {total_objects} translation units with make -j{jobs}"
    );

    if !toolchain::linker_exists(&linker) {
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
        .env("CC", toolchain::default_compiler_command())
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

fn find_shared_lib(dir: &std::path::Path) -> Result<PathBuf, CompileError> {
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use openvm_instructions::{exe::VmExe, instruction::Instruction};
    use openvm_stark_sdk::p3_baby_bear::BabyBear;
    use rvr_openvm_ir::LiftedInstr;
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;

    use super::{native_cache_key, CompileError, TracerMode};
    use crate::ChipMapping;
    use rvr_openvm_lift::{ExtensionRegistry, RvrExtension};

    struct MockExtension {
        header_content: String,
        source_content: String,
        extra_cflags: Vec<String>,
        extra_source_path: PathBuf,
        staticlib_path: PathBuf,
    }

    struct MockNativeInputs<'a> {
        header_content: &'a str,
        source_content: &'a str,
        extra_source_body: &'a str,
        include_dir_in_cflags: bool,
        extra_cflags: &'a [&'a str],
        staticlib_body: &'a str,
        include_body: &'a str,
    }

    impl RvrExtension<BabyBear> for MockExtension {
        fn try_lift(&self, _insn: &Instruction<BabyBear>, _pc: u32) -> Option<LiftedInstr> {
            None
        }

        fn c_headers(&self) -> Vec<(&str, &str)> {
            vec![("mock_ext.h", self.header_content.as_str())]
        }

        fn c_sources(&self) -> Vec<(&str, &str)> {
            vec![("mock_ext.c", self.source_content.as_str())]
        }

        fn staticlib_path(&self) -> &Path {
            &self.staticlib_path
        }

        fn extra_c_source_paths(&self) -> Vec<PathBuf> {
            vec![self.extra_source_path.clone()]
        }

        fn extra_cflags(&self) -> Vec<String> {
            self.extra_cflags.clone()
        }
    }

    fn mock_registry(dir: &TempDir, inputs: MockNativeInputs<'_>) -> ExtensionRegistry<BabyBear> {
        let suffix = format!("{:x}", Sha256::digest(inputs.header_content.as_bytes()));
        let extra_source_path = dir.path().join(format!("extra_{suffix}.c"));
        let staticlib_path = dir.path().join(format!("libmock_{suffix}.a"));
        let include_dir = dir.path().join(format!("include_{suffix}"));
        let include_path = include_dir.join("stamp.h");
        fs::write(&extra_source_path, inputs.extra_source_body).unwrap();
        fs::write(&staticlib_path, inputs.staticlib_body).unwrap();
        fs::create_dir_all(&include_dir).unwrap();
        fs::write(&include_path, inputs.include_body).unwrap();

        let mut resolved_cflags = Vec::new();
        if inputs.include_dir_in_cflags {
            resolved_cflags.push(format!("-I{}", include_dir.display()));
        }
        resolved_cflags.extend(inputs.extra_cflags.iter().map(|s| s.to_string()));

        let mut registry = ExtensionRegistry::new();
        registry.register(MockExtension {
            header_content: inputs.header_content.to_string(),
            source_content: inputs.source_content.to_string(),
            extra_cflags: resolved_cflags,
            extra_source_path,
            staticlib_path,
        });
        registry
    }

    fn cache_key(
        exe: &VmExe<BabyBear>,
        registry: &ExtensionRegistry<BabyBear>,
    ) -> Result<String, CompileError> {
        native_cache_key(exe, registry, TracerMode::Pure, None::<&ChipMapping>, false)
    }

    #[test]
    fn native_cache_key_tracks_extension_native_inputs() {
        let exe = VmExe::<BabyBear>::default();
        let dir = tempfile::tempdir().unwrap();
        let base = mock_registry(
            &dir,
            MockNativeInputs {
                header_content: "/* header */",
                source_content: "/* source */",
                extra_source_body: "int extra(void) { return 1; }\n",
                include_dir_in_cflags: true,
                extra_cflags: &["-Dbase"],
                staticlib_body: "staticlib-a",
                include_body: "/* include */",
            },
        );
        let base_key = cache_key(&exe, &base).unwrap();

        let variants = [
            mock_registry(
                &dir,
                MockNativeInputs {
                    header_content: "/* header changed */",
                    source_content: "/* source */",
                    extra_source_body: "int extra(void) { return 1; }\n",
                    include_dir_in_cflags: true,
                    extra_cflags: &["-Dbase"],
                    staticlib_body: "staticlib-a",
                    include_body: "/* include */",
                },
            ),
            mock_registry(
                &dir,
                MockNativeInputs {
                    header_content: "/* header */",
                    source_content: "/* source changed */",
                    extra_source_body: "int extra(void) { return 1; }\n",
                    include_dir_in_cflags: true,
                    extra_cflags: &["-Dbase"],
                    staticlib_body: "staticlib-a",
                    include_body: "/* include */",
                },
            ),
            mock_registry(
                &dir,
                MockNativeInputs {
                    header_content: "/* header */",
                    source_content: "/* source */",
                    extra_source_body: "int extra(void) { return 2; }\n",
                    include_dir_in_cflags: true,
                    extra_cflags: &["-Dbase"],
                    staticlib_body: "staticlib-a",
                    include_body: "/* include */",
                },
            ),
            mock_registry(
                &dir,
                MockNativeInputs {
                    header_content: "/* header */",
                    source_content: "/* source */",
                    extra_source_body: "int extra(void) { return 1; }\n",
                    include_dir_in_cflags: true,
                    extra_cflags: &["-Dchanged"],
                    staticlib_body: "staticlib-a",
                    include_body: "/* include */",
                },
            ),
            mock_registry(
                &dir,
                MockNativeInputs {
                    header_content: "/* header */",
                    source_content: "/* source */",
                    extra_source_body: "int extra(void) { return 1; }\n",
                    include_dir_in_cflags: true,
                    extra_cflags: &["-Dbase"],
                    staticlib_body: "staticlib-b",
                    include_body: "/* include */",
                },
            ),
            mock_registry(
                &dir,
                MockNativeInputs {
                    header_content: "/* header */",
                    source_content: "/* source */",
                    extra_source_body: "int extra(void) { return 1; }\n",
                    include_dir_in_cflags: true,
                    extra_cflags: &["-Dbase"],
                    staticlib_body: "staticlib-a",
                    include_body: "/* include changed */",
                },
            ),
        ];

        for variant in variants {
            assert_ne!(base_key, cache_key(&exe, &variant).unwrap());
        }
    }
}
