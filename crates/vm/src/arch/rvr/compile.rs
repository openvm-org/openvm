//! IR -> CProject -> make -> .so pipeline.

use std::{
    fs::{self, File},
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

use super::debug::GuestDebugMap;
use crate::{
    arch::execution_mode::metered::{
        ctx::DEFAULT_PAGE_BITS, segment_ctx::DEFAULT_SEGMENT_CHECK_INSNS,
    },
    system::memory::{merkle::public_values::PUBLIC_VALUES_AS, CHUNK},
};

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
            guest_debug_map: None,
            native_debug_info: false,
        },
    )
}

fn openvm_compile_constants() -> (u32, u32, u32, u32) {
    (
        DEFAULT_SEGMENT_CHECK_INSNS as u32,
        DEFAULT_PAGE_BITS as u32,
        CHUNK.ilog2(),
        PUBLIC_VALUES_AS,
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

    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;

    let (segment_check_insns, page_bits, chunk_bits, public_values_as) = openvm_compile_constants();
    let valid_pcs: std::collections::HashSet<u32> = ir.iter().map(|li| li.pc()).collect();
    let extra_targets = scan_init_memory_for_code_pointers(exe, &valid_pcs);
    let blocks = build_blocks(&ir, &extra_targets, segment_check_insns);

    let temp_dir = tempfile::tempdir()?;
    let output_dir = temp_dir.path();

    let memory_bits = openvm_platform::memory::MEM_BITS as u8;
    let mut project = CProject::new(
        output_dir,
        &base_name,
        opts.tracer_mode,
        memory_bits,
        page_bits,
        chunk_bits,
        segment_check_insns,
        public_values_as,
    );

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
        temp_dir: Some(temp_dir),
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
