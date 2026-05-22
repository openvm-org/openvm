//! IR -> CProject -> make -> .so pipeline.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use openvm_instructions::{
    exe::VmExe, program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{CProject, SuspendPolicy, TracerMode};
use rvr_openvm_lift::{
    build_blocks, convert_vmexe_to_ir_with_debug, scan_init_memory_for_code_pointers, AirIndex,
    ExtensionRegistry, TraceChipIndex,
};

use super::debug::GuestDebugMap;
use crate::arch::ExecutorInventory;

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
    #[error("C project write failed under {}: {source}", path.display())]
    CProject {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("make failed: {stderr}")]
    Make { stderr: String },
    #[error("toolchain unavailable: {0}")]
    Toolchain(#[from] rvr_openvm::RuntimeToolchainError),
    #[error("failed to run build tool '{command}': {source}")]
    ToolchainCommand {
        command: String,
        #[source]
        source: std::io::Error,
    },
    #[error("library load failed: {0}")]
    LibLoad(String),
    #[error("unknown opcode {opcode:?} at pc {pc:#x}")]
    UnknownOpcode { pc: u32, opcode: VmOpcode },
    #[error("invalid compile options: {0}")]
    InvalidOptions(&'static str),
}

/// Chip mapping information for hardcoding chip indices into generated code.
#[derive(Clone)]
pub struct ChipMapping {
    /// Per-PC chip index. Index i = chip for PC = pc_base + i*4.
    pub pc_to_chip: Vec<TraceChipIndex>,
    /// Per-AIR widths (MeteredCost mode only). When present, the emitter
    /// precomputes `sum(width[chip] * count)` per block so the generated C
    /// increments `cost` by a single constant instead of loading from the
    /// `chip_widths` array at runtime.
    pub chip_widths: Option<Vec<u64>>,
}

pub fn build_pc_to_chip<F, E>(
    exe: &VmExe<F>,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
) -> Result<Vec<TraceChipIndex>, CompileError>
where
    F: PrimeField32,
{
    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
    exe.program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let Some((inst, _)) = slot else {
                return Ok(TraceChipIndex::NoChip);
            };
            let opcode: VmOpcode = inst.opcode;
            if opcode == terminate_opcode {
                return Ok(TraceChipIndex::NoChip);
            }
            let &executor_idx = inventory.instruction_lookup.get(&opcode).ok_or_else(|| {
                CompileError::UnknownOpcode {
                    pc: exe.program.pc_base + (i as u32) * DEFAULT_PC_STEP,
                    opcode,
                }
            })?;
            Ok(TraceChipIndex::Chip(AirIndex::new(
                executor_idx_to_air_idx[executor_idx as usize] as u32,
            )))
        })
        .collect()
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
    pub suspend_policy: Option<SuspendPolicy>,
}

/// Compile a VmExe into a shared library (pure execution, optional suspension).
pub fn compile<F: PrimeField32>(
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
            suspend_policy: None,
        },
    )
}

/// Compile a VmExe with per-chip metered execution.
pub fn compile_metered<F: PrimeField32>(
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
            suspend_policy: None,
        },
    )
}

/// Compile a VmExe with per-chip metered execution and segment-boundary suspension.
pub fn compile_metered_segment_boundary<F: PrimeField32>(
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
            suspend_policy: Some(SuspendPolicy::SegmentBoundary),
        },
    )
}

/// Compile a VmExe with metered cost tracer.
pub fn compile_metered_cost<F: PrimeField32>(
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
            suspend_policy: None,
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
    let toolchain = ensure_toolchain_available()?;

    let base_name = sanitize_base_name(opts.base_name.unwrap_or("openvm"));

    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;

    let valid_pcs: std::collections::HashSet<u32> = ir.iter().map(|li| li.pc()).collect();
    let extra_targets = scan_init_memory_for_code_pointers(exe, &valid_pcs);
    let blocks = build_blocks(&ir, &extra_targets);

    let temp_root = std::env::temp_dir();
    let temp_dir = tempfile::Builder::new()
        .prefix("openvm-rvr-")
        .tempdir_in(&temp_root)
        .map_err(|source| CompileError::CProject {
            path: temp_root,
            source,
        })?;
    let output_dir = temp_dir.path();

    let mut project = CProject::new(output_dir, &base_name, opts.tracer_mode);
    if let Some(suspend_policy) = opts.suspend_policy {
        project.suspend_policy = suspend_policy;
    }
    match (opts.tracer_mode, project.suspend_policy) {
        (TracerMode::Metered, SuspendPolicy::InstretLimit) => {
            return Err(CompileError::InvalidOptions(
                "metered rvr cannot use instret-limit suspension",
            ));
        }
        (TracerMode::Pure | TracerMode::MeteredCost, SuspendPolicy::SegmentBoundary) => {
            return Err(CompileError::InvalidOptions(
                "segment-boundary suspension requires metered rvr",
            ));
        }
        _ => {}
    }

    match opts.tracer_mode {
        TracerMode::Pure => {}
        TracerMode::Metered | TracerMode::MeteredCost => {
            let chips = opts.chips.ok_or(CompileError::InvalidOptions(
                "metered rvr compile requires ChipMapping",
            ))?;
            project.pc_to_chip = Some(chips.pc_to_chip.clone());
            project.pc_base = exe.program.pc_base;
            project.chip_widths = chips.chip_widths.clone();
        }
    }

    if cfg!(target_os = "macos") {
        project.enable_lto = false;
    }
    project.native_debug_info = opts.native_debug_info;

    let entry_point = exe.pc_start;
    let text_start = exe.program.pc_base;
    project
        .write_all(&blocks, entry_point, text_start, opts.extensions)
        .map_err(|source| CompileError::CProject {
            path: output_dir.to_path_buf(),
            source,
        })?;

    let ext_staticlibs = write_extension_staticlibs(output_dir, opts.extensions)?;
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
                .extra_c_sources()
                .into_iter()
                .map(|(filename, _)| filename.to_string()),
        )
        .collect();
    let ext_cflags = opts.extensions.extra_cflags();

    let make_args =
        project.make_args_with_extensions(&ext_staticlibs, &ext_sources, &ext_cflags);
    compile_generated_project(output_dir, &make_args, &toolchain)?;

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

pub fn ensure_toolchain_available() -> Result<rvr_openvm::RuntimeToolchain, CompileError> {
    Ok(rvr_openvm::runtime_toolchain()?)
}

fn write_extension_staticlibs<F: PrimeField32>(
    output_dir: &Path,
    extensions: &ExtensionRegistry<F>,
) -> Result<Vec<PathBuf>, CompileError> {
    let mut paths = Vec::new();
    for (filename, content) in extensions.staticlib_files() {
        let path = output_dir.join(filename);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| CompileError::CProject {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::write(&path, content).map_err(|source| CompileError::CProject {
            path: path.clone(),
            source,
        })?;
        paths.push(path);
    }
    Ok(paths)
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

fn compile_generated_project(
    output_dir: &Path,
    make_args: &[String],
    toolchain: &rvr_openvm::RuntimeToolchain,
) -> Result<(), CompileError> {
    let stdout_path = output_dir.join("make.stdout.log");
    let stderr_path = output_dir.join("make.stderr.log");
    let stdout_file = File::create(&stdout_path).map_err(|source| CompileError::CProject {
        path: stdout_path.clone(),
        source,
    })?;
    let stderr_file = File::create(&stderr_path).map_err(|source| CompileError::CProject {
        path: stderr_path.clone(),
        source,
    })?;
    let total_objects = count_outputs(output_dir, "c");
    let jobs = std::thread::available_parallelism()
        .map_or(4, |n| n.get().saturating_sub(2).max(1))
        .to_string();
    tracing::debug!(
        translation_units = total_objects,
        make = %toolchain.make,
        jobs = %jobs,
        "building rvr native library"
    );

    let mut child = Command::new(&toolchain.make)
        .arg("-C")
        .arg(output_dir)
        .arg("-j")
        .arg(&jobs)
        .arg("-s")
        .arg("shared")
        .args(make_args)
        .arg(format!("HOST_OS={}", toolchain.host_os))
        .env("CC", &toolchain.compiler)
        .env("LINKER", &toolchain.linker)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .map_err(|source| CompileError::ToolchainCommand {
            command: toolchain.make.clone(),
            source,
        })?;

    let progress_delay = Duration::from_secs(10);
    let progress_interval = Duration::from_secs(10);
    let started_at = Instant::now();
    let mut last_report_at = started_at;
    let mut reported_linking = false;

    loop {
        let done = count_outputs(output_dir, "o");
        let elapsed = started_at.elapsed();
        if elapsed >= progress_delay && done >= total_objects && !reported_linking {
            tracing::debug!(
                objects_done = done,
                objects_total = total_objects,
                elapsed_secs = elapsed.as_secs_f64(),
                "rvr native compile linking"
            );
            last_report_at = Instant::now();
            reported_linking = true;
        } else if elapsed >= progress_delay && last_report_at.elapsed() >= progress_interval {
            tracing::debug!(
                objects_done = done,
                objects_total = total_objects,
                elapsed_secs = elapsed.as_secs_f64(),
                "rvr native compile progress"
            );
            last_report_at = Instant::now();
        }

        if let Some(status) = child
            .try_wait()
            .map_err(|source| CompileError::ToolchainCommand {
                command: toolchain.make.clone(),
                source,
            })?
        {
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
    fs::read_dir(dir)
        .map_err(|source| CompileError::CProject {
            path: dir.to_path_buf(),
            source,
        })?
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
