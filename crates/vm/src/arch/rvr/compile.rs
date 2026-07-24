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
use rvr_openvm::{CProject, RvrExecutionKind};
use rvr_openvm_lift::{
    build_blocks, convert_vmexe_to_ir_with_debug, AirIndex, ExtensionRegistry, TraceChipIndex,
};

use super::debug::GuestDebugMap;
use crate::arch::ExecutorInventory;

/// A compiled rvr shared library ready for execution.
pub struct RvrCompiled {
    /// The loaded shared library.
    pub lib: libloading::Library,
    /// Path to the shared library file backing `lib`.
    lib_path: PathBuf,
    /// Directory holding generated C sources and build artifacts, if this
    /// library was compiled. `None` for libraries loaded from disk.
    artifact_dir: Option<ArtifactDir>,
    /// Generated state, tracing, and block ABI family baked into this artifact.
    execution_kind: RvrExecutionKind,
    /// Number of AIRs stored in a metered artifact.
    num_airs: Option<u32>,
}

enum ArtifactDir {
    Temp(tempfile::TempDir),
    Kept(PathBuf),
}

impl ArtifactDir {
    fn path(&self) -> &Path {
        match self {
            Self::Temp(dir) => dir.path(),
            Self::Kept(path) => path,
        }
    }
}

impl RvrCompiled {
    pub(crate) const fn execution_kind(&self) -> RvrExecutionKind {
        self.execution_kind
    }

    pub(crate) const fn num_airs(&self) -> Option<u32> {
        self.num_airs
    }

    pub(crate) fn require_execution_kind(
        &self,
        expected: &[RvrExecutionKind],
    ) -> Result<(), CompileError> {
        if expected.contains(&self.execution_kind) {
            return Ok(());
        }
        Err(CompileError::LibLoad(format!(
            "RVR execution kind mismatch: expected one of {expected:?}, found {:?}",
            self.execution_kind
        )))
    }

    /// Check that a metered-cost artifact contains the expected chip widths.
    pub(crate) fn require_chip_widths(&self, expected: &[usize]) -> Result<(), CompileError> {
        self.require_execution_kind(&[RvrExecutionKind::MeteredCost])?;
        let num_airs = self.num_airs.ok_or_else(|| {
            CompileError::LibLoad(
                "compiled metered-cost artifact does not declare its AIR count".to_string(),
            )
        })? as usize;
        if expected.len() != num_airs {
            return Err(CompileError::LibLoad(format!(
                "chip-width table has {} entries, but the compiled artifact expects {num_airs}",
                expected.len()
            )));
        }

        type ChipWidthsFn = unsafe extern "C" fn() -> *const u64;
        let chip_widths_fn: ChipWidthsFn = unsafe {
            *self
                .lib
                .get::<ChipWidthsFn>(b"rv_chip_widths")
                .map_err(|error| {
                    CompileError::LibLoad(format!(
                        "missing metered-cost rv_chip_widths marker: {error}"
                    ))
                })?
        };
        // SAFETY: the loaded symbol has the declared zero-argument C ABI.
        let actual_ptr = unsafe { chip_widths_fn() };
        if actual_ptr.is_null() {
            return Err(CompileError::LibLoad(
                "rv_chip_widths marker returned null".to_string(),
            ));
        }
        // SAFETY: the symbol points to `RV_NUM_AIRS` elements, and that count
        // was checked against `expected` above.
        let actual = unsafe { std::slice::from_raw_parts(actual_ptr, num_airs) };
        for (chip_idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let expected = u64::try_from(expected).map_err(|_| {
                CompileError::LibLoad(format!(
                    "chip width at AIR {chip_idx} exceeds the artifact u64 domain"
                ))
            })?;
            if actual != expected {
                return Err(CompileError::LibLoad(format!(
                    "chip width mismatch at AIR {chip_idx}: artifact has {actual}, current VM has {expected}"
                )));
            }
        }
        Ok(())
    }

    /// Path to the directory holding generated C sources and build artifacts,
    /// if this library was compiled (rather than loaded from an existing path).
    /// Valid while the returned [`RvrCompiled`] is alive.
    pub fn artifact_dir(&self) -> Option<&Path> {
        self.artifact_dir.as_ref().map(ArtifactDir::path)
    }

    pub fn lib_file_name_with_suffix(&self, suffix: &str) -> Result<String, CompileError> {
        let stem = self
            .lib_path
            .file_stem()
            .ok_or_else(|| {
                CompileError::LibLoad(format!(
                    "shared library path has no file stem: {}",
                    self.lib_path.display()
                ))
            })?
            .to_string_lossy();
        let ext = self
            .lib_path
            .extension()
            .ok_or_else(|| {
                CompileError::LibLoad(format!(
                    "shared library path has no extension: {}",
                    self.lib_path.display()
                ))
            })?
            .to_string_lossy();
        Ok(if suffix.is_empty() {
            format!("{stem}.{ext}")
        } else {
            format!("{stem}-{suffix}.{ext}")
        })
    }

    /// Copy the compiled shared library into `dest_lib`, creating parent
    /// directories if it doesn't exist. Returns the path of the copied
    /// library. Works for both freshly compiled artifacts and ones loaded from disk.
    pub fn save_artifact(&self, dest_lib: &Path) -> Result<PathBuf, CompileError> {
        if let Some(parent) = dest_lib.parent() {
            fs::create_dir_all(parent).map_err(|source| CompileError::CProject {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::copy(&self.lib_path, dest_lib).map_err(|source| CompileError::CProject {
            path: dest_lib.to_path_buf(),
            source,
        })?;
        Ok(dest_lib.to_path_buf())
    }

    /// Copy generated C sources for inspection.
    pub fn save_generated_sources(&self, dest_dir: &Path) -> Result<(), CompileError> {
        let source_dir = self.artifact_dir().ok_or_else(|| {
            CompileError::LibLoad("loaded rvr artifacts do not contain generated sources".into())
        })?;
        fs::create_dir_all(dest_dir).map_err(|source| CompileError::CProject {
            path: dest_dir.to_path_buf(),
            source,
        })?;
        for entry in fs::read_dir(source_dir).map_err(|source| CompileError::CProject {
            path: source_dir.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| CompileError::CProject {
                path: source_dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            let copy = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "Makefile")
                || path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| matches!(ext, "c" | "h"));
            if copy {
                fs::copy(&path, dest_dir.join(entry.file_name())).map_err(|source| {
                    CompileError::CProject {
                        path: dest_dir.to_path_buf(),
                        source,
                    }
                })?;
            }
        }
        Ok(())
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
    #[error("program counter for instruction index {instruction_index} exceeds the u32 PC domain")]
    ProgramCounterOutOfBounds { instruction_index: usize },
    #[error("executor index {executor_idx} at pc {pc:#x} has no AIR mapping")]
    ExecutorIndexOutOfBounds { pc: u32, executor_idx: u32 },
    #[error("AIR index {air_idx} at pc {pc:#x} exceeds the u32 chip-index domain")]
    AirIndexOutOfBounds { pc: u32, air_idx: usize },
    #[error("AIR count {num_airs} exceeds the generated C u32 domain")]
    AirCountOutOfBounds { num_airs: usize },
    #[error("chip mapping has {actual} entries, but the program has {expected} instruction slots")]
    ChipMappingLengthMismatch { expected: usize, actual: usize },
    #[error("chip index {chip_idx} at pc {pc:#x} is outside the {num_airs} AIRs")]
    ChipIndexOutOfBounds {
        pc: u32,
        chip_idx: u32,
        num_airs: u32,
    },
    #[error("invalid compile options: {0}")]
    InvalidOptions(&'static str),
}

/// Chip mapping information for hardcoding chip indices into generated code.
#[derive(Clone)]
pub struct ChipMapping {
    /// Total number of AIRs in the VM verifying key.
    pub num_airs: usize,
    /// Per-PC chip index. Index i = chip for PC = pc_base + i*4.
    pub pc_to_chip: Vec<TraceChipIndex>,
    /// Width of each AIR in metered-cost mode. The generator writes each width
    /// into its cost update and stores the list in the artifact for validation
    /// when it is loaded. Execution does not look up widths at runtime.
    pub chip_widths: Option<Vec<u64>>,
}

fn pc_for_instruction_index(pc_base: u32, instruction_index: usize) -> Result<u32, CompileError> {
    let instruction_index_u32 = u32::try_from(instruction_index)
        .map_err(|_| CompileError::ProgramCounterOutOfBounds { instruction_index })?;
    instruction_index_u32
        .checked_mul(DEFAULT_PC_STEP)
        .and_then(|offset| pc_base.checked_add(offset))
        .ok_or(CompileError::ProgramCounterOutOfBounds { instruction_index })
}

pub fn build_pc_to_chip<F, E>(
    exe: &VmExe<F>,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
) -> Result<Vec<TraceChipIndex>, CompileError> {
    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
    exe.program
        .instructions_and_debug_infos
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let pc = pc_for_instruction_index(exe.program.pc_base, i)?;
            let Some((inst, _)) = slot else {
                return Ok(TraceChipIndex::NoChip);
            };
            let opcode: VmOpcode = inst.opcode;
            if opcode == terminate_opcode {
                return Ok(TraceChipIndex::NoChip);
            }
            let &executor_idx = inventory
                .instruction_lookup
                .get(&opcode)
                .ok_or(CompileError::UnknownOpcode { pc, opcode })?;
            let &air_idx = executor_idx_to_air_idx
                .get(executor_idx as usize)
                .ok_or(CompileError::ExecutorIndexOutOfBounds { pc, executor_idx })?;
            let air_idx = u32::try_from(air_idx)
                .map_err(|_| CompileError::AirIndexOutOfBounds { pc, air_idx })?;
            Ok(TraceChipIndex::Chip(AirIndex::new(air_idx)))
        })
        .collect()
}

/// Options for the compilation pipeline.
pub struct CompileOptions<'a> {
    /// Base name for generated files and library artifact.
    pub base_name: Option<&'a str>,
    pub execution_kind: RvrExecutionKind,
    pub extensions: &'a ExtensionRegistry,
    pub chips: Option<&'a ChipMapping>,
    /// Guest debug map: OpenVM PC -> SourceLoc.
    pub guest_debug_map: Option<&'a GuestDebugMap>,
    /// Compile with `-g -fno-omit-frame-pointer` for profiling.
    pub native_debug_info: bool,
    /// Compile the generated C with trap-mode UBSan and bounds sanitizers.
    /// Too slow for production proving or profiling runs.
    pub sanitize: bool,
    /// Keep the generated native project after the compiled library is dropped.
    pub keep_artifacts: bool,
}

/// Sanitize in debug/test builds, but never when profiling.
const DEFAULT_SANITIZE: bool = cfg!(debug_assertions) && !cfg!(feature = "profiling");

pub fn compile_with_options<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: CompileOptions<'_>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(exe, &opts)
}

/// Compile a VmExe into a shared library for unlimited pure execution.
pub fn compile<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            execution_kind: RvrExecutionKind::Pure,
            extensions,
            chips: None,
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            sanitize: DEFAULT_SANITIZE,
            keep_artifacts: false,
        },
    )
}

/// Compile a VmExe for pure execution with instret tracking and block-boundary suspension.
pub fn compile_with_instret_tracking<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            execution_kind: RvrExecutionKind::PureWithInstretTracking,
            extensions,
            chips: None,
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            sanitize: DEFAULT_SANITIZE,
            keep_artifacts: false,
        },
    )
}

/// Compile a VmExe with per-chip metered execution.
pub fn compile_metered<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            execution_kind: RvrExecutionKind::Metered,
            extensions,
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            sanitize: DEFAULT_SANITIZE,
            keep_artifacts: false,
        },
    )
}

/// Compile a VmExe with per-chip metered execution and segment-boundary suspension.
pub fn compile_metered_segment_boundary<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            execution_kind: RvrExecutionKind::MeteredSegment,
            extensions,
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            sanitize: DEFAULT_SANITIZE,
            keep_artifacts: false,
        },
    )
}

/// Compile a VmExe with metered-cost tracking.
pub fn compile_metered_cost<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            execution_kind: RvrExecutionKind::MeteredCost,
            extensions,
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            sanitize: DEFAULT_SANITIZE,
            keep_artifacts: false,
        },
    )
}

/// Open a previously saved `.so`/`.dylib` and wrap it in an [`RvrCompiled`].
///
/// The generated execution kind is validated before the artifact can be
/// executed.
/// The caller must still ensure the artifact matches the current `exe`, config,
/// and codebase version.
pub fn load_compiled_from_path(lib_path: &Path) -> Result<RvrCompiled, CompileError> {
    tracing::warn!(
        path = %lib_path.display(),
        "loading rvr artifact with execution-kind validation only; \
         caller is responsible for matching exe, config, and code version"
    );
    let lib = unsafe {
        libloading::Library::new(lib_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", lib_path.display(), e)))?
    };
    let execution_kind = load_execution_kind(&lib)?;
    let num_airs = load_num_airs(&lib, execution_kind)?;
    Ok(RvrCompiled {
        lib,
        lib_path: lib_path.to_path_buf(),
        artifact_dir: None,
        execution_kind,
        num_airs,
    })
}

fn load_execution_kind(lib: &libloading::Library) -> Result<RvrExecutionKind, CompileError> {
    type ExecutionKindFn = unsafe extern "C" fn() -> u32;
    let execution_kind_fn: ExecutionKindFn = unsafe {
        *lib.get::<ExecutionKindFn>(b"rv_execution_kind")
            .map_err(|error| {
                CompileError::LibLoad(format!("missing rv_execution_kind marker: {error}"))
            })?
    };
    let marker = unsafe { execution_kind_fn() };
    RvrExecutionKind::try_from(marker)
        .map_err(|_| CompileError::LibLoad(format!("unknown rv_execution_kind marker {marker}")))
}

fn load_num_airs(
    lib: &libloading::Library,
    execution_kind: RvrExecutionKind,
) -> Result<Option<u32>, CompileError> {
    if matches!(
        execution_kind,
        RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking
    ) {
        return Ok(None);
    }

    type NumAirsFn = unsafe extern "C" fn() -> u32;
    let num_airs_fn: NumAirsFn = unsafe {
        *lib.get::<NumAirsFn>(b"rv_num_airs").map_err(|error| {
            CompileError::LibLoad(format!("missing rv_num_airs marker: {error}"))
        })?
    };
    let num_airs = unsafe { num_airs_fn() };
    if num_airs == 0 {
        return Err(CompileError::LibLoad(
            "rv_num_airs marker must be nonzero".to_string(),
        ));
    }
    Ok(Some(num_airs))
}

fn compile_impl<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: &CompileOptions<'_>,
) -> Result<RvrCompiled, CompileError> {
    let toolchain = ensure_toolchain_available()?;

    let base_name = sanitize_base_name(opts.base_name.unwrap_or("openvm"));

    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;

    let valid_pcs: std::collections::HashSet<u64> = ir.iter().map(|li| li.pc()).collect();
    let extra_targets = opts
        .extensions
        .extra_cfg_targets(&exe.init_memory, &valid_pcs);
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

    let mut project = CProject::new(output_dir, &base_name, opts.execution_kind);
    project.pc_base = u64::from(exe.program.pc_base);

    match opts.execution_kind {
        RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => {}
        RvrExecutionKind::Metered
        | RvrExecutionKind::MeteredSegment
        | RvrExecutionKind::MeteredCost => {
            let chips = opts.chips.ok_or(CompileError::InvalidOptions(
                "metered rvr compile requires ChipMapping",
            ))?;
            let num_airs =
                u32::try_from(chips.num_airs).map_err(|_| CompileError::AirCountOutOfBounds {
                    num_airs: chips.num_airs,
                })?;
            if num_airs == 0 {
                return Err(CompileError::InvalidOptions(
                    "metered rvr compile requires at least one AIR",
                ));
            }
            project.num_airs = Some(num_airs);
            let expected_slots = exe.program.instructions_and_debug_infos.len();
            if chips.pc_to_chip.len() != expected_slots {
                return Err(CompileError::ChipMappingLengthMismatch {
                    expected: expected_slots,
                    actual: chips.pc_to_chip.len(),
                });
            }
            for (slot, chip) in chips.pc_to_chip.iter().copied().enumerate() {
                let TraceChipIndex::Chip(chip) = chip else {
                    continue;
                };
                if chip.as_u32() >= num_airs {
                    return Err(CompileError::ChipIndexOutOfBounds {
                        pc: pc_for_instruction_index(exe.program.pc_base, slot)?,
                        chip_idx: chip.as_u32(),
                        num_airs,
                    });
                }
            }
            project.pc_to_chip = Some(chips.pc_to_chip.clone());
            project.chip_widths = chips.chip_widths.clone();
            match opts.execution_kind {
                RvrExecutionKind::MeteredCost => {
                    let widths =
                        project
                            .chip_widths
                            .as_ref()
                            .ok_or(CompileError::InvalidOptions(
                                "metered-cost rvr compile requires chip widths",
                            ))?;
                    if widths.len() != chips.num_airs {
                        return Err(CompileError::InvalidOptions(
                            "metered-cost chip widths must match the AIR count",
                        ));
                    }
                }
                RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => {
                    if project.chip_widths.is_some() {
                        return Err(CompileError::InvalidOptions(
                            "per-chip metered rvr compile does not use chip widths",
                        ));
                    }
                }
                RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => {
                    unreachable!()
                }
            }
        }
    }

    project.native_debug_info = opts.native_debug_info;
    project.sanitize = opts.sanitize;

    let entry_point = u64::from(exe.pc_start);
    let text_start = u64::from(exe.program.pc_base);
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
        .collect();
    let vendor_sources: Vec<String> = opts
        .extensions
        .vendored_c_sources()
        .into_iter()
        .map(|(filename, _)| filename.to_string())
        .collect();
    let ext_cflags = opts.extensions.extra_cflags();

    compile_generated_project(
        output_dir,
        &project.make_args_with_extensions(
            &ext_staticlibs,
            &ext_sources,
            &vendor_sources,
            &ext_cflags,
        ),
        &toolchain,
    )?;

    let lib_path = find_shared_lib(output_dir)?;
    let lib = unsafe {
        libloading::Library::new(&lib_path)
            .map_err(|e| CompileError::LibLoad(format!("{}: {}", lib_path.display(), e)))?
    };
    let execution_kind = load_execution_kind(&lib)?;
    if execution_kind != opts.execution_kind {
        return Err(CompileError::LibLoad(format!(
            "generated RVR execution kind mismatch: expected {:?}, found {execution_kind:?}",
            opts.execution_kind
        )));
    }
    let num_airs = load_num_airs(&lib, execution_kind)?;
    let artifact_dir = if opts.keep_artifacts {
        let path = temp_dir.keep();
        tracing::info!(
            path = %path.display(),
            "kept rvr generated native project"
        );
        ArtifactDir::Kept(path)
    } else {
        ArtifactDir::Temp(temp_dir)
    };

    Ok(RvrCompiled {
        lib,
        lib_path,
        artifact_dir: Some(artifact_dir),
        execution_kind,
        num_airs,
    })
}

pub fn ensure_toolchain_available() -> Result<rvr_openvm::RuntimeToolchain, CompileError> {
    Ok(rvr_openvm::runtime_toolchain()?)
}

fn write_extension_staticlibs(
    output_dir: &Path,
    extensions: &ExtensionRegistry,
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
        .ok_or_else(|| {
            CompileError::LibLoad(format!(
                "no shared library (.so/.dylib) found in {}",
                dir.display()
            ))
        })
}
