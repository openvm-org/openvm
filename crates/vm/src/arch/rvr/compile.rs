//! IR -> CProject -> make -> .so pipeline.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use openvm_instructions::{
    exe::VmExe, program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{
    inline_record_shape_for_instr, inline_record_shape_for_terminator, CProject, InlineRecordShape,
    RvrExecutionKind,
};
use rvr_openvm_ir::Block;
use rvr_openvm_lift::{
    build_blocks, convert_vmexe_to_ir_with_debug, opcode::lift_instruction,
    scan_init_memory_for_code_pointers, AirIndex, ExtensionRegistry, RvrInstruction,
    TraceChipIndex,
};
use sha2::{Digest, Sha256};

use super::{debug::GuestDebugMap, ArenaNativeGeometry, LogNativeOpcodeAdmitter};
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
    /// R3 inline-record metadata for preflight libraries; empty otherwise.
    inline_records: RvrInlineRecordsMeta,
}

/// Which instructions a compiled preflight library migrated to inline compact
/// records. The host must mirror the codegen exactly: skip the log assembler
/// for flagged pc slots (their memory-log entries are suppressed) and hand
/// each listed chip's C-written record buffer to the record arena directly.
#[derive(Clone, Debug, Default)]
pub struct RvrInlineRecordsMeta {
    /// Per program slot (`(pc - pc_base) / DEFAULT_PC_STEP`): true when the
    /// instruction emits an inline compact record.
    pub pc_slots: Arc<Vec<bool>>,
    /// `(air_idx, record_size_bytes)` per chip receiving inline records,
    /// sorted by `air_idx`.
    pub airs: Vec<(usize, usize)>,
    /// R4: `(air_idx, geometry)` for the subset of `airs` whose family has an
    /// arena-native emitter (full records at final arena positions), sorted
    /// by `air_idx`. Geometry comes from the assembler registry, which the
    /// owning extension populated from the real record types.
    pub arena_native_airs: Vec<(usize, ArenaNativeGeometry)>,
}

impl RvrInlineRecordsMeta {
    pub fn is_empty(&self) -> bool {
        self.airs.is_empty()
    }
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

    /// Path to the directory holding generated C sources and build artifacts,
    /// if this library was compiled (rather than loaded from an existing path).
    /// Valid while the returned [`RvrCompiled`] is alive.
    pub fn artifact_dir(&self) -> Option<&Path> {
        self.artifact_dir.as_ref().map(ArtifactDir::path)
    }

    /// R3 inline-record metadata for a preflight library (empty for other
    /// tracer modes and for libraries loaded without compile metadata).
    pub fn inline_records(&self) -> &RvrInlineRecordsMeta {
        &self.inline_records
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
    #[error("CFG construction failed: {0}")]
    Cfg(#[from] rvr_openvm_lift::CfgError),
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
    #[error(
        "rvr preflight has no registered log-native assembler for opcode {opcode:?} at pc {pc:#x}; requires config routing"
    )]
    PreflightExtensionOpcode { pc: u32, opcode: VmOpcode },
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

/// Opcode classification for rvr log-native preflight routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RvrPreflightOpcodeClass {
    Supported,
    Unsupported { pc: u32, opcode: VmOpcode },
}

impl RvrPreflightOpcodeClass {
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }
}

/// Classify whether the executable can be lifted by the base RV64IM/system
/// rvr lifter without any registered extensions.
pub fn classify_preflight_opcodes<F: PrimeField32>(exe: &VmExe<F>) -> RvrPreflightOpcodeClass {
    let base_only = ExtensionRegistry::new();
    classify_preflight_opcodes_with_extensions(exe, &base_only, &())
}

/// Classify whether every instruction can be lifted and has a log-native
/// assembler whenever lifting requires a registered extension.
pub fn classify_preflight_opcodes_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    assembler_admitter: &dyn LogNativeOpcodeAdmitter<F>,
) -> RvrPreflightOpcodeClass {
    let base_only = ExtensionRegistry::new();
    for (pc, insn, _) in exe.program.enumerate_by_pc() {
        let rvr_insn = RvrInstruction::from_field(&insn);
        if lift_instruction(&rvr_insn, u64::from(pc), &base_only).is_none() {
            if assembler_admitter.has_log_native_assembler(&insn)
                && lift_instruction(&rvr_insn, u64::from(pc), extensions).is_some()
            {
                continue;
            }
            return RvrPreflightOpcodeClass::Unsupported {
                pc,
                opcode: insn.opcode,
            };
        }
    }
    RvrPreflightOpcodeClass::Supported
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

/// Collect which instructions the preflight codegen migrates to inline compact
/// records: the emitter's per-instruction decision is
/// [`instr_emits_inline_record`] on the lifted IR instruction plus a
/// `TraceChipIndex::Chip` mapping for its pc (mirroring
/// `CProject::chip_idx_for_pc`). Walks the same lifted blocks the emitter
/// walks so the host metadata cannot drift from the generated C.
fn collect_inline_records_meta<F: PrimeField32>(
    exe: &VmExe<F>,
    blocks: &[Block],
    chips: &ChipMapping,
    admitter: Option<&dyn LogNativeOpcodeAdmitter<F>>,
) -> RvrInlineRecordsMeta {
    let num_slots = exe.program.instructions_and_debug_infos.len();
    let pc_base = u64::from(exe.program.pc_base);
    let mut pc_slots = vec![false; num_slots];
    let mut airs: BTreeMap<usize, usize> = BTreeMap::new();
    let mut arena_native: BTreeMap<usize, Option<ArenaNativeGeometry>> = BTreeMap::new();
    // R4 gate, DEFAULT ON (OPENVM_RVR_ARENA_NATIVE=0 opts out for A/B
    // measurement). The decision MUST be made here, in the metadata the host
    // and codegen both consume: gating only the codegen copy leaves the host
    // staging arena targets against a compact-compiled library — the C then
    // writes compact records at the arena row stride and the substituted
    // rows are garbage (a real bus-imbalance bug caught by the prove
    // fixtures when this gate briefly lived on the codegen side only).
    let arena_native_enabled = std::env::var("OPENVM_RVR_ARENA_NATIVE").as_deref() != Ok("0");
    let mut record = |pc: u64, shape: InlineRecordShape| {
        let Some(offset) = pc.checked_sub(pc_base) else {
            return;
        };
        let slot = (offset / u64::from(DEFAULT_PC_STEP)) as usize;
        let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) else {
            return;
        };
        if let Some(flag) = pc_slots.get_mut(slot) {
            *flag = true;
        }
        let size = inline_record_shape_size(shape);
        let air_idx = air.as_u32() as usize;
        let previous = airs.insert(air_idx, size);
        assert!(
            previous.is_none_or(|p| p == size),
            "conflicting inline record sizes for air {air:?}: {previous:?} vs {size}"
        );
        // R4: every pc routed to one air must agree on the family's
        // arena-native geometry (present with equal values, or absent).
        let geometry = admitter
            .filter(|_| arena_native_enabled)
            .and_then(|admitter| {
                exe.program
                    .instructions_and_debug_infos
                    .get(slot)
                    .and_then(|entry| entry.as_ref())
                    .and_then(|(instruction, _)| admitter.inline_arena_geometry_for(instruction))
            });
        let previous = arena_native.insert(air_idx, geometry);
        assert!(
            previous.is_none_or(|p| p == geometry),
            "conflicting arena-native geometry for air {air:?}: {previous:?} vs {geometry:?}"
        );
    };
    for block in blocks {
        for instr_at in &block.instructions {
            if let Some(shape) = inline_record_shape_for_instr(&instr_at.instr) {
                record(instr_at.pc, shape);
            }
        }
        if let Some(shape) = inline_record_shape_for_terminator(&block.terminator) {
            record(block.terminator_pc, shape);
        }
    }
    RvrInlineRecordsMeta {
        pc_slots: Arc::new(pc_slots),
        airs: airs.into_iter().collect(),
        arena_native_airs: arena_native
            .into_iter()
            .filter_map(|(air, geometry)| geometry.map(|g| (air, g)))
            .collect(),
    }
}

/// Compact record stride per wire shape (the C-side `_Static_assert`s guard
/// the layouts against these same constants).
fn inline_record_shape_size(shape: InlineRecordShape) -> usize {
    match shape {
        InlineRecordShape::Alu3 => rvr_openvm_ext_ffi_common::PREFLIGHT_ADDSUB_RECORD_SIZE,
        InlineRecordShape::Branch2 => rvr_openvm_ext_ffi_common::PREFLIGHT_BRANCH2_RECORD_SIZE,
        InlineRecordShape::Wr1 => rvr_openvm_ext_ffi_common::PREFLIGHT_WR1_RECORD_SIZE,
        InlineRecordShape::Rw1 => rvr_openvm_ext_ffi_common::PREFLIGHT_RW1_RECORD_SIZE,
    }
}

/// Options for the compilation pipeline.
pub struct CompileOptions<'a, F> {
    /// Base name for generated files and library artifact.
    pub base_name: Option<&'a str>,
    pub execution_kind: RvrExecutionKind,
    pub extensions: &'a ExtensionRegistry,
    /// Opcode admitter for extension-lifted preflight instructions.
    pub preflight_assembler_admitter: Option<&'a dyn LogNativeOpcodeAdmitter<F>>,
    pub chips: Option<&'a ChipMapping>,
    /// Guest debug map: OpenVM PC -> SourceLoc.
    pub guest_debug_map: Option<&'a GuestDebugMap>,
    /// Compile with `-g -fno-omit-frame-pointer` for profiling.
    pub native_debug_info: bool,
    /// Keep the generated native project after the compiled library is dropped.
    pub keep_artifacts: bool,
}

pub fn compile_with_options<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: CompileOptions<'_, F>,
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
            preflight_assembler_admitter: None,
            chips: None,
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
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
            preflight_assembler_admitter: None,
            chips: None,
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
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
            preflight_assembler_admitter: None,
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
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
            preflight_assembler_admitter: None,
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
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
            preflight_assembler_admitter: None,
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            keep_artifacts: false,
        },
    )
}

/// Compile a base-RV64IM VmExe with the preflight tracer.
pub fn compile_preflight<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    let extensions = ExtensionRegistry::new();
    compile_preflight_with_extensions(exe, &extensions, &(), chips, guest_debug_map)
}

/// Compile a VmExe with the preflight tracer and extension assemblers.
pub fn compile_preflight_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    assembler_admitter: &dyn LogNativeOpcodeAdmitter<F>,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    #[cfg(any(test, feature = "test-utils"))]
    PREFLIGHT_COMPILE_INVOCATIONS.with(|count| count.set(count.get() + 1));
    compile_impl(
        exe,
        &CompileOptions {
            base_name: None,
            execution_kind: RvrExecutionKind::Preflight,
            extensions,
            preflight_assembler_admitter: Some(assembler_admitter),
            chips: Some(chips),
            guest_debug_map,
            native_debug_info: cfg!(feature = "profiling"),
            keep_artifacts: false,
        },
    )
}

#[cfg(any(test, feature = "test-utils"))]
thread_local! {
    static PREFLIGHT_COMPILE_INVOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Resets the preflight compiler invocation count for the current test thread.
#[cfg(any(test, feature = "test-utils"))]
pub fn reset_preflight_compile_invocations_for_test() {
    PREFLIGHT_COMPILE_INVOCATIONS.with(|count| count.set(0));
}

/// Returns the preflight compiler invocation count for the current test thread.
#[cfg(any(test, feature = "test-utils"))]
pub fn preflight_compile_invocations_for_test() -> usize {
    PREFLIGHT_COMPILE_INVOCATIONS.with(std::cell::Cell::get)
}

/// Open a previously saved `.so`/`.dylib` and wrap it in an [`RvrCompiled`].
///
/// The generated execution kind is validated before the artifact can be
/// executed.
/// Inline-record metadata is not recoverable from a bare library, so
/// this path is only valid for tracer modes without inline records (pure /
/// metered); a preflight library compiled with inline records would suppress
/// memory-log entries the host then expects, failing loudly at record
/// assembly.
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
    Ok(RvrCompiled {
        lib,
        lib_path: lib_path.to_path_buf(),
        artifact_dir: None,
        execution_kind,
        inline_records: RvrInlineRecordsMeta::default(),
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

fn compile_impl<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: &CompileOptions<'_, F>,
) -> Result<RvrCompiled, CompileError> {
    if opts.execution_kind == RvrExecutionKind::Preflight {
        let no_extension_assemblers = ();
        let assembler_admitter = opts
            .preflight_assembler_admitter
            .unwrap_or(&no_extension_assemblers);
        if let RvrPreflightOpcodeClass::Unsupported { pc, opcode } =
            classify_preflight_opcodes_with_extensions(exe, opts.extensions, assembler_admitter)
        {
            return Err(CompileError::PreflightExtensionOpcode { pc, opcode });
        }
    }

    let toolchain = ensure_toolchain_available()?;

    let base_name = sanitize_base_name(opts.base_name.unwrap_or("openvm"));

    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;

    let valid_pcs: std::collections::HashSet<u64> = ir.iter().map(|li| li.pc()).collect();
    let extra_targets = scan_init_memory_for_code_pointers(&exe.init_memory, &valid_pcs);
    let blocks = build_blocks(&ir, &extra_targets)?;

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
    let lto_env = match opts.execution_kind {
        RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => {
            Some("OPENVM_RVR_METERED_LTO")
        }
        RvrExecutionKind::Preflight => Some("OPENVM_RVR_PREFLIGHT_LTO"),
        RvrExecutionKind::Pure
        | RvrExecutionKind::PureWithInstretTracking
        | RvrExecutionKind::MeteredCost => None,
    };
    let disable_lto = lto_env.is_some_and(env_flag_is_off);
    if disable_lto {
        project.enable_lto = false;
        tracing::info!(
            execution_kind = ?opts.execution_kind,
            "disabled ThinLTO for rvr native compilation"
        );
    }
    project.pc_base = u64::from(exe.program.pc_base);

    // Preflight compiles emit inline compact records for migrated opcodes by default.
    let inline_records = opts.execution_kind == RvrExecutionKind::Preflight
        && !env_flag_is_off("OPENVM_RVR_INLINE_RECORDS");
    let mut inline_meta = RvrInlineRecordsMeta::default();
    match opts.execution_kind {
        RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => {}
        RvrExecutionKind::Metered
        | RvrExecutionKind::MeteredSegment
        | RvrExecutionKind::MeteredCost
        | RvrExecutionKind::Preflight => {
            let chips = opts.chips.ok_or(CompileError::InvalidOptions(
                "metered/preflight rvr compile requires ChipMapping",
            ))?;
            project.pc_to_chip = Some(chips.pc_to_chip.clone());
            if matches!(
                opts.execution_kind,
                RvrExecutionKind::Metered
                    | RvrExecutionKind::MeteredSegment
                    | RvrExecutionKind::MeteredCost
            ) {
                project.chip_widths = chips.chip_widths.clone();
            }
            if inline_records {
                project.inline_records = true;
                inline_meta = collect_inline_records_meta(
                    exe,
                    &blocks,
                    chips,
                    opts.preflight_assembler_admitter,
                );
                project.arena_native_airs = inline_meta
                    .arena_native_airs
                    .iter()
                    .map(|&(air, geometry)| (air as u32, geometry))
                    .collect();
            }
        }
    }

    project.native_debug_info = opts.native_debug_info;

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
        .chain(
            opts.extensions
                .extra_c_sources()
                .into_iter()
                .map(|(filename, _)| filename.to_string()),
        )
        .collect();
    let ext_cflags = opts.extensions.extra_cflags();

    let mut make_args =
        project.make_args_with_extensions(&ext_staticlibs, &ext_sources, &ext_cflags);
    if disable_lto {
        // Override an inherited Make environment variable as well as CProject's default.
        make_args.push("LTO=".to_string());
    }
    let opt_env = match opts.execution_kind {
        RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => Some((
            "OPENVM_RVR_METERED_OPT",
            "OPENVM_RVR_METERED_OPT must be one of -O0, -O1, -O2, -O3, -Os, or -Oz",
        )),
        RvrExecutionKind::Preflight => Some((
            "OPENVM_RVR_PREFLIGHT_OPT",
            "OPENVM_RVR_PREFLIGHT_OPT must be one of -O0, -O1, -O2, -O3, -Os, or -Oz",
        )),
        RvrExecutionKind::Pure
        | RvrExecutionKind::PureWithInstretTracking
        | RvrExecutionKind::MeteredCost => None,
    };
    if let Some((env, invalid_message)) = opt_env {
        if let Some(opt) = native_opt_level(env, invalid_message)? {
            make_args.push(format!("OPT={opt}"));
        }
    }
    let cache_env = match opts.execution_kind {
        RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => {
            Some("OPENVM_RVR_METERED_LIB")
        }
        RvrExecutionKind::Preflight => Some("OPENVM_RVR_PREFLIGHT_LIB"),
        RvrExecutionKind::Pure
        | RvrExecutionKind::PureWithInstretTracking
        | RvrExecutionKind::MeteredCost => None,
    };
    let cache = if let Some(cache_env) = cache_env {
        std::env::var_os(cache_env)
            .filter(|path| !path.is_empty())
            .map(|path| {
                let lib_path = PathBuf::from(path);
                let key_path = preflight_cache_key_path(&lib_path);
                let key = generated_project_cache_key(output_dir, &make_args, &toolchain)?;
                Ok::<_, CompileError>((lib_path, key_path, key))
            })
            .transpose()?
    } else {
        None
    };

    if let Some((lib_path, key_path, expected_key)) = cache.as_ref() {
        if let Some((project_key, artifact_key)) = read_preflight_cache_manifest(key_path) {
            if project_key == *expected_key {
                if let Some(mut compiled) =
                    load_verified_preflight_cache_copy(lib_path, &artifact_key)?
                {
                    tracing::info!(
                        path = %lib_path.display(),
                        cache_key = %expected_key,
                        execution_kind = ?opts.execution_kind,
                        "loading hash-validated rvr native artifact"
                    );
                    // The cache key covers the generated C tree, which encodes
                    // the inline-record decision, so this metadata matches the
                    // cached library.
                    compiled.inline_records = inline_meta;
                    return Ok(compiled);
                }
            }
        }
        tracing::info!(
            path = %lib_path.display(),
            cache_key = %expected_key,
            execution_kind = ?opts.execution_kind,
            "rvr native artifact cache miss"
        );
    }

    compile_generated_project(output_dir, &make_args, &toolchain)?;

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

    let compiled = RvrCompiled {
        lib,
        lib_path,
        artifact_dir: Some(artifact_dir),
        execution_kind,
        inline_records: inline_meta,
    };
    if let Some((cache_lib, cache_key_path, cache_key)) = cache {
        publish_preflight_cache(&compiled, &cache_lib, &cache_key_path, &cache_key)?;
        tracing::info!(
            path = %cache_lib.display(),
            cache_key = %cache_key,
            execution_kind = ?opts.execution_kind,
            "saved hash-validated rvr native artifact"
        );
    }
    Ok(compiled)
}

pub(crate) fn env_flag_is_off(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
}

fn native_opt_level(
    env: &str,
    invalid_message: &'static str,
) -> Result<Option<String>, CompileError> {
    let Some(value) = std::env::var_os(env) else {
        return Ok(None);
    };
    let value = value.to_string_lossy().into_owned();
    if matches!(
        value.as_str(),
        "-O0" | "-O1" | "-O2" | "-O3" | "-Os" | "-Oz"
    ) {
        Ok(Some(value))
    } else {
        Err(CompileError::InvalidOptions(invalid_message))
    }
}

fn preflight_cache_key_path(lib_path: &Path) -> PathBuf {
    let mut path = lib_path.as_os_str().to_os_string();
    path.push(".sha256");
    PathBuf::from(path)
}

fn artifact_sha256(path: &Path) -> Result<String, CompileError> {
    let mut file = File::open(path).map_err(|source| CompileError::CProject {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| CompileError::CProject {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(hasher.finalize()))
}

fn read_preflight_cache_manifest(path: &Path) -> Option<(String, String)> {
    let manifest = fs::read_to_string(path).ok()?;
    let mut lines = manifest.lines();
    let project_key = lines.next()?.strip_prefix("project=")?.to_string();
    let artifact_key = lines.next()?.strip_prefix("artifact=")?.to_string();
    (lines.next().is_none()).then_some((project_key, artifact_key))
}

fn load_verified_preflight_cache_copy(
    cache_lib: &Path,
    expected_artifact_key: &str,
) -> Result<Option<RvrCompiled>, CompileError> {
    if !cache_lib.is_file() {
        return Ok(None);
    }
    let parent = cache_lib.parent().unwrap_or_else(|| Path::new("."));
    let temp_dir = tempfile::Builder::new()
        .prefix(".openvm-rvr-cache-load-")
        .tempdir_in(parent)
        .map_err(|source| CompileError::CProject {
            path: parent.to_path_buf(),
            source,
        })?;
    let file_name = cache_lib
        .file_name()
        .ok_or_else(|| CompileError::LibLoad("cached library path has no file name".into()))?;
    let private_lib = temp_dir.path().join(file_name);
    if fs::hard_link(cache_lib, &private_lib).is_err() {
        fs::copy(cache_lib, &private_lib).map_err(|source| CompileError::CProject {
            path: private_lib.clone(),
            source,
        })?;
    }
    if artifact_sha256(&private_lib)? != expected_artifact_key {
        return Ok(None);
    }
    let lib = unsafe {
        libloading::Library::new(&private_lib)
            .map_err(|err| CompileError::LibLoad(format!("{}: {err}", private_lib.display())))?
    };
    let execution_kind = load_execution_kind(&lib)?;
    Ok(Some(RvrCompiled {
        lib,
        lib_path: private_lib,
        artifact_dir: Some(ArtifactDir::Temp(temp_dir)),
        execution_kind,
        inline_records: RvrInlineRecordsMeta::default(),
    }))
}

fn publish_preflight_cache(
    compiled: &RvrCompiled,
    cache_lib: &Path,
    cache_key_path: &Path,
    project_key: &str,
) -> Result<(), CompileError> {
    let parent = cache_lib.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| CompileError::CProject {
        path: parent.to_path_buf(),
        source,
    })?;
    let temp_lib = tempfile::Builder::new()
        .prefix(".openvm-rvr-cache-lib-")
        .tempfile_in(parent)
        .map_err(|source| CompileError::CProject {
            path: parent.to_path_buf(),
            source,
        })?;
    compiled.save_artifact(temp_lib.path())?;
    temp_lib
        .as_file()
        .sync_all()
        .map_err(|source| CompileError::CProject {
            path: temp_lib.path().to_path_buf(),
            source,
        })?;
    let artifact_key = artifact_sha256(temp_lib.path())?;

    let key_parent = cache_key_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(key_parent).map_err(|source| CompileError::CProject {
        path: key_parent.to_path_buf(),
        source,
    })?;
    let temp_manifest = tempfile::Builder::new()
        .prefix(".openvm-rvr-cache-manifest-")
        .tempfile_in(key_parent)
        .map_err(|source| CompileError::CProject {
            path: key_parent.to_path_buf(),
            source,
        })?;
    fs::write(
        temp_manifest.path(),
        format!("project={project_key}\nartifact={artifact_key}\n"),
    )
    .map_err(|source| CompileError::CProject {
        path: temp_manifest.path().to_path_buf(),
        source,
    })?;
    temp_manifest
        .as_file()
        .sync_all()
        .map_err(|source| CompileError::CProject {
            path: temp_manifest.path().to_path_buf(),
            source,
        })?;

    temp_lib
        .persist(cache_lib)
        .map_err(|err| CompileError::CProject {
            path: cache_lib.to_path_buf(),
            source: err.error,
        })?;
    temp_manifest
        .persist(cache_key_path)
        .map_err(|err| CompileError::CProject {
            path: cache_key_path.to_path_buf(),
            source: err.error,
        })?;
    Ok(())
}

fn generated_project_cache_key(
    output_dir: &Path,
    make_args: &[String],
    toolchain: &rvr_openvm::RuntimeToolchain,
) -> Result<String, CompileError> {
    fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), CompileError> {
        for entry in fs::read_dir(dir).map_err(|source| CompileError::CProject {
            path: dir.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| CompileError::CProject {
                path: dir.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            if path.is_dir() {
                collect_files(&path, files)?;
            } else if path.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect_files(output_dir, &mut files)?;
    files.sort_by(|lhs, rhs| {
        lhs.strip_prefix(output_dir)
            .unwrap()
            .cmp(rhs.strip_prefix(output_dir).unwrap())
    });

    let mut hasher = Sha256::new();
    hasher.update(b"openvm-rvr-preflight-artifact-v2\0");
    hasher.update(std::env::consts::OS.as_bytes());
    hasher.update([0]);
    hasher.update(std::env::consts::ARCH.as_bytes());
    hasher.update([0]);
    hasher.update(toolchain.compiler.as_bytes());
    hasher.update([0]);
    hasher.update(toolchain.linker.as_bytes());
    hasher.update([0]);
    hasher.update(toolchain.make.as_bytes());
    hasher.update([0]);
    hasher.update(toolchain.host_os.as_bytes());
    hasher.update([0]);
    update_command_identity(&mut hasher, &toolchain.compiler, &["--version"])?;
    update_command_identity(&mut hasher, &toolchain.linker, &["--version"])?;
    update_command_identity(&mut hasher, &toolchain.make, &["--version"])?;
    update_command_identity(
        &mut hasher,
        &toolchain.compiler,
        &["-march=native", "-###", "-E", "-x", "c", "-"],
    )?;
    for name in [
        "OPT",
        "DEBUG",
        "LTO",
        "LDFLAGS",
        "LDLIBS",
        "EXT_LIBS",
        "EXT_SRCS",
        "EXT_CFLAGS",
        "LIB",
        "HOST_OS",
    ] {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        if let Some(value) = std::env::var_os(name) {
            hasher.update(value.as_encoded_bytes());
        }
        hasher.update([0]);
    }

    let output_dir_text = output_dir.to_string_lossy();
    for arg in make_args {
        hasher.update(arg.replace(output_dir_text.as_ref(), "$OUT").as_bytes());
        hasher.update([0]);
    }
    for path in files {
        let relative = path.strip_prefix(output_dir).unwrap();
        hasher.update(relative.as_os_str().as_encoded_bytes());
        hasher.update([0]);
        let contents = fs::read(&path).map_err(|source| CompileError::CProject {
            path: path.clone(),
            source,
        })?;
        hasher.update((contents.len() as u64).to_le_bytes());
        hasher.update(contents);
    }

    Ok(hex_digest(hasher.finalize()))
}

fn update_command_identity(
    hasher: &mut Sha256,
    command: &str,
    args: &[&str],
) -> Result<(), CompileError> {
    let output = Command::new(command)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|source| CompileError::ToolchainCommand {
            command: command.to_string(),
            source,
        })?;
    hasher.update(command.as_bytes());
    hasher.update([0]);
    for arg in args {
        hasher.update(arg.as_bytes());
        hasher.update([0]);
    }
    hasher.update(output.status.code().unwrap_or(-1).to_le_bytes());
    hasher.update((output.stdout.len() as u64).to_le_bytes());
    hasher.update(output.stdout);
    hasher.update((output.stderr.len() as u64).to_le_bytes());
    hasher.update(output.stderr);
    Ok(())
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut key = String::with_capacity(digest.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        key.push(HEX[(byte >> 4) as usize] as char);
        key.push(HEX[(byte & 0x0f) as usize] as char);
    }
    key
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
    let default_jobs =
        std::thread::available_parallelism().map_or(4, |n| n.get().saturating_sub(2).max(1));
    let jobs = std::env::var("OPENVM_RVR_MAKE_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|jobs| *jobs > 0)
        .unwrap_or(default_jobs)
        .to_string();
    tracing::info!(
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
