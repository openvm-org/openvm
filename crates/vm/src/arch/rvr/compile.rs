//! IR -> CProject -> make -> .so pipeline.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    fmt::{self, Write as _},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode, SystemOpcode,
    VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{
    inline_record_shape_for_instr, inline_record_shape_for_terminator, CProject,
    G2DsoManifestConfigV2, G2EmissionMode, InlineRecordShape, RvrExecutionKind,
};
use rvr_openvm_ir::{LiftedInstr, SourceLoc};
use rvr_openvm_lift::{
    build_blocks, convert_vmexe_to_ir_with_debug, opcode::lift_instruction, AirIndex,
    ExtensionRegistry, RvrInstruction, TraceChipIndex,
};
use sha2::{Digest, Sha256};

use super::{
    debug::GuestDebugMap, ArenaNativeGeometry, ArenaNativeLayout, LogNativeOpcodeAdmitter,
    RvrDeltaDecodeEntry, RvrG2AirBindingV1, RvrG2BlockEntryV1, RvrG2BlockHostCountsV1, RvrG2MetaV1,
    RvrG2OpaqueBindingV1,
};
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
    /// Stage-2: emit one chronological 24-byte delta stream instead of the
    /// per-AIR compact wire. `airs` still describes the decoder's output
    /// shapes and is used for fail-closed route validation.
    pub delta_records: bool,
    /// Program-only operand table and compiler-scope mixed-AIR
    /// classification, built during untimed preflight AOT compilation. The
    /// cached VM loads this directly; the first timed segment never rescans
    /// the executable.
    pub delta_decode: Option<Arc<RvrDeltaDecodePrecompute>>,
    /// Compiler-authoritative proof that every defined routed slot is owned
    /// by either the whole-AIR device decoder or an arena-native consumer.
    /// This makes the access-aux omission decision O(1) at cached-VM load.
    pub fully_direct_delta: bool,
    /// Private G2 transport metadata. `None` means the request negotiated to
    /// the established arena route before native execution.
    pub g2: Option<Arc<RvrG2MetaV1>>,
}

/// Extension-neutral persisted input for the CUDA delta decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RvrDeltaDecodePrecompute {
    pub pc_base: u32,
    pub entries: Arc<Vec<RvrDeltaDecodeEntry>>,
    /// Sorted by the extension's stable `repr(u8)` decoder kind.
    pub kind_to_air: Vec<(u8, usize)>,
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

    /// R3 inline-record metadata for a preflight library (empty for other
    /// tracer modes and for libraries loaded without compile metadata).
    pub fn inline_records(&self) -> &RvrInlineRecordsMeta {
        &self.inline_records
    }

    fn validate_g2_manifest(&self) -> Result<(), CompileError> {
        let Some(g2) = self.inline_records.g2.as_deref() else {
            return Ok(());
        };
        let manifest = unsafe {
            let symbol = self
                .lib
                .get::<*const OpenVmRvrG2DsoManifestV2>(b"openvm_rvr_g2_manifest_v2\0")
                .map_err(|err| {
                    CompileError::LibLoad(format!(
                        "G2 negotiation rejected DSO without manifest: {err}"
                    ))
                })?;
            **symbol
        };
        let mut expected_lanes =
            Vec::with_capacity(rvr_openvm_ext_ffi_common::G2_PRODUCER_LANE_COUNT);
        expected_lanes.push(OpenVmRvrG2DsoLaneManifestV1 {
            kind: 0x0001,
            elem_width: 4,
            encoding: 0,
            flags: 1,
            group_id: 0,
            arity: 0,
            reserved: [0; 3],
        });
        for (kind, width, flags, group_id) in [
            (0x0080, 8, 3, 2),
            (0x0081, 1, 3, 2),
            (0x0082, 8, 3, 2),
            (0x0083, 4, 3, 2),
            (0x0084, 4, 1, 0),
        ] {
            expected_lanes.push(OpenVmRvrG2DsoLaneManifestV1 {
                kind,
                elem_width: width,
                encoding: 0,
                flags,
                group_id,
                arity: 1,
                reserved: [0; 3],
            });
        }
        for kind in 0u8..rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT as u8 {
            if !rvr_openvm_ext_ffi_common::g2_standard_decoder_kind(kind) {
                continue;
            }
            for value_lane in [false, true] {
                let Some(width) =
                    rvr_openvm_ext_ffi_common::g2_standard_lane_width(kind, value_lane)
                else {
                    continue;
                };
                let load_store = rvr_openvm_ext_ffi_common::G2_LOAD_STORE_KINDS.contains(&kind);
                expected_lanes.push(OpenVmRvrG2DsoLaneManifestV1 {
                    kind: if value_lane {
                        rvr_openvm_ext_ffi_common::g2_lane_v1(kind)
                    } else {
                        rvr_openvm_ext_ffi_common::g2_lane_v0(kind)
                    },
                    elem_width: width,
                    encoding: 0,
                    flags: if load_store { 3 } else { 1 },
                    group_id: if load_store { 1 } else { 0 },
                    arity: g2_kind_arity(kind),
                    reserved: [0; 3],
                });
            }
        }
        expected_lanes.sort_unstable_by_key(|lane| lane.kind);
        let expected_lanes: [OpenVmRvrG2DsoLaneManifestV1;
            rvr_openvm_ext_ffi_common::G2_PRODUCER_LANE_COUNT] = expected_lanes
            .try_into()
            .expect("frozen G2 capability lane count");
        let mut expected_air_kinds = [u8::MAX; rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT];
        let mut expected_air_indices = [u32::MAX; rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT];
        for (index, binding) in g2.air_bindings.iter().enumerate() {
            expected_air_kinds[index] = binding.kind;
            expected_air_indices[index] = u32::try_from(binding.air_idx)
                .map_err(|_| CompileError::LibLoad("G2 AIR index exceeds u32".to_string()))?;
        }
        if manifest.magic != *b"OVMG2D2\0"
            || manifest.version != 2
            || manifest.manifest_bytes as usize != std::mem::size_of::<OpenVmRvrG2DsoManifestV2>()
            || manifest.header_size != 64
            || manifest.lane_desc_size != 32
            || manifest.lane_count as usize != rvr_openvm_ext_ffi_common::G2_PRODUCER_LANE_COUNT
            || manifest.wire_flags != 14
            || manifest.fingerprint != g2.fingerprint
            || manifest.producer_schema_fingerprint != g2.producer_schema_fingerprint
            || manifest.program_fingerprint != g2.program_fingerprint
            || manifest.block_fingerprint != g2.block_fingerprint
            || manifest.air_manifest_fingerprint != g2.air_manifest_fingerprint
            || self
                .inline_records
                .delta_decode
                .as_deref()
                .is_none_or(|decode| manifest.pc_base != decode.pc_base)
            || manifest.block_count as usize != g2.blocks.len()
            || manifest.air_count as usize != g2.air_bindings.len()
            || manifest.emission_mode != u32::from(g2.emission_mode)
            || manifest.air_kinds != expected_air_kinds
            || manifest.air_indices != expected_air_indices
            || manifest.lanes != expected_lanes
        {
            return Err(CompileError::LibLoad(
                "G2 DSO manifest differs from the VmExe/AIR binding".to_string(),
            ));
        }
        Ok(())
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

#[repr(C)]
#[derive(Clone, Copy)]
struct OpenVmRvrG2DsoManifestV2 {
    magic: [u8; 8],
    version: u16,
    manifest_bytes: u16,
    header_size: u16,
    lane_desc_size: u16,
    lane_count: u32,
    wire_flags: u32,
    fingerprint: [u8; 32],
    producer_schema_fingerprint: [u8; 32],
    program_fingerprint: [u8; 32],
    block_fingerprint: [u8; 32],
    air_manifest_fingerprint: [u8; 32],
    pc_base: u32,
    block_count: u32,
    air_count: u32,
    emission_mode: u32,
    air_kinds: [u8; rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT],
    air_indices: [u32; rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT],
    lanes: [OpenVmRvrG2DsoLaneManifestV1; rvr_openvm_ext_ffi_common::G2_PRODUCER_LANE_COUNT],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OpenVmRvrG2DsoLaneManifestV1 {
    kind: u16,
    elem_width: u8,
    encoding: u8,
    flags: u32,
    group_id: u32,
    arity: u8,
    reserved: [u8; 3],
}

const _: () = {
    assert!(std::mem::size_of::<OpenVmRvrG2DsoLaneManifestV1>() == 16);
    assert!(std::mem::size_of::<OpenVmRvrG2DsoManifestV2>() == 1176);
};

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
    #[error(
        "rvr preflight has no registered log-native assembler for opcode {opcode:?} at pc {pc:#x}; requires interpreter routing"
    )]
    PreflightExtensionOpcode { pc: u32, opcode: VmOpcode },
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

/// Coarse profiling family derived before lifting turns extension instructions
/// into a shared `Ext`/phantom IR surface. Numeric ranges are the stable VM
/// opcode class offsets; phantom discriminants distinguish IO and crypto
/// hints that intentionally share `SystemOpcode::PHANTOM`.
fn native_detail_family<F: PrimeField32>(instruction: &Instruction<F>) -> u8 {
    let opcode = instruction.opcode.as_usize();
    if instruction.opcode == SystemOpcode::PHANTOM.global_opcode() {
        return match instruction.c.as_canonical_u32() as u16 {
            0x20..=0x22 => 1, // RV64 IO hints
            0x30 => 6,        // pairing final-exp hint
            0x50..=0x51 => 5, // modular arithmetic hints
            0..=3 => 0,       // system phantoms
            _ => 8,
        };
    }
    match opcode {
        0x260..=0x261 => 1, // RV64 hint-store / hint-buffer
        // REVEAL is encoded as RV64 STORED into the public-values address space.
        0x214 if instruction.e.as_canonical_u32() == 3 => 1,
        0x310..=0x311 => 2,               // Keccak-f / XORIN
        0x320..=0x321 => 3,               // SHA-256 / SHA-512
        0x400..0x500 => 4,                // bigint
        0x500..0x600 | 0x700..0x800 => 5, // modular / Fp2 algebra
        0x600..0x700 => 6,                // elliptic-curve / pairing operations
        0x800..0x900 => 7,                // deferral
        0x300.. => 8,                     // extension class not covered above
        _ => 0,                           // system + RV64IM core
    }
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

/// Classify instructions handled by the base system lifter only.
pub fn classify_preflight_opcodes<F: PrimeField32>(exe: &VmExe<F>) -> RvrPreflightOpcodeClass {
    let extensions = ExtensionRegistry::new();
    classify_preflight_opcodes_with_extensions(exe, &extensions, &())
}

/// Classify whether every extension-lifted instruction has a registered
/// log-native assembler. #3059 moved RV64I/M into the extension registry, so
/// they use the same admission seam as the other RVR extensions.
pub fn classify_preflight_opcodes_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    assembler_admitter: &dyn LogNativeOpcodeAdmitter<F>,
) -> RvrPreflightOpcodeClass {
    let base_only = ExtensionRegistry::new();
    for (pc, insn, _) in exe.program.enumerate_by_pc() {
        let rvr_insn = RvrInstruction::from_field(&insn);
        if matches!(
            lift_instruction(&rvr_insn, u64::from(pc), &base_only),
            Ok(Some(_))
        ) {
            continue;
        }
        if assembler_admitter.has_log_native_assembler(&insn)
            && matches!(
                lift_instruction(&rvr_insn, u64::from(pc), extensions),
                Ok(Some(_))
            )
        {
            continue;
        }
        return RvrPreflightOpcodeClass::Unsupported {
            pc,
            opcode: rvr_insn.opcode,
        };
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

/// Collect which instructions the preflight codegen migrates to inline compact
/// records: the emitter's per-instruction decision is
/// [`instr_emits_inline_record`] on the lifted IR instruction plus a
/// `TraceChipIndex::Chip` mapping for its pc (mirroring
/// `CProject::chip_idx_for_pc`). Walks the same lifted instruction stream the
/// emitter consumes so the host metadata cannot drift from the generated C.
fn collect_inline_records_meta<F: PrimeField32>(
    exe: &VmExe<F>,
    ir: &[LiftedInstr],
    chips: &ChipMapping,
    admitter: Option<&dyn LogNativeOpcodeAdmitter<F>>,
    gpu_records_default: Option<&str>,
) -> RvrInlineRecordsMeta {
    let gpu_records = configured_gpu_records(gpu_records_default);
    collect_inline_records_meta_for_mode(exe, ir, chips, admitter, gpu_records.as_deref())
}

/// Explicit GPU record selection wins over the proving builder's default.
fn configured_gpu_records(gpu_records_default: Option<&str>) -> Option<String> {
    match std::env::var("OPENVM_RVR_GPU_RECORDS") {
        Ok(mode) => Some(mode),
        Err(std::env::VarError::NotPresent) => gpu_records_default.map(str::to_owned),
        Err(_) => None,
    }
}

fn configured_g2_emission_mode() -> Result<G2EmissionMode, CompileError> {
    match std::env::var("OPENVM_RVR_G2_EMISSION") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "checked" | "debug" => Ok(G2EmissionMode::Checked),
            "production" | "prod" => Ok(G2EmissionMode::Production),
            _ => Err(CompileError::InvalidOptions(
                "OPENVM_RVR_G2_EMISSION must be checked or production",
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(if cfg!(debug_assertions) {
            G2EmissionMode::Checked
        } else {
            G2EmissionMode::Production
        }),
        Err(_) => Err(CompileError::InvalidOptions(
            "OPENVM_RVR_G2_EMISSION must contain valid UTF-8",
        )),
    }
}

fn collect_inline_records_meta_for_mode<F: PrimeField32>(
    exe: &VmExe<F>,
    ir: &[LiftedInstr],
    chips: &ChipMapping,
    admitter: Option<&dyn LogNativeOpcodeAdmitter<F>>,
    gpu_records: Option<&str>,
) -> RvrInlineRecordsMeta {
    let num_slots = exe.program.instructions_and_debug_infos.len();
    let pc_base = u64::from(exe.program.pc_base);
    let mut pc_slots = vec![false; num_slots];
    let mut airs: BTreeMap<usize, usize> = BTreeMap::new();
    let mut arena_native: BTreeMap<usize, Option<ArenaNativeGeometry>> = BTreeMap::new();
    // Three-tier emission precedence: an explicit GPU compact request wins
    // over the default-on arena-native (G1) emission, then non-migrated
    // opcodes retain the log fallback. A fused-compiled library has no wire
    // records to decode, so allowing arena-native metadata under a compact
    // request would make the GPU builder request one shape while generated C
    // emits another. Keep this decision in the shared host/codegen metadata.
    let compact_wire_requested = matches!(gpu_records, Some("compact" | "g2"));
    let delta_records_requested = gpu_records == Some("delta");
    let g2_requested = gpu_records == Some("g2");
    let arena_native_enabled = (gpu_records == Some("arena-native")
        || std::env::var("OPENVM_RVR_ARENA_NATIVE").as_deref() != Ok("0"))
        && !compact_wire_requested;
    {
        let mut record = |pc: u64, shape: InlineRecordShape| {
            let Some(offset) = pc.checked_sub(pc_base) else {
                return;
            };
            let slot = (offset / u64::from(DEFAULT_PC_STEP)) as usize;
            let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) else {
                return;
            };
            let registered_geometry = admitter.and_then(|admitter| {
                exe.program
                    .instructions_and_debug_infos
                    .get(slot)
                    .and_then(|entry| entry.as_ref())
                    .and_then(|(instruction, _)| admitter.inline_arena_geometry_for(instruction))
            });
            let geometry = registered_geometry
                .filter(|geometry| {
                    arena_native_enabled
                        || g2_requested
                            && matches!(
                                geometry.layout,
                                ArenaNativeLayout::Custom { .. }
                                    | ArenaNativeLayout::CustomVariableRows { .. }
                            )
                })
                .filter(|geometry| {
                    !delta_records_requested
                        || matches!(
                            geometry.layout,
                            ArenaNativeLayout::Custom {
                                residual_memory_chronology: true,
                                ..
                            }
                        )
                        || matches!(
                            geometry.layout,
                            ArenaNativeLayout::CustomVariableRows {
                                residual_memory_chronology: true
                            }
                        )
                });
            // Extension-owned custom records have no generic Stage-2 delta
            // encoding. Without their complete-record arena target, keep the
            // instruction on the verbose program-log/assembler route.
            if geometry.is_none()
                && (matches!(shape, InlineRecordShape::CustomVariableRows { .. })
                    || delta_records_requested && matches!(shape, InlineRecordShape::Custom { .. }))
            {
                return;
            }
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
            let previous = arena_native.insert(air_idx, geometry);
            assert!(
                previous.is_none_or(|p| p == geometry),
                "conflicting arena-native geometry for air {air:?}: {previous:?} vs {geometry:?}"
            );
        };
        for lifted in ir {
            match lifted {
                LiftedInstr::Body(instr_at) => {
                    if let Some(shape) = inline_record_shape_for_instr(instr_at.instr.as_ref()) {
                        record(instr_at.pc, shape);
                    }
                }
                LiftedInstr::Term { pc, terminator, .. } => {
                    if let Some(shape) = inline_record_shape_for_terminator(terminator) {
                        record(*pc, shape);
                    }
                }
            }
        }
    }

    // Arena-native targets replace an AIR's entire assembled arena after the
    // log walk. They are therefore sound only when every program instruction
    // routed to that AIR emits an inline record into the target. A REVEAL is
    // the important mixed case: narrow REVEAL shares a LoadStore AIR with
    // inline main-memory loads/stores, but remains on the verbose log path.
    // Staging that AIR would discard the log-assembled REVEAL rows at
    // substitution. Taint mixed AIRs back to compact emission so the host
    // assembler composes both record sources into one arena.
    let mut tainted_delta_airs = BTreeSet::new();
    for (slot, entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        if entry.is_none() || pc_slots.get(slot).copied().unwrap_or(false) {
            continue;
        }
        if let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) {
            let air_idx = air.as_u32() as usize;
            arena_native.remove(&air_idx);
            if delta_records_requested {
                tainted_delta_airs.insert(air_idx);
            }
        }
    }

    // Device delta partitioning owns a complete AIR. If one program slot on
    // that AIR is not delta-inline (for example a narrow AS=3 REVEAL sharing a
    // LoadStore AIR), clear every inline slot on the AIR so the entire family
    // retains its program/access logs and uses host assembly. This also keeps
    // extension-owned custom schemas fail closed.
    if !tainted_delta_airs.is_empty() {
        for (slot, flag) in pc_slots.iter_mut().enumerate() {
            if !*flag {
                continue;
            }
            if let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) {
                if tainted_delta_airs.contains(&(air.as_u32() as usize)) {
                    *flag = false;
                }
            }
        }
        airs.retain(|air, _| !tainted_delta_airs.contains(air));
    }

    // Narrow public-values stores are not part of the established compact or
    // delta inline routes, but G2 owns them directly. Admit them to the decode
    // table before whole-AIR tainting so a LoadStore AIR shared with ordinary
    // memory accesses can negotiate G2 as one complete device-owned family.
    let mut decode_pc_slots = pc_slots.clone();
    if g2_requested {
        for (slot, program_entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
            if decode_pc_slots.get(slot).copied().unwrap_or(false) {
                continue;
            }
            let Some((instruction, _)) = program_entry else {
                continue;
            };
            let narrow_reveal = admitter
                .and_then(|admitter| admitter.delta_decode_for(instruction))
                .is_some_and(|decoded| {
                    decoded.entry.flags & (1 << 4) != 0
                        && decoded.entry.access_pattern == 3
                        && decoded.entry.local_opcode != 4
                });
            if narrow_reveal {
                decode_pc_slots[slot] = true;
            }
        }
    }
    let mut delta_decode = if compact_wire_requested || delta_records_requested {
        admitter
            .filter(|admitter| admitter.has_delta_decode())
            .map(|admitter| build_delta_decode_precompute(exe, chips, admitter, &decode_pc_slots))
    } else {
        None
    };
    if g2_requested {
        if let (Some(decode), Some(admitter)) = (delta_decode.as_mut(), admitter) {
            augment_g2_custom_decode(exe, chips, admitter, decode);
        }
    }
    let g2_supported = g2_requested
        && delta_decode.as_ref().is_some_and(|precomputed| {
            !precomputed.kind_to_air.is_empty()
                && precomputed
                    .kind_to_air
                    .iter()
                    .all(|&(kind, _)| g2_decoder_kind(kind))
                && exe
                    .program
                    .instructions_and_debug_infos
                    .iter()
                    .enumerate()
                    .all(|(slot, instruction)| {
                        if instruction.is_none() {
                            return true;
                        }
                        let Some(TraceChipIndex::Chip(_)) = chips.pc_to_chip.get(slot) else {
                            return instruction.as_ref().is_some_and(|(instruction, _)| {
                                instruction.opcode == SystemOpcode::TERMINATE.global_opcode()
                            });
                        };
                        precomputed.entries.get(slot).is_some_and(|entry| {
                            if entry.air_idx == u8::MAX {
                                return false;
                            }
                            let kind = precomputed
                                .kind_to_air
                                .iter()
                                .find(|(_, air)| *air == entry.air_idx as usize)
                                .map(|(kind, _)| *kind);
                            let narrow_reveal = entry.flags & (1 << 4) != 0
                                && entry.access_pattern == 3
                                && entry.local_opcode != 4;
                            let inline_ok =
                                pc_slots.get(slot).copied().unwrap_or(false) || narrow_reveal;
                            inline_ok
                                && instruction.as_ref().is_some_and(|(instruction, _)| {
                                    admitter
                                        .and_then(|admitter| match kind {
                                            Some(_) => admitter.inline_g2_geometry_for(instruction),
                                            None => admitter.inline_arena_geometry_for(instruction),
                                        })
                                        .is_some_and(|geometry| match kind {
                                            Some(kind) => {
                                                g2_access_pattern_matches(
                                                    kind,
                                                    entry.access_pattern,
                                                ) && g2_layout_matches(kind, geometry.layout)
                                            }
                                            None => {
                                                matches!(entry.access_pattern, 10 | 11)
                                                    && matches!(
                                                        geometry.layout,
                                                        ArenaNativeLayout::Custom {
                                                            residual_memory_chronology: true,
                                                            ..
                                                        }
                                                    )
                                            }
                                        })
                                })
                        })
                    })
        });
    if g2_requested && !g2_supported {
        tracing::info!(
            "G2 fixed-standard negotiation rejected this executable; selecting arena route before native execution"
        );
        return collect_inline_records_meta_for_mode(
            exe,
            ir,
            chips,
            admitter,
            Some("arena-native"),
        );
    }
    if g2_supported {
        let decode = delta_decode
            .as_ref()
            .expect("supported G2 route has a decode table");
        for (slot, entry) in decode.entries.iter().enumerate() {
            let narrow_reveal = entry.air_idx != u8::MAX
                && entry.flags & (1 << 4) != 0
                && entry.access_pattern == 3
                && entry.local_opcode != 4;
            if narrow_reveal {
                pc_slots[slot] = true;
            }
        }
    }
    let hintstore_air = g2_supported.then_some(()).and_then(|()| {
        delta_decode
            .as_ref()
            .and_then(|decode| decode.kind_to_air.iter().find(|&&(kind, _)| kind == 30))
            .map(|&(_, air)| air)
    });
    let arena_native_airs = arena_native
        .into_iter()
        .filter_map(|(air, geometry)| {
            (Some(air) != hintstore_air).then(|| geometry.map(|g| (air, g)))?
        })
        .collect::<Vec<_>>();
    if g2_supported {
        for &(_, air) in &delta_decode
            .as_ref()
            .expect("supported G2 route has a decode table")
            .kind_to_air
        {
            airs.entry(air)
                .or_insert(rvr_openvm_ext_ffi_common::PREFLIGHT_ADDSUB_RECORD_SIZE);
        }
    }
    let mut direct_airs = delta_decode
        .iter()
        .flat_map(|precomputed| precomputed.kind_to_air.iter().map(|&(_, air)| air))
        .collect::<BTreeSet<_>>();
    direct_airs.extend(arena_native_airs.iter().map(|&(air, _)| air));
    let fully_direct_delta = delta_records_requested
        && exe
            .program
            .instructions_and_debug_infos
            .iter()
            .enumerate()
            .all(|(slot, instruction)| {
                if instruction.is_none() {
                    return true;
                }
                let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) else {
                    return true;
                };
                pc_slots.get(slot).copied().unwrap_or(false)
                    && direct_airs.contains(&(air.as_u32() as usize))
            });

    RvrInlineRecordsMeta {
        pc_slots: Arc::new(pc_slots),
        airs: airs.into_iter().collect(),
        arena_native_airs,
        delta_records: delta_records_requested,
        delta_decode: delta_decode.map(Arc::new),
        fully_direct_delta,
        g2: None,
    }
}

fn g2_decoder_kind(kind: u8) -> bool {
    (kind as usize) < rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT
}

fn augment_g2_custom_decode<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
    admitter: &dyn LogNativeOpcodeAdmitter<F>,
    decode: &mut RvrDeltaDecodePrecompute,
) {
    let mut entries = (*decode.entries).clone();
    let mut hintstore_air = None;
    for (slot, program_entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = program_entry else {
            continue;
        };
        let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) else {
            continue;
        };
        let air_idx = air.as_u32() as usize;
        if decode
            .kind_to_air
            .iter()
            .any(|&(_, bound)| bound == air_idx)
        {
            continue;
        }
        let Some(geometry) = admitter.inline_arena_geometry_for(instruction) else {
            continue;
        };
        let entry = &mut entries[slot];
        entry.a = instruction.a.as_canonical_u32();
        entry.b = instruction.b.as_canonical_u32();
        entry.c = instruction.c.as_canonical_u32();
        entry.air_idx = u8::try_from(air_idx)
            .expect("G2 custom AIR index exceeds the persisted u8 operand ABI");
        match geometry.layout {
            ArenaNativeLayout::CustomVariableRows {
                residual_memory_chronology: true,
            } => {
                entry.local_opcode = u8::from(!instruction.a.is_zero());
                entry.access_pattern = 9;
                if let Some(previous) = hintstore_air.replace(air_idx) {
                    assert_eq!(previous, air_idx, "HintStore maps to multiple AIRs");
                }
            }
            ArenaNativeLayout::Custom {
                residual_memory_chronology: true,
                ..
            } => {
                entry.access_pattern =
                    if instruction.opcode == SystemOpcode::PHANTOM.global_opcode() {
                        11
                    } else {
                        10
                    };
            }
            _ => entry.air_idx = u8::MAX,
        }
    }
    if let Some(air) = hintstore_air {
        decode.kind_to_air.push((30, air));
        decode.kind_to_air.sort_unstable_by_key(|&(kind, _)| kind);
    }
    decode.entries = Arc::new(entries);
}

fn g2_access_pattern_matches(kind: u8, pattern: u8) -> bool {
    match kind {
        0 | 15..=19 => pattern == 1,
        1..=7 => pattern == 0,
        8 | 9 | 20..=28 => matches!(pattern, 2 | 3),
        10 | 11 => pattern == 4,
        12 => pattern == 5,
        13 => pattern == 7,
        14 => pattern == 6,
        29 => pattern == 8,
        30 => pattern == 9,
        31..=37 => pattern == 8,
        _ => false,
    }
}

fn g2_layout_matches(kind: u8, layout: ArenaNativeLayout) -> bool {
    match kind {
        0..=7 | 15..=19 => matches!(layout, ArenaNativeLayout::Alu3(_)),
        8 | 9 | 20..=28 => matches!(layout, ArenaNativeLayout::LoadStore(_)),
        10 | 11 => matches!(layout, ArenaNativeLayout::Branch2(_)),
        12 | 14 => matches!(layout, ArenaNativeLayout::Wr1(_)),
        13 => matches!(layout, ArenaNativeLayout::Rw1(_)),
        29 => matches!(layout, ArenaNativeLayout::AddI(_)),
        30 => matches!(
            layout,
            ArenaNativeLayout::CustomVariableRows {
                residual_memory_chronology: true
            }
        ),
        31..=37 => matches!(layout, ArenaNativeLayout::AluImm(_)),
        _ => false,
    }
}

/// Build the operand table and whole-AIR classification while
/// `compile_preflight` is already walking the immutable compiled program.
/// Unsupported slots taint their entire AIR, matching the runtime oracle.
fn build_delta_decode_precompute<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
    admitter: &dyn LogNativeOpcodeAdmitter<F>,
    inline_pc_slots: &[bool],
) -> RvrDeltaDecodePrecompute {
    let program = &exe.program;
    let mut entries = vec![
        RvrDeltaDecodeEntry {
            air_idx: u8::MAX,
            access_pattern: u8::MAX,
            filtered_index: u32::MAX,
            ..RvrDeltaDecodeEntry::default()
        };
        program.instructions_and_debug_infos.len()
    ];
    let mut kind_to_air = BTreeMap::new();
    let mut air_to_kind = BTreeMap::new();
    let mut tainted = BTreeSet::new();

    let mut filtered_index = 0u32;
    for (slot, program_entry) in program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = program_entry else {
            continue;
        };
        entries[slot].filtered_index = filtered_index;
        filtered_index = filtered_index
            .checked_add(1)
            .expect("filtered program index exceeds u32 ABI");
        let Some(TraceChipIndex::Chip(air)) = chips.pc_to_chip.get(slot) else {
            continue;
        };
        let air_idx = air.as_u32() as usize;
        if !inline_pc_slots.get(slot).copied().unwrap_or(false) {
            tainted.insert(air_idx);
            continue;
        }
        let Some(mut decoded) = admitter.delta_decode_for(instruction) else {
            tainted.insert(air_idx);
            continue;
        };
        decoded.entry.air_idx =
            u8::try_from(air_idx).expect("delta device AIR index exceeds the persisted u8 ABI");
        decoded.entry.filtered_index = entries[slot].filtered_index;
        entries[slot] = decoded.entry;
        if let Some(previous) = kind_to_air.insert(decoded.kind, air_idx) {
            assert_eq!(
                previous, air_idx,
                "delta kind {} maps to multiple AIRs",
                decoded.kind
            );
        }
        if let Some(previous) = air_to_kind.insert(air_idx, decoded.kind) {
            assert_eq!(
                previous, decoded.kind,
                "AIR {air_idx} maps to multiple delta kinds"
            );
        }
    }
    kind_to_air.retain(|_, air| !tainted.contains(air));
    RvrDeltaDecodePrecompute {
        pc_base: program.pc_base,
        entries: Arc::new(entries),
        kind_to_air: kind_to_air.into_iter().collect(),
    }
}

fn build_g2_meta_v1<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
    decode: &RvrDeltaDecodePrecompute,
    blocks: &[rvr_openvm_ir::Block],
    admitter: &dyn LogNativeOpcodeAdmitter<F>,
    emission_mode: G2EmissionMode,
) -> Result<RvrG2MetaV1, CompileError> {
    if decode.kind_to_air.is_empty()
        || decode
            .kind_to_air
            .iter()
            .any(|&(kind, _)| !g2_decoder_kind(kind))
    {
        return Err(CompileError::InvalidOptions(
            "G2 requires fixed standard RV64 plus HintStore device-decode AIRs",
        ));
    }
    let mut geometries = BTreeMap::new();
    let mut opaque_geometries = BTreeMap::<usize, ArenaNativeGeometry>::new();
    let mut opaque_opcodes = BTreeMap::<usize, BTreeSet<usize>>::new();
    for (slot, program_entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = program_entry else {
            continue;
        };
        let Some(entry) = decode.entries.get(slot) else {
            continue;
        };
        let binding = decode
            .kind_to_air
            .iter()
            .find(|(_, air)| *air == entry.air_idx as usize)
            .copied();
        if binding.is_none() && matches!(entry.access_pattern, 10 | 11) {
            let air_idx = entry.air_idx as usize;
            let geometry = admitter.inline_arena_geometry_for(instruction).ok_or(
                CompileError::InvalidOptions("G2 opaque AIR has no registry geometry"),
            )?;
            if !matches!(
                geometry.layout,
                ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    ..
                }
            ) || opaque_geometries
                .insert(air_idx, geometry)
                .is_some_and(|previous| previous != geometry)
            {
                return Err(CompileError::InvalidOptions(
                    "G2 opaque AIR registry geometry is inconsistent",
                ));
            }
            opaque_opcodes
                .entry(air_idx)
                .or_default()
                .insert(instruction.opcode.as_usize());
            continue;
        }
        let Some((kind, air_idx)) = binding else {
            continue;
        };
        let geometry =
            admitter
                .inline_g2_geometry_for(instruction)
                .ok_or(CompileError::InvalidOptions(
                    "G2 AIR has no registry geometry",
                ))?;
        let valid_layout = g2_access_pattern_matches(kind, entry.access_pattern)
            && g2_layout_matches(kind, geometry.layout);
        if !valid_layout
            || geometries
                .insert(kind, (air_idx, geometry))
                .is_some_and(|previous| previous != (air_idx, geometry))
        {
            return Err(CompileError::InvalidOptions(
                "G2 AIR registry geometry is inconsistent",
            ));
        }
    }
    if geometries.len() != decode.kind_to_air.len() {
        return Err(CompileError::InvalidOptions(
            "G2 AIR is absent from the registry manifest",
        ));
    }
    let pc_base = u64::from(exe.program.pc_base);
    let mut kind_by_air = [u8::MAX; 256];
    for &(kind, air_idx) in &decode.kind_to_air {
        kind_by_air[air_idx] = kind;
    }
    let mut block_entries = Vec::with_capacity(blocks.len());
    for block in blocks {
        let offset = block
            .start_pc
            .checked_sub(pc_base)
            .ok_or(CompileError::InvalidOptions(
                "G2 block starts below the program pc base",
            ))?;
        if !offset.is_multiple_of(u64::from(DEFAULT_PC_STEP)) || block.insn_count() == 0 {
            return Err(CompileError::InvalidOptions(
                "G2 static block table contains an invalid entry",
            ));
        }
        let program_slot = u32::try_from(offset / u64::from(DEFAULT_PC_STEP)).map_err(|_| {
            CompileError::InvalidOptions("G2 program slot exceeds the frozen u32 schema")
        })?;
        if program_slot as usize >= exe.program.instructions_and_debug_infos.len() {
            return Err(CompileError::InvalidOptions(
                "G2 block entry exceeds the program table",
            ));
        }
        let mut host_counts = RvrG2BlockHostCountsV1::default();
        for local in 0..block.insn_count() {
            let entry = decode
                .entries
                .get(program_slot as usize + local as usize)
                .ok_or(CompileError::InvalidOptions(
                    "G2 block instruction exceeds the operand table",
                ))?;
            match kind_by_air[entry.air_idx as usize] {
                10 => host_counts.kind10 += 1,
                11 => host_counts.kind11 += 1,
                12 => host_counts.kind12 += 1,
                13 => host_counts.kind13 += 1,
                14 => host_counts.kind14 += 1,
                30 => host_counts.kind30 += 1,
                _ => {}
            }
        }
        block_entries.push((
            RvrG2BlockEntryV1 {
                program_slot,
                instruction_count: block.insn_count(),
            },
            host_counts,
        ));
    }
    block_entries.sort_unstable_by_key(|(entry, _)| entry.program_slot);
    if block_entries
        .windows(2)
        .any(|pair| pair[0].0.program_slot == pair[1].0.program_slot)
    {
        return Err(CompileError::InvalidOptions(
            "G2 static block table contains duplicate entries",
        ));
    }
    let (block_entries, block_host_counts): (Vec<_>, Vec<_>) = block_entries.into_iter().unzip();

    let mut program_table = Sha256::new();
    program_table.update(b"openvm-rvr-g2-program-table-v1\0");
    program_table.update(exe.program.pc_base.to_le_bytes());
    program_table.update(exe.pc_start.to_le_bytes());
    program_table.update((exe.program.instructions_and_debug_infos.len() as u64).to_le_bytes());
    for (slot, entry) in exe.program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = entry else {
            program_table.update([0]);
            continue;
        };
        program_table.update([1]);
        program_table.update((instruction.opcode.as_usize() as u64).to_le_bytes());
        for operand in [
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
            instruction.f,
            instruction.g,
        ] {
            program_table.update(operand.as_canonical_u32().to_le_bytes());
        }
        match chips.pc_to_chip.get(slot) {
            Some(TraceChipIndex::Chip(air)) => {
                program_table.update([1]);
                program_table.update(air.as_u32().to_le_bytes());
            }
            _ => program_table.update([0]),
        }
        let operand = decode.entries.get(slot).copied().unwrap_or_default();
        program_table.update(operand.a.to_le_bytes());
        program_table.update(operand.b.to_le_bytes());
        program_table.update(operand.c.to_le_bytes());
        program_table.update(operand.filtered_index.to_le_bytes());
        program_table.update([
            operand.air_idx,
            operand.local_opcode,
            operand.flags,
            operand.access_pattern,
        ]);
    }
    let program_fingerprint: [u8; 32] = program_table.finalize().into();

    let mut block_table = Sha256::new();
    block_table.update(b"openvm-rvr-g2-static-block-table-v1\0");
    block_table.update((block_entries.len() as u64).to_le_bytes());
    for entry in &block_entries {
        block_table.update(entry.program_slot.to_le_bytes());
        block_table.update(entry.instruction_count.to_le_bytes());
    }
    let block_fingerprint: [u8; 32] = block_table.finalize().into();

    let air_manifest = geometries
        .iter()
        .map(|(&kind, &(air_idx, geometry))| (kind, air_idx, geometry))
        .collect::<Vec<_>>();
    let opaque_bindings = opaque_geometries
        .into_iter()
        .map(|(air_idx, geometry)| {
            let (layout_id, max_residual_events_per_record) = match geometry.layout {
                ArenaNativeLayout::Custom {
                    layout_id,
                    max_residual_events_per_record,
                    ..
                } => (layout_id, max_residual_events_per_record),
                _ => unreachable!("opaque G2 binding must have a custom layout"),
            };
            let mut identity = Sha256::new();
            identity.update(b"openvm-rvr-g2-custom-air-identity-v1\0");
            identity.update(layout_id.as_bytes());
            for opcode in opaque_opcodes.get(&air_idx).into_iter().flatten() {
                identity.update((*opcode as u64).to_le_bytes());
            }
            let mut layout = Sha256::new();
            layout.update(b"openvm-rvr-g2-custom-layout-v1\0");
            layout.update(layout_id.as_bytes());
            RvrG2OpaqueBindingV1 {
                air_idx,
                geometry,
                max_residual_events_per_record,
                air_identity_digest: identity.finalize().into(),
                layout_digest: layout.finalize().into(),
            }
        })
        .collect::<Vec<_>>();
    let air_manifest_fingerprint = g2_air_manifest_fingerprint(&air_manifest, &opaque_bindings)?;

    let mut fingerprint = Sha256::new();
    fingerprint.update(b"openvm-rvr-g2-private-wire-v4-current-replay\0");
    fingerprint.update(1u16.to_le_bytes());
    fingerprint.update(b"header:magic8,version2,header_bytes2,lane_count2,flags2,segment_id4,instruction_count4,run_count4,residual_count4,fingerprint32;");
    fingerprint.update(
        b"lane:kind2,width1,encoding1,flags4,count4,payload_bytes4,offset8,group4,reserved4;",
    );
    fingerprint.update(b"lane:0001,width4,fixed,required,arity=run;lane:0080,width8,fixed,required+atomic,group=2;lane:0081,width1,fixed,required+atomic,group=2;lane:0082,width8,fixed,required+atomic,group=2;lane:0083,width4,fixed,required+atomic,group=2,arity=opaque-occurrence;");
    fingerprint.update(
        b"loadstore:v1-lanes=pointer4+block8;noncrossing=native60+absent-u32max;crossing=residual2x-full-block+native60;",
    );
    for kind in 0u8..rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT as u8 {
        if !rvr_openvm_ext_ffi_common::g2_standard_decoder_kind(kind) {
            continue;
        }
        fingerprint.update([kind, g2_kind_arity(kind)]);
        for value_lane in [false, true] {
            let Some(width) = rvr_openvm_ext_ffi_common::g2_standard_lane_width(kind, value_lane)
            else {
                continue;
            };
            fingerprint.update(if value_lane {
                rvr_openvm_ext_ffi_common::g2_lane_v1(kind).to_le_bytes()
            } else {
                rvr_openvm_ext_ffi_common::g2_lane_v0(kind).to_le_bytes()
            });
            let load_store = rvr_openvm_ext_ffi_common::G2_LOAD_STORE_KINDS.contains(&kind);
            fingerprint.update([width, 0, if load_store { 3 } else { 1 }]);
            fingerprint.update(
                if load_store {
                    rvr_openvm_ext_ffi_common::G2_GROUP_LOAD_STORE
                } else {
                    0
                }
                .to_le_bytes(),
            );
        }
    }
    for opaque in &opaque_bindings {
        fingerprint.update(opaque.lane_kind().to_le_bytes());
        fingerprint.update([0, rvr_openvm_ext_ffi_common::G2_ENCODING_OPAQUE_FINAL]);
        fingerprint.update(rvr_openvm_ext_ffi_common::G2_LANE_FLAG_OPAQUE_FINAL.to_le_bytes());
        fingerprint.update((opaque.air_idx as u64).to_le_bytes());
        fingerprint.update(opaque.air_identity_digest);
        fingerprint.update(opaque.layout_digest);
    }
    fingerprint.update(program_fingerprint);
    fingerprint.update(block_fingerprint);
    fingerprint.update(air_manifest_fingerprint);

    let fingerprint: [u8; 32] = fingerprint.finalize().into();
    Ok(RvrG2MetaV1 {
        fingerprint,
        producer_schema_fingerprint: g2_producer_schema_fingerprint(fingerprint, emission_mode),
        emission_mode: emission_mode as u8,
        program_fingerprint,
        block_fingerprint,
        air_manifest_fingerprint,
        blocks: Arc::new(block_entries),
        block_host_counts: Arc::new(block_host_counts),
        air_bindings: Arc::new(
            decode
                .kind_to_air
                .iter()
                .map(|&(kind, air_idx)| RvrG2AirBindingV1 { kind, air_idx })
                .collect(),
        ),
        opaque_bindings: Arc::new(opaque_bindings),
    })
}

fn g2_producer_schema_fingerprint(
    wire_fingerprint: [u8; 32],
    emission_mode: G2EmissionMode,
) -> [u8; 32] {
    let mut producer_schema = Sha256::new();
    producer_schema.update(b"openvm-rvr-g2-block-span-producer-v4-current-replay\0");
    producer_schema.update(wire_fingerprint);
    producer_schema.update([emission_mode as u8]);
    producer_schema.update(
        b"producer-lane-24;checked-expected-cursors;production-device-replay-cursors;static-run-standard-spans;single-exit-commit;floor-static-timestamp;no-hot-chip-counters;grouped-custom-residual;hint-word-counts;branch-free-production-scratch;",
    );
    producer_schema.finalize().into()
}

fn g2_kind_arity(kind: u8) -> u8 {
    match kind {
        0..=7 | 15..=19 | 29 | 31..=37 => 1,
        8 | 9 | 20..=28 => 2,
        10..=14 => 0,
        _ => u8::MAX,
    }
}

fn g2_air_manifest_fingerprint(
    airs: &[(u8, usize, ArenaNativeGeometry)],
    opaque: &[RvrG2OpaqueBindingV1],
) -> Result<[u8; 32], CompileError> {
    let mut air_manifest = Sha256::new();
    air_manifest.update(b"openvm-rvr-g2-air-registry-manifest-v1\0");
    air_manifest.update(((airs.len() + opaque.len()) as u32).to_le_bytes());
    for &(kind, air_idx, geometry) in airs {
        air_manifest.update([kind]);
        air_manifest.update((air_idx as u64).to_le_bytes());
        air_manifest.update([1]); // Dense arena flavor.
        for value in [
            geometry.adapter_size,
            geometry.adapter_align,
            geometry.core_size,
            geometry.core_align,
            geometry.core_off_matrix,
            geometry.core_off_dense(),
            geometry.stride_dense(),
        ] {
            air_manifest.update((value as u64).to_le_bytes());
        }
        match geometry.layout {
            ArenaNativeLayout::AddI(offsets) if kind == 29 => {
                air_manifest.update([0]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rd_ptr,
                    offsets.rs1_ptr,
                    offsets.read_prev_ts,
                    offsets.write_prev_ts,
                    offsets.write_prev_data,
                    offsets.core_rs1,
                    offsets.core_imm_low11,
                    offsets.core_imm_sign,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
            }
            ArenaNativeLayout::Alu3(offsets) if matches!(kind, 0..=7 | 15..=19) => {
                air_manifest.update([2]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rd_ptr,
                    offsets.rs1_ptr,
                    offsets.rs2,
                    offsets.rs2_as,
                    offsets.rs2_imm_sign,
                    offsets.reads_aux0_prev_ts,
                    offsets.reads_aux1_prev_ts,
                    offsets.write_prev_ts,
                    offsets.write_prev_data,
                    offsets.core_b,
                    offsets.core_c,
                    offsets.core_local_opcode,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
                match offsets.w {
                    Some(w) => {
                        air_manifest.update([1]);
                        for value in [w.rs1_high, w.rs2_high, w.result_word_msl, w.result_sign] {
                            air_manifest.update((value as u64).to_le_bytes());
                        }
                        air_manifest.update([w.result_word_msl_shift, w.result_word_msl_bytes]);
                    }
                    None => air_manifest.update([0]),
                }
            }
            ArenaNativeLayout::AluImm(offsets) if matches!(kind, 31..=37) => {
                air_manifest.update([7]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rd_ptr,
                    offsets.rs1_ptr,
                    offsets.read_prev_ts,
                    offsets.write_prev_ts,
                    offsets.write_prev_data,
                    offsets.rs1_high,
                    offsets.result_high,
                    offsets.core_b,
                    offsets.core_imm_low11,
                    offsets.core_imm_sign,
                    offsets.core_shamt,
                    offsets.core_local_opcode,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
                air_manifest.update([offsets.core_b_limb_bytes, offsets.core_b_limb_count]);
            }
            ArenaNativeLayout::Branch2(offsets) if matches!(kind, 10 | 11) => {
                air_manifest.update([3]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rs1_ptr,
                    offsets.rs2_ptr,
                    offsets.reads_aux0_prev_ts,
                    offsets.reads_aux1_prev_ts,
                    offsets.core_a,
                    offsets.core_b,
                    offsets.core_imm,
                    offsets.core_local_opcode,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
            }
            ArenaNativeLayout::Wr1(offsets) if matches!(kind, 12 | 14) => {
                air_manifest.update([4]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rd_ptr,
                    offsets.rd_prev_ts,
                    offsets.rd_prev_data,
                    offsets.core_imm,
                    offsets.core_rd_data,
                    offsets.core_is_jal,
                    offsets.core_from_pc,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
            }
            ArenaNativeLayout::Rw1(offsets) if kind == 13 => {
                air_manifest.update([5]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rs1_ptr,
                    offsets.rd_ptr,
                    offsets.read_prev_ts,
                    offsets.write_prev_ts,
                    offsets.write_prev_data,
                    offsets.core_imm,
                    offsets.core_from_pc,
                    offsets.core_rs1_val,
                    offsets.core_imm_sign,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
            }
            ArenaNativeLayout::LoadStore(offsets) if kind != 29 => {
                air_manifest.update([1]);
                for value in [
                    offsets.from_pc,
                    offsets.from_timestamp,
                    offsets.rs1_ptr,
                    offsets.rs1_val,
                    offsets.rs1_aux_prev_ts,
                    offsets.rd_rs2_ptr,
                    offsets.read_data_aux_prev_ts,
                    offsets.imm,
                    offsets.imm_sign,
                    offsets.mem_as,
                    offsets.write_prev_ts,
                    offsets.write_prev_data,
                    offsets.core_local_opcode,
                    offsets.core_is_byte,
                    offsets.core_is_word,
                    offsets.core_shift_amount,
                    offsets.core_read_data,
                    offsets.core_prev_data,
                ] {
                    air_manifest.update((value as u64).to_le_bytes());
                }
            }
            ArenaNativeLayout::CustomVariableRows {
                residual_memory_chronology: true,
            } if kind == 30 => {
                air_manifest.update([6]);
            }
            _ => {
                return Err(CompileError::InvalidOptions(
                    "G2 registry manifest contains an unsupported layout",
                ));
            }
        }
    }
    for binding in opaque {
        air_manifest.update([u8::MAX]);
        air_manifest.update((binding.air_idx as u64).to_le_bytes());
        air_manifest.update([1]); // Dense arena flavor.
        for value in [
            binding.geometry.adapter_size,
            binding.geometry.adapter_align,
            binding.geometry.core_size,
            binding.geometry.core_align,
            binding.geometry.core_off_matrix,
            binding.geometry.core_off_dense(),
            binding.geometry.stride_dense(),
        ] {
            air_manifest.update((value as u64).to_le_bytes());
        }
        air_manifest.update(binding.air_identity_digest);
        air_manifest.update(binding.layout_digest);
        air_manifest.update(binding.max_residual_events_per_record.to_le_bytes());
    }
    Ok(air_manifest.finalize().into())
}

/// Fail cheaply during native-project preparation if the requested GPU
/// record shape and the shape encoded in compile metadata diverge. Under a
/// compact request every inline AIR (including mixed LoadStore/REVEAL) must
/// remain wire-shaped; the GPU decode router may select any subset of them.
fn validate_requested_inline_record_shape(
    inline_meta: &RvrInlineRecordsMeta,
) -> Result<(), CompileError> {
    let requested = std::env::var("OPENVM_RVR_GPU_RECORDS").ok();
    let custom_only = || {
        inline_meta.arena_native_airs.iter().all(|(_, geometry)| {
            matches!(
                geometry.layout,
                ArenaNativeLayout::Custom {
                    residual_memory_chronology: true,
                    ..
                }
            )
        })
    };
    let invalid_arena = requested.as_deref() == Some("compact")
        || requested.as_deref() == Some("g2") && inline_meta.g2.is_some() && !custom_only()
        || requested.as_deref() == Some("delta")
            && inline_meta.arena_native_airs.iter().any(|(_, geometry)| {
                !matches!(
                    geometry.layout,
                    ArenaNativeLayout::Custom {
                        residual_memory_chronology: true,
                        ..
                    }
                ) && !matches!(
                    geometry.layout,
                    ArenaNativeLayout::CustomVariableRows {
                        residual_memory_chronology: true
                    }
                )
            });
    if invalid_arena && !inline_meta.arena_native_airs.is_empty() {
        return Err(CompileError::InvalidOptions(
            "requested GPU record shape produced an incompatible arena-native AIR",
        ));
    }
    Ok(())
}

/// Compact record stride per wire shape (the C-side `_Static_assert`s guard
/// the layouts against these same constants).
fn inline_record_shape_size(shape: InlineRecordShape) -> usize {
    match shape {
        InlineRecordShape::Alu3 => rvr_openvm_ext_ffi_common::PREFLIGHT_ADDSUB_RECORD_SIZE,
        InlineRecordShape::Branch2 => rvr_openvm_ext_ffi_common::PREFLIGHT_BRANCH2_RECORD_SIZE,
        InlineRecordShape::Wr1 => rvr_openvm_ext_ffi_common::PREFLIGHT_WR1_RECORD_SIZE,
        InlineRecordShape::Rw1 => rvr_openvm_ext_ffi_common::PREFLIGHT_RW1_RECORD_SIZE,
        InlineRecordShape::Custom { record_size } => record_size,
        InlineRecordShape::CustomVariableRows { capacity_per_row } => capacity_per_row,
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
    compile_impl(exe, &opts, None)
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
        None,
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
        None,
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
        None,
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
        None,
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
        None,
    )
}

/// Compile an executable handled by the base system lifter only.
///
/// RV64I/M callers should use [`compile_preflight_with_extensions`] after
/// #3059 moved those instructions into extensions.
pub fn compile_preflight<F: PrimeField32>(
    exe: &VmExe<F>,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    let extensions = ExtensionRegistry::new();
    compile_preflight_with_extensions(exe, &extensions, &(), chips, guest_debug_map)
}

/// Compile a registered-RVR executable with the preflight tracer.
pub fn compile_preflight_with_extensions<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    assembler_admitter: &dyn LogNativeOpcodeAdmitter<F>,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
) -> Result<RvrCompiled, CompileError> {
    compile_preflight_with_extensions_and_default(
        exe,
        extensions,
        assembler_admitter,
        chips,
        guest_debug_map,
        None,
    )
}

pub(crate) fn compile_preflight_with_extensions_and_default<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry,
    assembler_admitter: &dyn LogNativeOpcodeAdmitter<F>,
    chips: &ChipMapping,
    guest_debug_map: Option<&GuestDebugMap>,
    gpu_records_default: Option<&str>,
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
        gpu_records_default,
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
    let num_airs = load_num_airs(&lib, execution_kind)?;
    Ok(RvrCompiled {
        lib,
        lib_path: lib_path.to_path_buf(),
        artifact_dir: None,
        execution_kind,
        num_airs,
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

struct RvrNativeCache {
    lib_path: PathBuf,
    key_path: PathBuf,
    input_key: Option<String>,
    manifest: Option<RvrNativeCacheManifest>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RvrNativeCacheManifest {
    project_key: String,
    artifact_key: String,
    /// Cheap fingerprint of the VmExe, instantiated codegen config, native
    /// build identity, and generator binary. Absent in legacy manifests.
    input_key: Option<String>,
}

const DEFAULT_THINLTO_JOBS_MAX: usize = 32;
const THINLTO_CACHE_VERSION: &str = "v1";

#[derive(Clone, Debug, PartialEq, Eq)]
struct ThinLtoBuildOptions {
    jobs: usize,
    cache_dir: Option<PathBuf>,
}

fn compile_impl<F: PrimeField32>(
    exe: &VmExe<F>,
    opts: &CompileOptions<'_, F>,
    gpu_records_default: Option<&str>,
) -> Result<RvrCompiled, CompileError> {
    let prepare_started = Instant::now();
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
    if let Some(launcher) = toolchain.unavailable_compiler_launcher.as_deref() {
        tracing::warn!(
            launcher,
            compiler = %toolchain.compiler,
            "configured rvr compiler launcher is unavailable; using the compiler directly"
        );
    }

    let base_name = sanitize_base_name(opts.base_name.unwrap_or("openvm"));
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
        tracing::info!(
            execution_kind = ?opts.execution_kind,
            "disabled ThinLTO for rvr native compilation"
        );
    }
    let thinlto_jobs = (!disable_lto)
        .then(configured_thinlto_jobs)
        .transpose()?
        .unwrap_or(1);
    // Preflight codegen emits inline records by default. The metadata is
    // reconstructed from the lifted instruction stream even on a cache hit;
    // only CFG/project generation and the recursive project hash are skipped.
    let inline_records = opts.execution_kind == RvrExecutionKind::Preflight
        && !env_flag_is_off("OPENVM_RVR_INLINE_RECORDS");
    if opts.execution_kind == RvrExecutionKind::Preflight
        && matches!(
            std::env::var("OPENVM_RVR_GPU_RECORDS").as_deref(),
            Ok("compact" | "delta" | "g2")
        )
        && !inline_records
    {
        return Err(CompileError::InvalidOptions(
            "OPENVM_RVR_GPU_RECORDS=compact|delta|g2 requires inline record emission",
        ));
    }
    let (chips, num_airs) = match opts.execution_kind {
        RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => (None, None),
        RvrExecutionKind::Metered
        | RvrExecutionKind::MeteredSegment
        | RvrExecutionKind::MeteredCost
        | RvrExecutionKind::Preflight => {
            let chips = opts.chips.ok_or(CompileError::InvalidOptions(
                "metered/preflight rvr compile requires ChipMapping",
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
            (Some(chips), Some(num_airs))
        }
    };
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
    let native_opt = opt_env
        .map(|(env, invalid_message)| native_opt_level(env, invalid_message))
        .transpose()?
        .flatten();
    let native_detail = opts.execution_kind == RvrExecutionKind::Preflight
        && std::env::var("OPENVM_RVR_NATIVE_DETAIL").as_deref() == Ok("1");

    let ir_started = Instant::now();
    let ir = convert_vmexe_to_ir_with_debug(exe, opts.extensions, |pc| {
        opts.guest_debug_map
            .and_then(|debug_map| debug_map.get(pc).cloned())
    })?;
    let ir_elapsed = ir_started.elapsed();

    let inline_meta_started = Instant::now();
    let mut inline_meta = RvrInlineRecordsMeta::default();
    if inline_records {
        inline_meta = collect_inline_records_meta(
            exe,
            &ir,
            chips.expect("preflight chip mapping checked above"),
            opts.preflight_assembler_admitter,
            gpu_records_default,
        );
        validate_requested_inline_record_shape(&inline_meta)?;
    }
    let inline_meta_elapsed = inline_meta_started.elapsed();

    // G2 binds the exact generated basic-block table, so it performs CFG
    // construction before a cache lookup. Established routes retain their
    // fast cache-hit path and do not pay this scan.
    let g2_negotiated = configured_gpu_records(gpu_records_default).as_deref() == Some("g2")
        && inline_meta
            .delta_decode
            .as_ref()
            .is_some_and(|precomputed| {
                !precomputed.kind_to_air.is_empty()
                    && precomputed
                        .kind_to_air
                        .iter()
                        .all(|&(kind, _)| g2_decoder_kind(kind))
            });
    let mut prebuilt_blocks = None;
    if g2_negotiated {
        let emission_mode = configured_g2_emission_mode()?;
        let valid_pcs: std::collections::HashSet<u64> = ir.iter().map(|li| li.pc()).collect();
        let extra_targets = opts
            .extensions
            .extra_cfg_targets(&exe.init_memory, &valid_pcs);
        let blocks = build_blocks(&ir, &extra_targets);
        inline_meta.g2 = Some(Arc::new(build_g2_meta_v1(
            exe,
            chips.expect("G2 preflight chip mapping checked above"),
            inline_meta
                .delta_decode
                .as_deref()
                .expect("G2 negotiation requires operand metadata"),
            &blocks,
            opts.preflight_assembler_admitter
                .expect("G2 negotiation requires the assembler registry"),
            emission_mode,
        )?));
        prebuilt_blocks = Some(blocks);
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
    let mut input_key_elapsed = Duration::ZERO;
    let cache_path =
        cache_env.and_then(|cache_env| std::env::var_os(cache_env).filter(|path| !path.is_empty()));
    let cache = if let Some(path) = cache_path {
        let lib_path = PathBuf::from(path);
        let key_path = preflight_cache_key_path(&lib_path);
        let input_key_started = Instant::now();
        let input_key = generated_project_input_cache_key(
            exe,
            &ir,
            opts,
            chips,
            &base_name,
            disable_lto,
            inline_records,
            native_detail,
            native_opt.as_deref(),
            &inline_meta,
            &toolchain,
        )?;
        input_key_elapsed = input_key_started.elapsed();
        if input_key.is_none() {
            tracing::info!(
                path = %lib_path.display(),
                execution_kind = ?opts.execution_kind,
                "rvr input cache disabled by an extension without a canonical fingerprint"
            );
        }
        let manifest = read_preflight_cache_manifest(&key_path);
        Some(RvrNativeCache {
            lib_path,
            key_path,
            input_key,
            manifest,
        })
    } else {
        None
    };
    let thinlto_cache_root = if disable_lto {
        None
    } else {
        configured_thinlto_cache_root(cache.as_ref())?
    };

    if let Some(cache) = cache.as_ref() {
        if let (Some(input_key), Some(manifest)) = (
            cache.input_key.as_deref(),
            cache
                .manifest
                .as_ref()
                .filter(|manifest| manifest.input_key.as_deref() == cache.input_key.as_deref()),
        ) {
            let artifact_started = Instant::now();
            if let Some(mut compiled) =
                load_verified_preflight_cache_copy(&cache.lib_path, &manifest.artifact_key)?
            {
                let artifact_elapsed = artifact_started.elapsed();
                tracing::info!(
                    path = %cache.lib_path.display(),
                    input_key = %input_key,
                    project_key = %manifest.project_key,
                    execution_kind = ?opts.execution_kind,
                    "loading input-validated rvr native artifact without project regeneration"
                );
                if std::env::var("OPENVM_RVR_CACHE_PROFILE").as_deref() == Ok("1") {
                    eprintln!(
                        "OPENVM_RVR_CACHE_HIT_TIMING mode={:?} total_us={} ir_us={} \
                         inline_meta_us={} input_key_us={} artifact_us={}",
                        opts.execution_kind,
                        prepare_started.elapsed().as_micros(),
                        ir_elapsed.as_micros(),
                        inline_meta_elapsed.as_micros(),
                        input_key_elapsed.as_micros(),
                        artifact_elapsed.as_micros(),
                    );
                }
                compiled.inline_records = inline_meta;
                compiled.validate_g2_manifest()?;
                return Ok(compiled);
            }
        }
        tracing::info!(
            path = %cache.lib_path.display(),
            input_key = ?cache.input_key,
            execution_kind = ?opts.execution_kind,
            "rvr native artifact input cache miss; regenerating project for validation"
        );
        if std::env::var("OPENVM_RVR_CACHE_PROFILE").as_deref() == Ok("1") {
            eprintln!(
                "OPENVM_RVR_CACHE_MISS mode={:?} input_key={}",
                opts.execution_kind,
                cache.input_key.as_deref().unwrap_or("disabled"),
            );
        }
    }

    // CFG construction scans the complete initial memory image for indirect
    // code pointers and dominates cache-hit preparation for large guests.
    // It is needed only when a project must actually be regenerated.
    let cfg_started = Instant::now();
    let blocks = if let Some(blocks) = prebuilt_blocks {
        blocks
    } else {
        let valid_pcs: std::collections::HashSet<u64> = ir.iter().map(|li| li.pc()).collect();
        let extra_targets = opts
            .extensions
            .extra_cfg_targets(&exe.init_memory, &valid_pcs);
        build_blocks(&ir, &extra_targets)
    };
    let cfg_elapsed = cfg_started.elapsed();

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
    if disable_lto {
        project.enable_lto = false;
    }
    project.pc_base = u64::from(exe.program.pc_base);
    let mut next_exec_idx = 0u32;
    project.pc_to_exec_idx = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .map(|slot| {
            if slot.is_some() {
                let idx = next_exec_idx;
                next_exec_idx = next_exec_idx
                    .checked_add(1)
                    .expect("defined program instruction count exceeds u32");
                idx
            } else {
                u32::MAX
            }
        })
        .collect();
    project.native_detail_pc_families = exe
        .program
        .instructions_and_debug_infos
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map_or(8, |(instruction, _)| native_detail_family(instruction))
        })
        .collect();

    // R3: preflight compiles emit inline compact records (with memory-log
    // suppression) for migrated opcodes by default; OPENVM_RVR_INLINE_RECORDS
    // set to 0/false/off opts out (verbose log + host assembler for every
    // opcode). The collected metadata mirrors the codegen decision exactly so
    // the host skips the assembler for precisely the suppressed pcs.
    match opts.execution_kind {
        RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => {}
        RvrExecutionKind::Metered
        | RvrExecutionKind::MeteredSegment
        | RvrExecutionKind::MeteredCost
        | RvrExecutionKind::Preflight => {
            let chips = chips.expect("chip mapping checked above");
            project.num_airs = num_airs;
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
                RvrExecutionKind::Preflight => {
                    project.chip_widths = None;
                }
            }
            if inline_records {
                project.inline_records = true;
                project.inline_pc_slots = (*inline_meta.pc_slots).clone();
                project.delta_records = inline_meta.delta_records;
                project.arena_native_airs = inline_meta
                    .arena_native_airs
                    .iter()
                    .map(|&(air, geometry)| (air as u32, geometry))
                    .collect();
                if let Some(g2) = inline_meta.g2.as_deref() {
                    let mut air_kinds = [u8::MAX; rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT];
                    let mut air_indices =
                        [u32::MAX; rvr_openvm_ext_ffi_common::G2_DECODER_KIND_COUNT];
                    for (index, binding) in g2.air_bindings.iter().enumerate() {
                        air_kinds[index] = binding.kind;
                        air_indices[index] =
                            u32::try_from(binding.air_idx).expect("G2 AIR index exceeds u32");
                    }
                    project.g2_records = true;
                    project.g2_emission_mode = match g2.emission_mode {
                        value if value == G2EmissionMode::Checked as u8 => G2EmissionMode::Checked,
                        value if value == G2EmissionMode::Production as u8 => {
                            G2EmissionMode::Production
                        }
                        value => panic!("invalid persisted G2 emission mode {value}"),
                    };
                    let decode = inline_meta
                        .delta_decode
                        .as_deref()
                        .expect("G2 metadata requires a persisted operand table");
                    project.g2_pc_kinds = decode
                        .entries
                        .iter()
                        .map(|entry| {
                            g2.air_bindings
                                .iter()
                                .find(|binding| binding.air_idx == entry.air_idx as usize)
                                .map_or(u8::MAX, |binding| binding.kind)
                        })
                        .collect();
                    project.g2_pc_arities = decode
                        .entries
                        .iter()
                        .map(|entry| {
                            let kind = g2.air_bindings.iter().find_map(|binding| {
                                (binding.air_idx == entry.air_idx as usize).then_some(binding.kind)
                            });
                            match kind {
                                Some(0..=7 | 15..=19 | 29 | 31..=37) => 1,
                                Some(8 | 9 | 20..=28) => 2,
                                Some(_) | None => 0,
                            }
                        })
                        .collect();
                    project.g2_manifest = Some(G2DsoManifestConfigV2 {
                        fingerprint: g2.fingerprint,
                        producer_schema_fingerprint: g2.producer_schema_fingerprint,
                        emission_mode: u32::from(g2.emission_mode),
                        program_fingerprint: g2.program_fingerprint,
                        block_fingerprint: g2.block_fingerprint,
                        air_manifest_fingerprint: g2.air_manifest_fingerprint,
                        pc_base: exe.program.pc_base,
                        block_count: u32::try_from(g2.blocks.len())
                            .expect("G2 static block count exceeds u32"),
                        air_count: u32::try_from(g2.air_bindings.len())
                            .expect("G2 AIR binding count exceeds u32"),
                        air_kinds,
                        air_indices,
                    });
                }
            }
        }
    }

    project.native_debug_info = opts.native_debug_info;
    project.native_detail = native_detail;

    let entry_point = u64::from(exe.pc_start);
    let text_start = u64::from(exe.program.pc_base);
    let emit_started = Instant::now();
    project
        .write_all(&blocks, entry_point, text_start, opts.extensions)
        .map_err(|source| CompileError::CProject {
            path: output_dir.to_path_buf(),
            source,
        })?;

    let ext_staticlibs = write_extension_staticlibs(output_dir, opts.extensions)?;
    let emit_elapsed = emit_started.elapsed();
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

    let mut make_args = project.make_args_with_extensions(
        &ext_staticlibs,
        &ext_sources,
        &vendor_sources,
        &ext_cflags,
    );
    if disable_lto {
        // Override an inherited Make environment variable as well as CProject's default.
        make_args.push("LTO=".to_string());
    }
    if let Some(opt) = native_opt.as_ref() {
        make_args.push(format!("OPT={opt}"));
    }

    let hash_started = Instant::now();
    let project_key = (cache.is_some() || thinlto_cache_root.is_some())
        .then(|| generated_project_cache_key(output_dir, &make_args, &toolchain))
        .transpose()?;
    let hash_elapsed = hash_started.elapsed();
    if cache.is_some() || thinlto_cache_root.is_some() {
        tracing::info!(
            execution_kind = ?opts.execution_kind,
            ir_ms = ir_elapsed.as_millis(),
            cfg_ms = cfg_elapsed.as_millis(),
            emit_ms = emit_elapsed.as_millis(),
            project_hash_ms = hash_elapsed.as_millis(),
            total_ms = prepare_started.elapsed().as_millis(),
            "rvr generated-project cache preparation breakdown"
        );
    }

    if let (Some(cache), Some(expected_key)) = (cache.as_ref(), project_key.as_ref()) {
        if let Some(manifest) = cache
            .manifest
            .as_ref()
            .filter(|manifest| manifest.project_key == *expected_key)
        {
            if let Some(mut compiled) =
                load_verified_preflight_cache_copy(&cache.lib_path, &manifest.artifact_key)?
            {
                write_preflight_cache_manifest(
                    &cache.key_path,
                    expected_key,
                    &manifest.artifact_key,
                    cache.input_key.as_deref(),
                )?;
                tracing::info!(
                    path = %cache.lib_path.display(),
                    input_key = ?cache.input_key,
                    project_key = %expected_key,
                    execution_kind = ?opts.execution_kind,
                    "loading project-validated rvr artifact and upgrading input cache"
                );
                compiled.inline_records = inline_meta;
                compiled.validate_g2_manifest()?;
                return Ok(compiled);
            }
        }
        tracing::info!(
            path = %cache.lib_path.display(),
            cache_key = %expected_key,
            execution_kind = ?opts.execution_kind,
            "rvr native artifact cache miss"
        );
    }

    let thinlto = ThinLtoBuildOptions {
        jobs: thinlto_jobs,
        cache_dir: thinlto_cache_root
            .zip(project_key.as_deref())
            .map(|(root, key)| root.join(THINLTO_CACHE_VERSION).join(key)),
    };
    compile_generated_project(output_dir, &make_args, &toolchain, &thinlto)?;

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

    let compiled = RvrCompiled {
        lib,
        lib_path,
        artifact_dir: Some(artifact_dir),
        execution_kind,
        num_airs,
        inline_records: inline_meta,
    };
    compiled.validate_g2_manifest()?;
    if let (Some(cache), Some(cache_key)) = (cache, project_key) {
        publish_preflight_cache(
            &compiled,
            &cache.lib_path,
            &cache.key_path,
            &cache_key,
            cache.input_key.as_deref(),
        )?;
        tracing::info!(
            path = %cache.lib_path.display(),
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

fn configured_thinlto_jobs() -> Result<usize, CompileError> {
    thinlto_jobs(
        std::env::var_os("OPENVM_RVR_THINLTO_JOBS").as_deref(),
        std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get),
    )
}

fn thinlto_jobs(
    value: Option<&OsStr>,
    available_parallelism: usize,
) -> Result<usize, CompileError> {
    let Some(value) = value else {
        return Ok(available_parallelism.clamp(1, DEFAULT_THINLTO_JOBS_MAX));
    };
    value
        .to_str()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|jobs| *jobs > 0)
        .ok_or(CompileError::InvalidOptions(
            "OPENVM_RVR_THINLTO_JOBS must be a positive integer",
        ))
}

fn configured_thinlto_cache_root(
    cache: Option<&RvrNativeCache>,
) -> Result<Option<PathBuf>, CompileError> {
    let configured = std::env::var_os("OPENVM_RVR_THINLTO_CACHE_DIR");
    let cache_root = cache.and_then(|cache| cache.lib_path.parent());
    thinlto_cache_root(configured.as_deref(), cache_root)
        .map(|root| {
            if root.is_absolute() {
                Ok(root)
            } else {
                std::env::current_dir()
                    .map(|cwd| resolve_cache_root(root.clone(), &cwd))
                    .map_err(|source| CompileError::CProject { path: root, source })
            }
        })
        .transpose()
}

fn thinlto_cache_root(configured: Option<&OsStr>, cache_root: Option<&Path>) -> Option<PathBuf> {
    match configured {
        Some(path) if path.is_empty() => None,
        Some(path) => Some(PathBuf::from(path)),
        None => cache_root.map(|root| root.join("thinlto-cache")),
    }
}

fn resolve_cache_root(root: PathBuf, current_dir: &Path) -> PathBuf {
    if root.is_absolute() {
        root
    } else {
        current_dir.join(root)
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

fn read_preflight_cache_manifest(path: &Path) -> Option<RvrNativeCacheManifest> {
    let manifest = fs::read_to_string(path).ok()?;
    let mut project_key = None;
    let mut artifact_key = None;
    let mut input_key = None;
    for line in manifest.lines() {
        if let Some(value) = line.strip_prefix("project=") {
            if project_key.replace(value.to_string()).is_some() {
                return None;
            }
        } else if let Some(value) = line.strip_prefix("artifact=") {
            if artifact_key.replace(value.to_string()).is_some() {
                return None;
            }
        } else if let Some(value) = line.strip_prefix("input=") {
            if input_key.replace(value.to_string()).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }
    Some(RvrNativeCacheManifest {
        project_key: project_key?,
        artifact_key: artifact_key?,
        input_key,
    })
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
    let num_airs = load_num_airs(&lib, execution_kind)?;
    Ok(Some(RvrCompiled {
        lib,
        lib_path: private_lib,
        artifact_dir: Some(ArtifactDir::Temp(temp_dir)),
        execution_kind,
        num_airs,
        inline_records: RvrInlineRecordsMeta::default(),
    }))
}

fn publish_preflight_cache(
    compiled: &RvrCompiled,
    cache_lib: &Path,
    cache_key_path: &Path,
    project_key: &str,
    input_key: Option<&str>,
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

    temp_lib
        .persist(cache_lib)
        .map_err(|err| CompileError::CProject {
            path: cache_lib.to_path_buf(),
            source: err.error,
        })?;
    write_preflight_cache_manifest(cache_key_path, project_key, &artifact_key, input_key)
}

fn write_preflight_cache_manifest(
    cache_key_path: &Path,
    project_key: &str,
    artifact_key: &str,
    input_key: Option<&str>,
) -> Result<(), CompileError> {
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
    let mut manifest = format!("project={project_key}\nartifact={artifact_key}\n");
    if let Some(input_key) = input_key {
        manifest.push_str(&format!("input={input_key}\n"));
    }
    fs::write(temp_manifest.path(), manifest).map_err(|source| CompileError::CProject {
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
    temp_manifest
        .persist(cache_key_path)
        .map_err(|err| CompileError::CProject {
            path: cache_key_path.to_path_buf(),
            source: err.error,
        })?;
    Ok(())
}

/// Cheap, fail-closed identity for the inputs that determine a generated RVR
/// project. Unlike [`generated_project_cache_key`], this never writes C files.
/// The full project key remains the authority on a miss; a successful slow
/// validation upgrades the manifest with this input key for subsequent runs.
#[allow(clippy::too_many_arguments)]
fn generated_project_input_cache_key<F: PrimeField32>(
    exe: &VmExe<F>,
    ir: &[LiftedInstr],
    opts: &CompileOptions<'_, F>,
    chips: Option<&ChipMapping>,
    base_name: &str,
    disable_lto: bool,
    inline_records: bool,
    native_detail: bool,
    native_opt: Option<&str>,
    inline_meta: &RvrInlineRecordsMeta,
    toolchain: &rvr_openvm::RuntimeToolchain,
) -> Result<Option<String>, CompileError> {
    let Some(extension_fingerprints) = opts.extensions.codegen_fingerprints() else {
        return Ok(None);
    };
    let mut hasher = Sha256::new();
    hasher.update(b"openvm-rvr-generated-project-input-v3-execution-kind\0");
    update_framed(&mut hasher, generator_binary_sha256()?.as_bytes());
    update_framed(&mut hasher, std::env::consts::OS.as_bytes());
    update_framed(&mut hasher, std::env::consts::ARCH.as_bytes());
    update_framed(&mut hasher, base_name.as_bytes());
    hasher.update((opts.execution_kind as u32).to_le_bytes());
    hasher.update([
        disable_lto as u8,
        inline_records as u8,
        opts.native_debug_info as u8,
        native_detail as u8,
    ]);
    update_optional(&mut hasher, native_opt.map(str::as_bytes));

    // VmExe hash. Program DebugInfo is intentionally excluded: the lifter
    // ignores it and native source locations arrive through guest_debug_map,
    // which is represented in the lifted IR below.
    hasher.update(exe.program.pc_base.to_le_bytes());
    hasher.update(exe.pc_start.to_le_bytes());
    hasher.update((exe.program.instructions_and_debug_infos.len() as u64).to_le_bytes());
    for entry in &exe.program.instructions_and_debug_infos {
        let Some((instruction, _)) = entry else {
            hasher.update([0]);
            continue;
        };
        hasher.update([1]);
        hasher.update((instruction.opcode.as_usize() as u64).to_le_bytes());
        for operand in [
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
            instruction.f,
            instruction.g,
        ] {
            hasher.update(operand.as_canonical_u32().to_le_bytes());
        }
    }
    hasher.update((exe.init_memory.len() as u64).to_le_bytes());
    for (&(address_space, address), &value) in &exe.init_memory {
        hasher.update(address_space.to_le_bytes());
        hasher.update(address.to_le_bytes());
        hasher.update([value]);
    }
    hasher.update((exe.fn_bounds.len() as u64).to_le_bytes());
    for (&pc, bound) in &exe.fn_bounds {
        hasher.update(pc.to_le_bytes());
        hasher.update(bound.start.to_le_bytes());
        hasher.update(bound.end.to_le_bytes());
        update_framed(&mut hasher, bound.name.as_bytes());
    }

    if let Some(chips) = chips {
        hasher.update([1]);
        hasher.update((chips.pc_to_chip.len() as u64).to_le_bytes());
        for chip in &chips.pc_to_chip {
            match chip {
                TraceChipIndex::Chip(air) => {
                    hasher.update([1]);
                    hasher.update(air.as_u32().to_le_bytes());
                }
                TraceChipIndex::NoChip => hasher.update([0]),
            }
        }
        match chips.chip_widths.as_ref() {
            Some(widths) => {
                hasher.update([1]);
                hasher.update((widths.len() as u64).to_le_bytes());
                for width in widths {
                    hasher.update(width.to_le_bytes());
                }
            }
            None => hasher.update([0]),
        }
    } else {
        hasher.update([0]);
    }

    // Source locations cover the guest debug map because they change
    // generated #line directives.
    for lifted in ir {
        match lifted {
            LiftedInstr::Body(instr) => {
                update_source_location(&mut hasher, instr.source_loc.as_ref());
            }
            LiftedInstr::Term { source_loc, .. } => {
                update_source_location(&mut hasher, source_loc.as_ref());
            }
        }
    }

    // Canonical instance state for every extension, in registration order.
    // Unknown/custom extensions default to `None` above and cannot use the
    // fast path until they explicitly implement the fingerprint contract.
    hasher.update((extension_fingerprints.len() as u64).to_le_bytes());
    for fingerprint in extension_fingerprints {
        update_framed(&mut hasher, &fingerprint);
    }

    // The inline metadata is both host-consumed and baked into preflight C.
    hasher.update((inline_meta.pc_slots.len() as u64).to_le_bytes());
    let mut packed = 0u8;
    for (idx, &enabled) in inline_meta.pc_slots.iter().enumerate() {
        packed |= (enabled as u8) << (idx % 8);
        if idx % 8 == 7 {
            hasher.update([packed]);
            packed = 0;
        }
    }
    if !inline_meta.pc_slots.len().is_multiple_of(8) {
        hasher.update([packed]);
    }
    update_debug(&mut hasher, &inline_meta.airs)?;
    update_debug(&mut hasher, &inline_meta.arena_native_airs)?;
    hasher.update([inline_meta.delta_records as u8]);
    update_debug(&mut hasher, &inline_meta.delta_decode)?;
    hasher.update([inline_meta.fully_direct_delta as u8]);
    if let Some(g2) = inline_meta.g2.as_deref() {
        hasher.update([1]);
        hasher.update(g2.fingerprint);
        hasher.update(g2.producer_schema_fingerprint);
        hasher.update([g2.emission_mode]);
        update_debug(&mut hasher, &g2.blocks)?;
        update_debug(&mut hasher, &g2.air_bindings)?;
    } else {
        hasher.update([0]);
    }

    // Registry selection and embedded assets can vary independently of the
    // VmExe. Hash their actual bytes so dynamically linked consumers retain
    // the same fail-closed property as statically linked binaries.
    update_named_text_assets(&mut hasher, b"headers", opts.extensions.c_headers());
    update_named_text_assets(&mut hasher, b"sources", opts.extensions.c_sources());
    update_named_text_assets(
        &mut hasher,
        b"vendored-sources",
        opts.extensions.vendored_c_sources(),
    );
    update_named_text_assets(
        &mut hasher,
        b"extra-includes",
        opts.extensions.extra_c_include_files(),
    );
    hasher.update(b"staticlibs\0");
    for (name, contents) in opts.extensions.staticlib_files() {
        update_framed(&mut hasher, name.as_bytes());
        update_framed(&mut hasher, contents);
    }
    hasher.update(b"cflags\0");
    for flag in opts.extensions.extra_cflags() {
        update_framed(&mut hasher, flag.as_bytes());
    }

    update_native_build_identity(&mut hasher, toolchain)?;
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
        update_framed(&mut hasher, name.as_bytes());
        update_optional(
            &mut hasher,
            std::env::var_os(name)
                .as_ref()
                .map(|value| value.as_encoded_bytes()),
        );
    }

    Ok(Some(hex_digest(hasher.finalize())))
}

fn generator_binary_sha256() -> Result<&'static str, CompileError> {
    static KEY: OnceLock<Result<String, String>> = OnceLock::new();
    match KEY.get_or_init(|| {
        let path = std::env::current_exe()
            .map_err(|err| format!("failed to locate current executable: {err}"))?;
        artifact_sha256(&path).map_err(|err| err.to_string())
    }) {
        Ok(key) => Ok(key),
        Err(err) => Err(CompileError::LibLoad(err.clone())),
    }
}

fn update_native_build_identity(
    hasher: &mut Sha256,
    toolchain: &rvr_openvm::RuntimeToolchain,
) -> Result<(), CompileError> {
    for value in [
        toolchain.compiler.as_str(),
        toolchain.linker.as_str(),
        toolchain.make.as_str(),
        toolchain.host_os,
    ] {
        update_framed(hasher, value.as_bytes());
    }
    update_command_identity(hasher, &toolchain.compiler, &["--version"])?;
    update_command_identity(hasher, &toolchain.linker, &["--version"])?;
    update_command_identity(hasher, &toolchain.make, &["--version"])?;
    update_command_identity(
        hasher,
        &toolchain.compiler,
        &["-march=native", "-###", "-E", "-x", "c", "-"],
    )
}

fn update_named_text_assets(
    hasher: &mut Sha256,
    category: &[u8],
    assets: Vec<(&'static str, &'static str)>,
) {
    update_framed(hasher, category);
    for (name, contents) in assets {
        update_framed(hasher, name.as_bytes());
        update_framed(hasher, contents.as_bytes());
    }
}

fn update_source_location(hasher: &mut Sha256, location: Option<&SourceLoc>) {
    let Some(location) = location else {
        hasher.update([0]);
        return;
    };
    hasher.update([1]);
    update_framed(hasher, location.file.as_bytes());
    hasher.update(location.line.to_le_bytes());
    update_framed(hasher, location.function.as_bytes());
}

fn update_debug(hasher: &mut Sha256, value: &dyn fmt::Debug) -> Result<(), CompileError> {
    struct DigestWriter<'a>(&'a mut Sha256);
    impl fmt::Write for DigestWriter<'_> {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            self.0.update(value.as_bytes());
            Ok(())
        }
    }
    let mut value_hasher = Sha256::new();
    write!(DigestWriter(&mut value_hasher), "{value:?}")
        .map_err(|_| CompileError::LibLoad("failed to hash rvr codegen configuration".into()))?;
    hasher.update(value_hasher.finalize());
    Ok(())
}

fn update_optional(hasher: &mut Sha256, value: Option<&[u8]>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_framed(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn update_framed(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
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
    thinlto: &ThinLtoBuildOptions,
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
    if let Some(cache_dir) = thinlto.cache_dir.as_ref() {
        fs::create_dir_all(cache_dir).map_err(|source| CompileError::CProject {
            path: cache_dir.clone(),
            source,
        })?;
    }
    let requested_launcher = toolchain.compiler_launcher.as_deref();
    let active_launcher = requested_launcher.filter(|launcher| sccache_liveness(launcher));
    let sccache_before = active_launcher.and_then(sccache_stats);
    if let Some(launcher) = requested_launcher.filter(|_| active_launcher.is_none()) {
        tracing::warn!(
            launcher,
            compiler = %toolchain.compiler,
            "rvr compiler launcher probe failed; using the compiler directly"
        );
    }
    let compiler_command = active_launcher.map_or_else(
        || toolchain.compiler.clone(),
        |_| toolchain.compiler_command(),
    );
    tracing::info!(
        translation_units = total_objects,
        make = %toolchain.make,
        jobs = %jobs,
        thinlto_jobs = thinlto.jobs,
        thinlto_cache = thinlto.cache_dir.as_ref().map(|path| path.display().to_string()),
        compiler_launcher = ?active_launcher,
        "building rvr native library"
    );

    let mut make = Command::new(&toolchain.make);
    make.arg("-C")
        .arg(output_dir)
        .arg("-j")
        .arg(&jobs)
        .arg("-s")
        .arg("shared")
        .args(make_args)
        .arg(format!("HOST_OS={}", toolchain.host_os))
        .arg(format!("THINLTO_JOBS={}", thinlto.jobs))
        .args(
            thinlto
                .cache_dir
                .as_ref()
                .map(|path| format!("THINLTO_CACHE_DIR={}", path.display())),
        )
        .env("CC", &compiler_command)
        .env("LINKER", &toolchain.linker)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));
    if active_launcher.is_some() {
        make.env("SCCACHE_BASEDIR", output_dir);
    }
    let mut child = make
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
            report_sccache_stats(active_launcher, sccache_before);
            if status.success() {
                return Ok(());
            }

            let failure = read_make_failure(&stdout_path, &stderr_path);
            if let Some(launcher) = active_launcher {
                tracing::warn!(
                    launcher,
                    compiler = %toolchain.compiler,
                    "rvr compiler launcher build failed; cleaning and retrying directly"
                );
                clean_generated_project(output_dir, make_args, toolchain, &failure)?;
                let mut direct_toolchain = toolchain.clone();
                direct_toolchain.compiler_launcher = None;
                return compile_generated_project(
                    output_dir,
                    make_args,
                    &direct_toolchain,
                    thinlto,
                );
            }

            return Err(CompileError::Make { stderr: failure });
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn clean_generated_project(
    output_dir: &Path,
    make_args: &[String],
    toolchain: &rvr_openvm::RuntimeToolchain,
    original_failure: &str,
) -> Result<(), CompileError> {
    let output = Command::new(&toolchain.make)
        .arg("-C")
        .arg(output_dir)
        .arg("-s")
        .arg("clean")
        .args(make_args)
        .arg(format!("HOST_OS={}", toolchain.host_os))
        .env("CC", &toolchain.compiler)
        .env("LINKER", &toolchain.linker)
        .output()
        .map_err(|source| CompileError::ToolchainCommand {
            command: toolchain.make.clone(),
            source,
        })?;
    if output.status.success() {
        return Ok(());
    }
    Err(CompileError::Make {
        stderr: format!(
            "{original_failure}\n\nsccache fallback cleanup failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout),
        ),
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SccacheStats {
    requests: u64,
    hits: u64,
    misses: u64,
    errors: u64,
}

impl SccacheStats {
    fn delta(self, before: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(before.requests),
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
            errors: self.errors.saturating_sub(before.errors),
        }
    }
}

fn sccache_stats(launcher: &str) -> Option<SccacheStats> {
    let output = Command::new(launcher)
        .args(["--show-stats", "--stats-format=json"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| sccache_stats_json(&output.stdout))
        .flatten()
}

fn sccache_liveness(launcher: &str) -> bool {
    Command::new(launcher)
        .arg("--show-stats")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn sccache_stats_json(json: &[u8]) -> Option<SccacheStats> {
    let value: serde_json::Value = serde_json::from_slice(json).ok()?;
    let stats = value.get("stats")?;
    Some(SccacheStats {
        requests: json_count(stats.get("compile_requests")?),
        hits: sccache_event_count(stats.get("cache_hits")?),
        misses: sccache_event_count(stats.get("cache_misses")?),
        errors: sccache_event_count(stats.get("cache_errors")?),
    })
}

fn sccache_event_count(value: &serde_json::Value) -> u64 {
    let Some(value) = value.as_object() else {
        return json_count(value);
    };
    let count = value.get("counts").map_or(0, json_count);
    if count > 0 {
        count
    } else {
        value.get("adv_counts").map_or(0, json_count)
    }
}

fn json_count(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::Number(number) => number.as_u64().unwrap_or_default(),
        serde_json::Value::Array(values) => values.iter().map(json_count).sum(),
        serde_json::Value::Object(values) => values.values().map(json_count).sum(),
        _ => 0,
    }
}

fn report_sccache_stats(launcher: Option<&str>, before: Option<SccacheStats>) {
    let Some(launcher) = launcher else {
        return;
    };
    match (before, sccache_stats(launcher)) {
        (Some(before), Some(after)) => {
            let stats = after.delta(before);
            tracing::info!(
                requests = stats.requests,
                hits = stats.hits,
                misses = stats.misses,
                errors = stats.errors,
                "rvr sccache compile statistics"
            );
        }
        _ => tracing::warn!("rvr sccache statistics unavailable"),
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

#[cfg(test)]
mod tests {
    use rvr_openvm_lift::RvrExtension;

    use super::*;
    use crate::arch::rvr::AddIArenaFieldOffsets;

    #[test]
    fn g2_emission_modes_share_wire_fingerprint_but_not_producer_schema() {
        let wire_fingerprint = [0x5a; 32];
        let checked = g2_producer_schema_fingerprint(wire_fingerprint, G2EmissionMode::Checked);
        let production =
            g2_producer_schema_fingerprint(wire_fingerprint, G2EmissionMode::Production);

        assert_ne!(checked, production);
        assert_eq!(wire_fingerprint, [0x5a; 32]);
    }

    #[cfg(unix)]
    #[test]
    fn failed_sccache_build_cleans_and_retries_with_direct_compiler() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let launcher = dir.path().join("sccache");
        fs::write(
            &launcher,
            format!(
                "#!/bin/sh\n\
                 if [ \"$1\" = \"--show-stats\" ]; then\n\
                   printf '%s' 'legacy text statistics'\n\
                   exit 0\n\
                 fi\n\
                 touch '{}'\n\
                 exit 1\n",
                dir.path().join("launcher-ran").display()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&launcher).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&launcher, permissions).unwrap();
        fs::write(
            dir.path().join("Makefile"),
            "shared:\n\t$(CC)\nclean:\n\ttouch clean-ran\n",
        )
        .unwrap();

        let toolchain = rvr_openvm::RuntimeToolchain {
            compiler_launcher: Some(launcher.display().to_string()),
            unavailable_compiler_launcher: None,
            compiler: "/usr/bin/true".into(),
            linker: "true".into(),
            make: "make".into(),
            host_os: "Linux",
        };
        compile_generated_project(
            dir.path(),
            &[],
            &toolchain,
            &ThinLtoBuildOptions {
                jobs: 1,
                cache_dir: None,
            },
        )
        .unwrap();
        assert!(dir.path().join("launcher-ran").is_file());
        assert!(dir.path().join("clean-ran").is_file());
    }

    #[test]
    fn sccache_json_stats_are_summed_and_reported_as_a_delta() {
        let before = sccache_stats_json(
            br#"{
                "stats": {
                    "compile_requests": 41,
                    "cache_hits": {"counts": {"C/C++": 7}, "adv_counts": {"c [clang]": 7}},
                    "cache_misses": {"counts": {"C/C++": 31}, "adv_counts": {}},
                    "cache_errors": {"counts": {"timeout": 2}, "adv_counts": {"remote": 2}}
                }
            }"#,
        )
        .unwrap();
        let after = sccache_stats_json(
            br#"{
                "stats": {
                    "compile_requests": 63,
                    "cache_hits": {"counts": {"C/C++": 27}, "adv_counts": {"c [clang]": 27}},
                    "cache_misses": {"counts": {}, "adv_counts": {"c [clang]": 32}},
                    "cache_errors": {"counts": {"timeout": 2}, "adv_counts": {"remote": 2}}
                }
            }"#,
        )
        .unwrap();
        assert_eq!(
            after.delta(before),
            SccacheStats {
                requests: 22,
                hits: 20,
                misses: 1,
                errors: 0,
            }
        );
    }

    #[test]
    fn thinlto_jobs_use_a_bounded_host_default_and_allow_single_backend_fallback() {
        assert_eq!(thinlto_jobs(None, 1).unwrap(), 1);
        assert_eq!(thinlto_jobs(None, 16).unwrap(), 16);
        assert_eq!(thinlto_jobs(None, 64).unwrap(), DEFAULT_THINLTO_JOBS_MAX);
        assert_eq!(thinlto_jobs(Some(OsStr::new("1")), 64).unwrap(), 1);
        assert_eq!(thinlto_jobs(Some(OsStr::new("48")), 8).unwrap(), 48);
        assert!(thinlto_jobs(Some(OsStr::new("0")), 8).is_err());
        assert!(thinlto_jobs(Some(OsStr::new("invalid")), 8).is_err());
    }

    #[test]
    fn thinlto_cache_defaults_below_native_cache_and_honors_override() {
        let native_root = Path::new("/cache/rvr");
        assert_eq!(
            thinlto_cache_root(None, Some(native_root)),
            Some(native_root.join("thinlto-cache"))
        );
        assert_eq!(
            thinlto_cache_root(Some(OsStr::new("/other/cache")), Some(native_root)),
            Some(PathBuf::from("/other/cache"))
        );
        assert_eq!(
            thinlto_cache_root(Some(OsStr::new("")), Some(native_root)),
            None
        );
        assert_eq!(thinlto_cache_root(None, None), None);
    }

    #[test]
    fn thinlto_relative_cache_root_is_resolved_before_make_changes_directory() {
        let root = thinlto_cache_root(Some(OsStr::new("relative/cache")), None).unwrap();
        assert_eq!(
            resolve_cache_root(root, Path::new("/caller/worktree")),
            PathBuf::from("/caller/worktree/relative/cache")
        );
    }

    struct UnfingerprintedExtension;

    impl RvrExtension for UnfingerprintedExtension {
        fn try_lift(&self, _insn: &RvrInstruction, _pc: u64) -> Option<LiftedInstr> {
            None
        }

        fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
            Vec::new()
        }

        fn max_main_memory_pages_per_instruction(&self) -> usize {
            0
        }
    }

    #[test]
    fn extension_without_fingerprint_disables_input_cache() {
        let mut registry = ExtensionRegistry::new();
        assert_eq!(registry.codegen_fingerprints(), Some(Vec::new()));
        registry.register(UnfingerprintedExtension);
        assert_eq!(registry.codegen_fingerprints(), None);
    }

    #[test]
    fn native_cache_manifest_reads_legacy_and_input_keyed_formats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.so.sha256");

        fs::write(&path, "project=project-key\nartifact=artifact-key\n").unwrap();
        assert_eq!(
            read_preflight_cache_manifest(&path),
            Some(RvrNativeCacheManifest {
                project_key: "project-key".into(),
                artifact_key: "artifact-key".into(),
                input_key: None,
            })
        );

        write_preflight_cache_manifest(&path, "project-key", "artifact-key", Some("input-key"))
            .unwrap();
        assert_eq!(
            read_preflight_cache_manifest(&path),
            Some(RvrNativeCacheManifest {
                project_key: "project-key".into(),
                artifact_key: "artifact-key".into(),
                input_key: Some("input-key".into()),
            })
        );
    }

    #[test]
    fn native_cache_manifest_rejects_ambiguous_or_incomplete_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.so.sha256");
        for invalid in [
            "project=p\n",
            "artifact=a\n",
            "project=p\nartifact=a\nunknown=x\n",
            "project=p\nproject=q\nartifact=a\n",
            "project=p\nartifact=a\ninput=x\ninput=y\n",
        ] {
            fs::write(&path, invalid).unwrap();
            assert_eq!(read_preflight_cache_manifest(&path), None);
        }
    }

    #[test]
    fn g2_air_manifest_binds_registry_geometry_and_air_index() {
        let geometry = ArenaNativeGeometry {
            adapter_size: 101,
            adapter_align: 8,
            core_size: 37,
            core_align: 4,
            core_off_matrix: 104,
            layout: ArenaNativeLayout::AddI(AddIArenaFieldOffsets {
                from_pc: 1,
                from_timestamp: 2,
                rd_ptr: 3,
                rs1_ptr: 4,
                read_prev_ts: 5,
                write_prev_ts: 6,
                write_prev_data: 7,
                core_rs1: 8,
                core_imm_low11: 9,
                core_imm_sign: 10,
            }),
        };
        let fingerprint = g2_air_manifest_fingerprint(&[(29, 7, geometry)], &[]).unwrap();

        let mut changed_geometry = geometry;
        changed_geometry.adapter_size += 1;
        assert_ne!(
            fingerprint,
            g2_air_manifest_fingerprint(&[(29, 7, changed_geometry)], &[]).unwrap()
        );
        assert_ne!(
            fingerprint,
            g2_air_manifest_fingerprint(&[(29, 8, geometry)], &[]).unwrap()
        );

        let opaque = RvrG2OpaqueBindingV1 {
            air_idx: 9,
            geometry,
            max_residual_events_per_record: 7,
            air_identity_digest: [0x3c; 32],
            layout_digest: [0x5a; 32],
        };
        let opaque_fingerprint = g2_air_manifest_fingerprint(&[], &[opaque]).unwrap();
        assert_ne!(
            opaque_fingerprint,
            g2_air_manifest_fingerprint(
                &[],
                &[RvrG2OpaqueBindingV1 {
                    layout_digest: [0xa5; 32],
                    ..opaque
                }],
            )
            .unwrap()
        );
        assert_ne!(
            opaque_fingerprint,
            g2_air_manifest_fingerprint(
                &[],
                &[RvrG2OpaqueBindingV1 {
                    max_residual_events_per_record: 8,
                    ..opaque
                }],
            )
            .unwrap()
        );
        assert_ne!(
            opaque_fingerprint,
            g2_air_manifest_fingerprint(
                &[],
                &[RvrG2OpaqueBindingV1 {
                    air_identity_digest: [0xc3; 32],
                    ..opaque
                }],
            )
            .unwrap()
        );
        assert_ne!(
            opaque_fingerprint,
            g2_air_manifest_fingerprint(
                &[],
                &[RvrG2OpaqueBindingV1 {
                    air_idx: 10,
                    ..opaque
                }],
            )
            .unwrap()
        );
    }
}
