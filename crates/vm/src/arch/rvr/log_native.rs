//! Extension-extensible record assembly from rvr preflight logs.

use std::{cell::Cell, collections::HashMap};

use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    PreflightMemoryAccessAux, PreflightRawLogs, ProgramLogEntry, RvrDeltaRecords,
    RvrInlineChipRecords, RvrPreflightOutput, PREFLIGHT_DELTA_RECORD_SIZE,
    PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_TOUCH, PREFLIGHT_MEMORY_KIND_WRITE,
};
use crate::arch::{Arena, ExecutionError};

/// Timestamp-indexed view of normalized rvr preflight memory accesses.
///
/// The access-aux slice is already sorted by (strictly increasing) timestamp
/// (log emission order), and assemblers request timestamps monotonically
/// (program-log order, and `ts`, `ts+1`, … within an instruction). So instead
/// of a per-access `HashMap`, this holds a forward cursor into the slice: each
/// `expect` advances it, giving amortized O(1), cache-friendly lookups with no
/// build cost. A request behind the cursor (never expected in practice) restarts
/// the forward scan, keeping correctness independent of call order.
///
/// Assemblers pass their expected access width to [`Self::expect`], so the
/// same view can validate records for extensions with different memory widths.
pub struct LogNativeAccessView<'a, F> {
    access_aux: &'a [PreflightMemoryAccessAux<F>],
    cursor: Cell<usize>,
}

impl<'a, F: PrimeField32> LogNativeAccessView<'a, F> {
    pub fn new(access_aux: &'a [PreflightMemoryAccessAux<F>]) -> Result<Self, ExecutionError> {
        Ok(Self {
            access_aux,
            cursor: Cell::new(0),
        })
    }

    /// Find the access with the given timestamp by advancing the forward cursor.
    fn find(&self, timestamp: u32) -> Option<&'a PreflightMemoryAccessAux<F>> {
        let n = self.access_aux.len();
        let mut i = self.cursor.get();
        // Requested timestamp is behind the cursor: restart the scan. Does not
        // happen for in-order assembly, but keeps `find` order-independent.
        if i >= n || self.access_aux[i].entry.timestamp > timestamp {
            i = 0;
        }
        while i < n && self.access_aux[i].entry.timestamp < timestamp {
            i += 1;
        }
        if i < n && self.access_aux[i].entry.timestamp == timestamp {
            self.cursor.set(i);
            Some(&self.access_aux[i])
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn expect(
        &self,
        timestamp: u32,
        kind: u8,
        addr_space: u32,
        address: u64,
        width: usize,
        pc: u32,
    ) -> Result<&'a PreflightMemoryAccessAux<F>, ExecutionError> {
        let aux = self.find(timestamp).ok_or_else(|| {
            rvr_error(format!(
                "missing memory-log entry for pc {pc:#x} timestamp {timestamp}"
            ))
        })?;
        let entry = aux.entry;
        if entry.kind != kind
            || entry.addr_space as u32 != addr_space
            || entry.address != address
            || entry.width as usize != width
        {
            return Err(rvr_error(format!(
                "unexpected memory-log entry for pc {pc:#x} timestamp {timestamp}: \
                 got kind={} as={} addr={} width={}, expected kind={} as={} addr={} width={}",
                entry.kind,
                entry.addr_space,
                entry.address,
                entry.width,
                kind,
                addr_space,
                address,
                width
            )));
        }
        Ok(aux)
    }
}

/// Monomorphized assembler for one or more opcodes.
pub type LogNativeAssembler<F, RA> = for<'a> fn(
    &mut RA,
    &LogNativeAccessView<'a, F>,
    &Instruction<F>,
    u32,
    u32,
) -> Result<(), ExecutionError>;

type LogNativeAdmitPredicate<F> = fn(&Instruction<F>) -> bool;

struct RegisteredAssembler<F, RA> {
    admit: LogNativeAdmitPredicate<F>,
    assemble: LogNativeAssembler<F, RA>,
}

/// Assembler for one instruction's inline compact record (R3): expands the
/// C-written `compact` bytes into the chip's full record allocated from the
/// arena, re-deriving program-redundant operands from `instruction`.
pub type LogNativeInlineAssembler<F, RA> =
    fn(&mut RA, &Instruction<F>, &[u8], u32) -> Result<(), ExecutionError>;

/// R4 arena-native geometry types live in `rvr_openvm` (codegen consumes
/// them to bake literal store offsets); extensions supply the values from
/// the real record types at registration. Re-exported here so extension
/// crates keep a single import path.
pub use rvr_openvm::{
    Alu3ArenaFieldOffsets, Alu3WArenaFieldOffsets, ArenaNativeGeometry, ArenaNativeLayout,
    Branch2ArenaFieldOffsets, LoadStoreArenaFieldOffsets, Rw1ArenaFieldOffsets,
    Wr1ArenaFieldOffsets,
};

struct RegisteredInlineAssembler<F, RA> {
    /// Compact record stride in bytes; must match the compile metadata.
    record_size: usize,
    assemble: LogNativeInlineAssembler<F, RA>,
    /// Present iff this family has an R4 arena-native emitter: the generated
    /// C then writes the full record at final arena positions and the host
    /// skips `assemble` for it entirely. Families without geometry keep the
    /// R3 compact wire + host expansion, so R4 rolls out shape by shape.
    arena_native: Option<ArenaNativeGeometry>,
    delta_pattern: Option<DeltaAccessPattern>,
    /// Width used to reconstruct a Store-pattern block update from v2.
    delta_store_width: Option<u8>,
}

/// How one chronological Stage-2 delta record touches timestamp-shadow
/// blocks. Values are still carried by the record; only the three previous
/// timestamps are reconstructed from these program-derived addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaAccessPattern {
    Alu3,
    Alu3Reg,
    Load,
    Store,
    Branch2,
    Wr1,
    Wr1Always,
    Rw1,
}

/// One compiled-program operand-table entry for the CUDA delta decoder.
///
/// The owning extension derives these fields once during preflight AOT
/// compilation. Keeping the ABI-shaped value in `openvm-circuit` lets the
/// compiled artifact retain it without making the VM crate depend on a
/// particular extension's opcode enum.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct RvrDeltaDecodeEntry {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub flags: u8,
    pub local_opcode: u8,
    pub air_idx: u8,
    pub access_pattern: u8,
}

const _: () = assert!(core::mem::size_of::<RvrDeltaDecodeEntry>() == 16);
const _: () = assert!(core::mem::align_of::<RvrDeltaDecodeEntry>() == 4);

/// Extension-independent result of classifying one program instruction for
/// delta decode. `kind` is the owning extension's stable `repr(u8)` decoder
/// discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvrDeltaDecodeInfo {
    pub entry: RvrDeltaDecodeEntry,
    pub kind: u8,
}

type RvrDeltaDecodeFn<F> = fn(&Instruction<F>) -> Option<RvrDeltaDecodeInfo>;
type RvrDeltaWriteValueFn<F> = fn(&Instruction<F>, u32, u64, u64) -> Option<u64>;

/// Opcode-keyed registry of log-native record assemblers.
pub struct LogNativeAssemblerRegistry<F, RA> {
    assemblers: HashMap<VmOpcode, RegisteredAssembler<F, RA>>,
    inline_assemblers: HashMap<VmOpcode, RegisteredInlineAssembler<F, RA>>,
    delta_decode: Option<RvrDeltaDecodeFn<F>>,
    delta_write_value: Option<RvrDeltaWriteValueFn<F>>,
}

impl<F, RA> LogNativeAssemblerRegistry<F, RA> {
    pub fn new() -> Self {
        Self {
            assemblers: HashMap::new(),
            inline_assemblers: HashMap::new(),
            delta_decode: None,
            delta_write_value: None,
        }
    }

    /// Register the program-only delta operand/classification derivation.
    /// This is deliberately one callback for the composed RV64 decoder: AIR
    /// taint must be computed across every opcode routed to the AIR, not from
    /// a collection of independently partial family registries.
    pub fn register_delta_decode(&mut self, decode: RvrDeltaDecodeFn<F>) {
        assert!(
            self.delta_decode.replace(decode).is_none(),
            "multiple RVR delta decode classifiers registered"
        );
    }

    /// Register the extension-owned post-write derivation used by the
    /// 24-byte delta schema. It is separate from access reconstruction so the
    /// extension that owns the opcodes also owns their execution semantics.
    pub fn register_delta_write_value(&mut self, derive: RvrDeltaWriteValueFn<F>) {
        assert!(
            self.delta_write_value.replace(derive).is_none(),
            "multiple RVR delta write-value derivations registered"
        );
    }

    /// Register the compact-record assembler for opcodes the preflight codegen
    /// can migrate to inline records (R3). The log assembler for these opcodes
    /// stays registered — it still runs when a library is compiled without
    /// inline records; which path a pc takes is decided by the compile
    /// metadata (`RvrPreflightOutput::inline_pc_slots`).
    pub fn register_inline(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        record_size: usize,
        assembler: LogNativeInlineAssembler<F, RA>,
    ) {
        self.register_inline_impl(opcodes, record_size, assembler, None)
    }

    /// Like [`Self::register_inline`], additionally declaring the family's
    /// full-record arena geometry so the R4 arena-native emitter can place
    /// records at final arena positions (the compact assembler stays
    /// registered for compact-wire compiles).
    pub fn register_inline_arena_native(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        record_size: usize,
        assembler: LogNativeInlineAssembler<F, RA>,
        geometry: ArenaNativeGeometry,
    ) {
        self.register_inline_impl(opcodes, record_size, assembler, Some(geometry))
    }

    fn register_inline_impl(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        record_size: usize,
        assembler: LogNativeInlineAssembler<F, RA>,
        arena_native: Option<ArenaNativeGeometry>,
    ) {
        for opcode in opcodes {
            assert!(
                self.inline_assemblers
                    .insert(
                        opcode,
                        RegisteredInlineAssembler {
                            record_size,
                            assemble: assembler,
                            arena_native,
                            delta_pattern: None,
                            delta_store_width: None,
                        }
                    )
                    .is_none(),
                "multiple inline record assemblers registered for opcode {opcode:?}"
            );
        }
    }

    /// Attach the program-derived access pattern used by the Stage-2 delta
    /// decoder. Kept separate from `register_inline*` so existing extension
    /// registrations remain source-compatible and a missing pattern fails
    /// closed only when delta mode is requested.
    pub fn register_delta_pattern(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        pattern: DeltaAccessPattern,
    ) {
        for opcode in opcodes {
            let registered = self
                .inline_assemblers
                .get_mut(&opcode)
                .unwrap_or_else(|| panic!("delta pattern without inline assembler for {opcode:?}"));
            assert!(
                registered.delta_pattern.replace(pattern).is_none(),
                "multiple delta patterns registered for opcode {opcode:?}"
            );
        }
    }

    /// Attach the byte width for a Stage-2 Store-pattern opcode. The delta
    /// record carries the scalar source while CPU/CUDA replay patches the
    /// seeded pre-write block.
    pub fn register_delta_store_width(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        width: u8,
    ) {
        assert!(matches!(width, 1 | 2 | 4 | 8));
        for opcode in opcodes {
            let registered = self.inline_assemblers.get_mut(&opcode).unwrap_or_else(|| {
                panic!("delta store width without inline assembler for {opcode:?}")
            });
            assert_eq!(registered.delta_pattern, Some(DeltaAccessPattern::Store));
            assert!(
                registered.delta_store_width.replace(width).is_none(),
                "multiple delta store widths registered for opcode {opcode:?}"
            );
        }
    }

    /// R4: the arena-native geometry registered for `opcode`, if its family
    /// has a fused emitter.
    pub fn inline_arena_geometry(&self, opcode: &VmOpcode) -> Option<ArenaNativeGeometry> {
        self.inline_assemblers
            .get(opcode)
            .and_then(|reg| reg.arena_native)
    }

    /// Register `assembler` for every opcode in `opcodes`.
    ///
    /// Duplicate ownership is a configuration error: exactly one extension
    /// must own record assembly for an opcode.
    pub fn register(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        assembler: LogNativeAssembler<F, RA>,
    ) {
        self.register_if(opcodes, |_| true, assembler);
    }

    /// Register an assembler whose opcode has instruction-level variants.
    ///
    /// The predicate is consulted by both routing and assembly. This supports
    /// cases such as a shared opcode whose address-space operand selects an
    /// extension-owned operation.
    pub fn register_if(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        admit: LogNativeAdmitPredicate<F>,
        assembler: LogNativeAssembler<F, RA>,
    ) {
        for opcode in opcodes {
            assert!(
                self.assemblers
                    .insert(
                        opcode,
                        RegisteredAssembler {
                            admit,
                            assemble: assembler
                        }
                    )
                    .is_none(),
                "multiple log-native assemblers registered for opcode {opcode:?}"
            );
        }
    }

    pub fn contains_instruction(&self, instruction: &Instruction<F>) -> bool {
        self.assemblers
            .get(&instruction.opcode)
            .is_some_and(|registered| (registered.admit)(instruction))
    }

    pub fn is_empty(&self) -> bool {
        self.assemblers.is_empty()
    }

    fn assemble(
        &self,
        arena: &mut RA,
        access: &LogNativeAccessView<'_, F>,
        instruction: &Instruction<F>,
        pc: u32,
        timestamp: u32,
    ) -> Result<(), ExecutionError> {
        let registered = self.assemblers.get(&instruction.opcode).ok_or_else(|| {
            rvr_error(format!(
                "unsupported opcode {:?} at pc {pc:#x} in rvr log-native record assembly",
                instruction.opcode
            ))
        })?;
        if !(registered.admit)(instruction) {
            return Err(rvr_error(format!(
                "unsupported operands for opcode {:?} at pc {pc:#x} in rvr log-native record assembly",
                instruction.opcode
            )));
        }
        (registered.assemble)(arena, access, instruction, pc, timestamp)
    }
}

impl<F, RA> Default for LogNativeAssemblerRegistry<F, RA> {
    fn default() -> Self {
        Self::new()
    }
}

/// Self-registration seam for an extension's log-native assemblers.
pub trait VmRvrLogNativeExtension<F: PrimeField32, RA: Arena> {
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>);
}

impl<F, RA, EXT> VmRvrLogNativeExtension<F, RA> for Option<EXT>
where
    F: PrimeField32,
    RA: Arena,
    EXT: VmRvrLogNativeExtension<F, RA>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        if let Some(extension) = self {
            extension.extend_rvr_log_native(registry);
        }
    }
}

/// Predicate consumed by the preflight classifier.
///
/// The registry itself implements this trait, which keeps routing and record
/// assembly on the same source of truth.
pub trait LogNativeOpcodeAdmitter<F> {
    fn has_log_native_assembler(&self, instruction: &Instruction<F>) -> bool;

    /// R4: the arena-native geometry for `instruction`'s family, if it has a
    /// fused emitter. Compile metadata uses this to decide per air whether
    /// the generated C emits full records at arena positions.
    fn inline_arena_geometry_for(
        &self,
        _instruction: &Instruction<F>,
    ) -> Option<ArenaNativeGeometry> {
        None
    }

    /// AOT-only operand/classification derivation for the CUDA delta decoder.
    fn delta_decode_for(&self, _instruction: &Instruction<F>) -> Option<RvrDeltaDecodeInfo> {
        None
    }

    /// Distinguishes an available classifier whose valid result is empty
    /// from a registry that cannot precompute delta decode at all.
    fn has_delta_decode(&self) -> bool {
        false
    }
}

impl<F, RA> LogNativeOpcodeAdmitter<F> for LogNativeAssemblerRegistry<F, RA> {
    fn has_log_native_assembler(&self, instruction: &Instruction<F>) -> bool {
        self.contains_instruction(instruction)
    }

    fn inline_arena_geometry_for(
        &self,
        instruction: &Instruction<F>,
    ) -> Option<ArenaNativeGeometry> {
        self.inline_arena_geometry(&instruction.opcode)
    }

    fn delta_decode_for(&self, instruction: &Instruction<F>) -> Option<RvrDeltaDecodeInfo> {
        self.delta_decode.and_then(|decode| decode(instruction))
    }

    fn has_delta_decode(&self) -> bool {
        self.delta_decode.is_some()
    }
}

impl<F> LogNativeOpcodeAdmitter<F> for () {
    fn has_log_native_assembler(&self, _instruction: &Instruction<F>) -> bool {
        false
    }
}

/// Assemble every non-system record arena for a preflight segment.
///
/// Instructions migrated to inline compact records (R3) have no memory-log
/// entries and no log assembler run: for each such pc (per
/// `output.inline_pc_slots`, the compile metadata mirroring the generated C),
/// the next compact record is consumed from that chip's C-written buffer —
/// records are emitted in program-log order per chip — and expanded into the
/// arena by the opcode's registered inline assembler, which re-derives
/// program-redundant operands from the instruction. Every chip's buffer must
/// be consumed exactly (cursor == written bytes), so a record dropped in C
/// (buffer overflow, unmapped chip) fails loudly here rather than surfacing
/// as a bus imbalance at proving.
pub fn generate_record_arenas_from_logs<F: PrimeField32, RA: Arena>(
    registry: &LogNativeAssemblerRegistry<F, RA>,
    exe: &VmExe<F>,
    output: &mut RvrPreflightOutput<F>,
    capacities: &[(usize, usize)],
    pc_to_air_idx: &[Option<usize>],
) -> Result<Vec<RA>, ExecutionError> {
    let (arenas, _) = generate_record_arenas_from_logs_with_compact(
        registry,
        exe,
        output,
        capacities,
        pc_to_air_idx,
        &std::collections::HashSet::new(),
    )?;
    Ok(arenas)
}

/// [`generate_record_arenas_from_logs`] with a compact bypass: for AIRs in
/// `compact_airs`, host expansion is skipped — the walk keeps the per-record
/// `from_pc` order guard and the exact-consumption check, but the records stay
/// in wire form and are returned as the second tuple element (those AIRs'
/// arenas are left empty; the caller adopts the wire buffers into its concrete
/// arena type, e.g. for GPU on-device decode or zero-copy full-record feeds).
pub fn generate_record_arenas_from_logs_with_compact<F: PrimeField32, RA: Arena>(
    registry: &LogNativeAssemblerRegistry<F, RA>,
    exe: &VmExe<F>,
    output: &mut RvrPreflightOutput<F>,
    capacities: &[(usize, usize)],
    pc_to_air_idx: &[Option<usize>],
    compact_airs: &std::collections::HashSet<usize>,
) -> Result<(Vec<RA>, Vec<RvrInlineChipRecords>), ExecutionError> {
    let detailed_profile =
        std::env::var("OPENVM_RVR_PREFLIGHT_PROFILE_DETAIL").as_deref() == Ok("1");
    let detail_started = std::time::Instant::now();
    let delta_recycle = expand_delta_records(registry, exe, output, pc_to_air_idx)?;
    let delta_expand_finished = std::time::Instant::now();
    let inline_records = std::mem::take(&mut output.inline_records);
    let inline_pc_slots = output.inline_pc_slots.clone();
    // Airs whose records C already wrote into direct-final caller backings.
    // ZG2 suppresses their duplicate program-log entries entirely; the C
    // cursor is the exact record-count oracle and the caller substitutes the
    // finished arena into the empty placeholder below.
    let arena_native_expected: HashMap<usize, u32> =
        output.arena_native_written.iter().copied().collect();
    // Compact-aware capacity map: airs whose arenas the caller substitutes —
    // C-staged (arena-native or wire targets) or compact-adopted — get a
    // zero-capacity placeholder. A full-size arena here would be allocated,
    // zeroed, never written, and dropped at substitution (the measured G2
    // "buffer #1" cost). A record wrongly routed at a placeholder still fails
    // loudly on its capacity assert instead of writing.
    let mut arenas = capacities
        .iter()
        .enumerate()
        .map(|(air_idx, &(height, width))| {
            if arena_native_expected.contains_key(&air_idx) || compact_airs.contains(&air_idx) {
                RA::with_capacity(0, width)
            } else {
                RA::with_capacity(height, width)
            }
        })
        .collect::<Vec<_>>();
    let arena_alloc_finished = std::time::Instant::now();
    let access = LogNativeAccessView::new(&output.access_aux)?;
    let access_view_finished = std::time::Instant::now();

    // Per-air (compact bytes, cursor, compiled record stride).
    let mut inline_bufs: HashMap<usize, (&[u8], usize, usize)> = inline_records
        .iter()
        .map(|chip| {
            (
                chip.air_idx,
                (chip.bytes.as_slice(), 0usize, chip.record_size),
            )
        })
        .collect();
    let inline_map_finished = std::time::Instant::now();
    let mut arena_native_skipped = 0usize;
    let mut compact_seen = 0usize;
    let mut host_assembled = 0usize;

    for program_entry in &output.raw_logs.program_log {
        let pc = program_entry.pc();
        let Some((instruction, air_idx, slot_idx)) =
            instruction_and_air_idx(exe, pc_to_air_idx, pc)?
        else {
            continue;
        };
        // In combined delta + arena-native mode these entries exist only for
        // device-side memory chronology. C already wrote the final record
        // into its arena.
        if arena_native_expected.contains_key(&air_idx) {
            arena_native_skipped += 1;
            continue;
        }
        if !output.access_aux_complete {
            return Err(rvr_error(format!(
                "host log-native record reached finalization at pc {pc:#x} after compiler-scope \
                 direct coverage omitted its access replay"
            )));
        }
        let num_arenas = arenas.len();
        let arena = arenas.get_mut(air_idx).ok_or_else(|| {
            rvr_error(format!(
                "pc {:#x} maps to air_idx {} but only {} arenas exist",
                program_entry.pc(),
                air_idx,
                num_arenas
            ))
        })?;
        if inline_pc_slots.get(slot_idx).copied().unwrap_or(false) {
            let registered = registry
                .inline_assemblers
                .get(&instruction.opcode)
                .ok_or_else(|| {
                    rvr_error(format!(
                        "no inline record assembler registered for opcode {:?} at pc {pc:#x}",
                        instruction.opcode
                    ))
                })?;
            let (bytes, cursor, compiled_size) =
                inline_bufs.get_mut(&air_idx).ok_or_else(|| {
                    rvr_error(format!(
                        "pc {pc:#x} is inline but air_idx {air_idx} has no compact record buffer"
                    ))
                })?;
            if registered.record_size != *compiled_size {
                return Err(rvr_error(format!(
                    "compact record size mismatch for air_idx {air_idx}: compiled \
                     {compiled_size} vs registered {}",
                    registered.record_size
                )));
            }
            let end = *cursor + registered.record_size;
            let compact = bytes.get(*cursor..end).ok_or_else(|| {
                rvr_error(format!(
                    "compact record buffer for air_idx {air_idx} exhausted at pc {pc:#x} \
                     (cursor {cursor}, {} bytes written — record dropped in C?)",
                    bytes.len()
                ))
            })?;
            *cursor = end;
            if compact_airs.contains(&air_idx) {
                // Compact bypass: keep the order guard, skip host expansion. Every
                // wire format leads with `from_pc: u32` (little-endian).
                let from_pc = u32::from_le_bytes(compact[0..4].try_into().expect("4-byte from_pc"));
                if from_pc != pc {
                    return Err(rvr_error(format!(
                        "inline record order mismatch (compact bypass): record from_pc \
                         {from_pc:#x} vs program-log pc {pc:#x}"
                    )));
                }
                compact_seen += 1;
                continue;
            }
            (registered.assemble)(arena, instruction, compact, pc)?;
            host_assembled += 1;
            continue;
        }
        registry.assemble(arena, &access, instruction, pc, program_entry.timestamp)?;
        host_assembled += 1;
    }
    let program_walk_finished = std::time::Instant::now();

    // Every compact buffer must be consumed exactly.
    for chip in &inline_records {
        let (bytes, cursor, _) = inline_bufs[&chip.air_idx];
        if cursor != bytes.len() {
            return Err(rvr_error(format!(
                "compact record buffer for air_idx {} has {} bytes but the program log consumed \
                 {cursor} (record count mismatch between C and the program log)",
                chip.air_idx,
                bytes.len()
            )));
        }
    }
    drop(inline_bufs);
    let validation_finished = std::time::Instant::now();

    // Compact-air wire buffers are handed to the caller for adoption (GPU
    // on-device decode); every other chip's byte buffer goes back through the
    // output so the caller can return it to the preflight buffer pool for the
    // next segment. A compact air that was C-staged (its records already sit
    // in a caller-provided wire target) has only an empty placeholder here —
    // that goes back to the pool, not to adoption.
    let (wire_buffers, pooled): (Vec<_>, Vec<_>) = inline_records.into_iter().partition(|chip| {
        compact_airs.contains(&chip.air_idx) && !arena_native_expected.contains_key(&chip.air_idx)
    });
    output.inline_records = pooled;
    output.delta_records = delta_recycle;
    let partition_finished = std::time::Instant::now();

    if detailed_profile {
        eprintln!(
            "OPENVM_RVR_LOG_FINALIZE_DETAIL delta_expand_us={} arena_alloc_us={} \
             access_view_us={} inline_map_us={} program_walk_us={} validation_us={} \
             partition_us={} program_records={} arena_native_skipped={} compact_seen={} \
             host_assembled={} arenas={} arena_native_airs={} compact_airs={} \
             access_aux_complete={}",
            (delta_expand_finished - detail_started).as_micros(),
            (arena_alloc_finished - delta_expand_finished).as_micros(),
            (access_view_finished - arena_alloc_finished).as_micros(),
            (inline_map_finished - access_view_finished).as_micros(),
            (program_walk_finished - inline_map_finished).as_micros(),
            (validation_finished - program_walk_finished).as_micros(),
            (partition_finished - validation_finished).as_micros(),
            output.raw_logs.program_log.len(),
            arena_native_skipped,
            compact_seen,
            host_assembled,
            arenas.len(),
            arena_native_expected.len(),
            compact_airs.len(),
            output.access_aux_complete as u8,
        );
    }

    Ok((arenas, wire_buffers))
}

#[derive(Clone, Copy)]
struct DeltaRecord {
    from_pc: u32,
    from_timestamp: u32,
    v1: u64,
    v2: u64,
}

fn delta_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("delta u32"))
}

fn delta_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("delta u64"))
}

fn append_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn append_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn delta_access_count(pattern: DeltaAccessPattern) -> usize {
    match pattern {
        DeltaAccessPattern::Alu3
        | DeltaAccessPattern::Alu3Reg
        | DeltaAccessPattern::Load
        | DeltaAccessPattern::Store => 3,
        DeltaAccessPattern::Branch2 | DeltaAccessPattern::Rw1 => 2,
        DeltaAccessPattern::Wr1 | DeltaAccessPattern::Wr1Always => 1,
    }
}

fn delta_accesses<F: PrimeField32>(
    instruction: &Instruction<F>,
    pattern: DeltaAccessPattern,
    v1: u64,
) -> [Option<(u8, u64)>; 3] {
    let reg = |ptr: u32| {
        Some((
            openvm_instructions::riscv::RV64_REGISTER_AS as u8,
            u64::from(ptr),
        ))
    };
    let tick = None;
    let memory_address = || {
        let imm = instruction.c.as_canonical_u32() as u16;
        let offset = if instruction.g.is_one() {
            i64::from(imm as i16)
        } else {
            i64::from(imm)
        };
        (v1 as u32).wrapping_add(offset as u32) as u64 & !7u64
    };
    match pattern {
        DeltaAccessPattern::Alu3 => [
            reg(instruction.b.as_canonical_u32()),
            if instruction.e.as_canonical_u32() == openvm_instructions::riscv::RV64_REGISTER_AS {
                reg(instruction.c.as_canonical_u32())
            } else {
                tick
            },
            reg(instruction.a.as_canonical_u32()),
        ],
        DeltaAccessPattern::Alu3Reg => [
            reg(instruction.b.as_canonical_u32()),
            reg(instruction.c.as_canonical_u32()),
            reg(instruction.a.as_canonical_u32()),
        ],
        DeltaAccessPattern::Load => [
            reg(instruction.b.as_canonical_u32()),
            Some((instruction.e.as_canonical_u32() as u8, memory_address())),
            if instruction.f.is_one() {
                reg(instruction.a.as_canonical_u32())
            } else {
                tick
            },
        ],
        DeltaAccessPattern::Store => [
            reg(instruction.b.as_canonical_u32()),
            reg(instruction.a.as_canonical_u32()),
            Some((instruction.e.as_canonical_u32() as u8, memory_address())),
        ],
        DeltaAccessPattern::Branch2 => [
            reg(instruction.a.as_canonical_u32()),
            reg(instruction.b.as_canonical_u32()),
            tick,
        ],
        DeltaAccessPattern::Wr1 => [
            if instruction.f.is_one() {
                reg(instruction.a.as_canonical_u32())
            } else {
                tick
            },
            tick,
            tick,
        ],
        DeltaAccessPattern::Wr1Always => [reg(instruction.a.as_canonical_u32()), tick, tick],
        DeltaAccessPattern::Rw1 => [
            reg(instruction.b.as_canonical_u32()),
            if instruction.f.is_one() {
                reg(instruction.a.as_canonical_u32())
            } else {
                tick
            },
            tick,
        ],
    }
}

fn replay_delta_touches(
    last_touch: &mut HashMap<(u8, u64), u32>,
    accesses: [Option<(u8, u64)>; 3],
    access_count: usize,
    timestamp: u32,
) -> [u32; 3] {
    let mut prev = [0u32; 3];
    for (offset, access) in accesses.into_iter().take(access_count).enumerate() {
        if let Some((addr_space, address)) = access {
            let key = (addr_space, address & !7u64);
            prev[offset] = last_touch.get(&key).copied().unwrap_or(0);
            last_touch.insert(key, timestamp + offset as u32);
        }
    }
    prev
}

fn delta_write_slot<F: PrimeField32>(
    instruction: &Instruction<F>,
    pattern: DeltaAccessPattern,
) -> Option<usize> {
    match pattern {
        DeltaAccessPattern::Alu3 | DeltaAccessPattern::Alu3Reg | DeltaAccessPattern::Store => {
            Some(2)
        }
        DeltaAccessPattern::Load => instruction.f.is_one().then_some(2),
        DeltaAccessPattern::Branch2 => None,
        DeltaAccessPattern::Wr1 => instruction.f.is_one().then_some(0),
        DeltaAccessPattern::Wr1Always => Some(0),
        DeltaAccessPattern::Rw1 => instruction.f.is_one().then_some(1),
    }
}

fn patch_delta_store<F: PrimeField32>(
    instruction: &Instruction<F>,
    width: u8,
    previous: u64,
    base: u64,
    value: u64,
) -> u64 {
    let imm = instruction.c.as_canonical_u32() as u16;
    let offset = if instruction.g.is_one() {
        i64::from(imm as i16)
    } else {
        i64::from(imm)
    };
    let address = (base as u32).wrapping_add(offset as u32);
    let shift = (address & 7) * 8;
    let mask = if width == 8 {
        u64::MAX
    } else {
        (1u64 << (u32::from(width) * 8)) - 1
    };
    (previous & !(mask << shift)) | ((value & mask) << shift)
}

#[derive(Clone, Copy, Debug)]
struct ResidualMemoryEvent {
    timestamp: u32,
    kind: u8,
    addr_space: u8,
    address: u64,
    value: u64,
}

fn residual_memory_event(
    logs: &PreflightRawLogs,
    index: usize,
) -> Result<Option<ResidualMemoryEvent>, ExecutionError> {
    if !logs.memory_log.is_empty() && !logs.delta_memory_log.is_empty() {
        return Err(rvr_error(
            "delta residual memory logs populated both full and compact schemas".to_string(),
        ));
    }
    if let Some(entry) = logs.memory_log.get(index) {
        return Ok(Some(ResidualMemoryEvent {
            timestamp: entry.timestamp,
            kind: entry.kind,
            addr_space: entry.addr_space,
            address: entry.address,
            value: entry.value,
        }));
    }
    let Some(entry) = logs.delta_memory_log.get(index) else {
        return Ok(None);
    };
    let valid_width = matches!(entry.width, 1 | 2 | 4 | 8)
        || (entry.kind == PREFLIGHT_MEMORY_KIND_TOUCH && entry.width == 0);
    if entry.complete != 1
        || entry._reserved != 0
        || !matches!(
            entry.kind,
            PREFLIGHT_MEMORY_KIND_READ | PREFLIGHT_MEMORY_KIND_WRITE | PREFLIGHT_MEMORY_KIND_TOUCH
        )
        || !valid_width
    {
        return Err(rvr_error(format!(
            "invalid compact delta residual memory entry {index}/{}: complete={} reserved={} \
             kind={} width={}",
            logs.delta_memory_log.len(),
            entry.complete,
            entry._reserved,
            entry.kind,
            entry.width,
        )));
    }
    Ok(Some(ResidualMemoryEvent {
        timestamp: entry.timestamp,
        kind: entry.kind,
        addr_space: entry.addr_space,
        address: u64::from(entry.address),
        value: entry.value,
    }))
}

/// CPU oracle for the Stage-2 device predecoder. It converts the global
/// chronological 24-byte stream back into the established per-AIR compact
/// wires. The production GPU path will perform the same merge/partition on
/// device; keeping this oracle here lets every existing arena/proof gate
/// compare the reconstructed bytes before a sandbox is considered.
fn expand_delta_records<F: PrimeField32, RA: Arena>(
    registry: &LogNativeAssemblerRegistry<F, RA>,
    exe: &VmExe<F>,
    output: &mut RvrPreflightOutput<F>,
    pc_to_air_idx: &[Option<usize>],
) -> Result<Option<RvrDeltaRecords>, ExecutionError> {
    let Some(delta) = output.delta_records.take() else {
        return Ok(None);
    };
    let delta_bytes = delta.bytes();
    if !delta_bytes
        .len()
        .is_multiple_of(PREFLIGHT_DELTA_RECORD_SIZE)
    {
        return Err(rvr_error(format!(
            "invalid delta stream: stride {}, bytes {}",
            PREFLIGHT_DELTA_RECORD_SIZE,
            delta_bytes.len()
        )));
    }
    // Validate the schema and every completeness guard up front. A trailing
    // residual event may not be needed to reconstruct a later delta record,
    // but it still must not let a partial compact write escape detection.
    if !output.raw_logs.memory_log.is_empty() && !output.raw_logs.delta_memory_log.is_empty() {
        return Err(rvr_error(
            "delta residual memory logs populated both full and compact schemas".to_string(),
        ));
    }
    for index in 0..output.raw_logs.delta_memory_log.len() {
        let _ = residual_memory_event(&output.raw_logs, index)?;
    }

    if let Some((index, entry)) =
        output
            .raw_logs
            .program_log
            .iter()
            .enumerate()
            .find(|(_, entry)| {
                entry.pc() < exe.program.pc_base
                    || !(entry.pc() - exe.program.pc_base).is_multiple_of(DEFAULT_PC_STEP)
            })
    {
        return Err(rvr_error(format!(
            "delta residual program log has invalid entry {index}/{}: pc={:#x}, timestamp={}",
            output.raw_logs.program_log.len(),
            entry.pc(),
            entry.timestamp
        )));
    }

    // The residual memory log is already in timestamp order. Merge its
    // accesses into the same last-touch map before each chronological delta
    // record; inline accesses themselves never appear in that log.
    let mut residual = 0usize;
    let mut last_touch: HashMap<(u8, u64), u32> = HashMap::new();
    let mut current_value = HashMap::with_capacity(output.raw_logs.touched.len());
    for seed in &output.raw_logs.touched {
        let key = (seed.addr_space as u8, u64::from(seed.block_addr));
        if current_value.insert(key, seed.initial_value).is_some() {
            return Err(rvr_error(format!(
                "duplicate delta first-touch seed for address space {} block {:#x}",
                seed.addr_space, seed.block_addr
            )));
        }
    }
    let mut per_air: std::collections::BTreeMap<usize, (usize, Vec<u8>)> =
        std::collections::BTreeMap::new();
    let mut synthetic_program = Vec::with_capacity(delta_bytes.len() / PREFLIGHT_DELTA_RECORD_SIZE);

    // Arena-native W records are already final, but in combined mode their
    // lightweight program-log entries carry the chronology needed to update
    // the same timestamp-shadow state as delta records. Only W geometries are
    // permitted alongside delta; register forms use ALU3Reg while W-family
    // opcodes shared with immediate forms use the dynamic ALU3 pattern.
    let arena_native_airs = output
        .arena_native_written
        .iter()
        .map(|&(air_idx, _)| air_idx)
        .collect::<std::collections::HashSet<_>>();
    let mut arena_events = Vec::new();
    for entry in &output.raw_logs.program_log {
        let pc = entry.pc();
        let Some((instruction, air_idx, _)) = instruction_and_air_idx(exe, pc_to_air_idx, pc)?
        else {
            continue;
        };
        if !arena_native_airs.contains(&air_idx) {
            continue;
        }
        let Some(pattern) = registry
            .inline_assemblers
            .get(&instruction.opcode)
            .and_then(|registered| registered.delta_pattern)
        else {
            // Custom direct-final records retain their own residual memory
            // events when needed; only base W families use the synthetic
            // program-log chronology decoded here.
            continue;
        };
        if !matches!(
            pattern,
            DeltaAccessPattern::Alu3 | DeltaAccessPattern::Alu3Reg
        ) {
            return Err(rvr_error(format!(
                "arena-native chronology pc {pc:#x} opcode {:?} has unsupported {pattern:?} pattern",
                instruction.opcode
            )));
        }
        if !entry.write_complete() {
            return Err(rvr_error(format!(
                "arena-native chronology pc {pc:#x} opcode {:?} omitted its post-write value",
                instruction.opcode
            )));
        }
        arena_events.push((entry.timestamp, pc, pattern, entry.write_value));
    }
    let mut arena_event = 0usize;
    let merge_residual_before =
        |timestamp: u32,
         residual: &mut usize,
         last_touch: &mut HashMap<(u8, u64), u32>,
         current_value: &mut HashMap<(u8, u64), u64>| {
            while let Some(entry) = residual_memory_event(&output.raw_logs, *residual)? {
                if entry.timestamp >= timestamp {
                    break;
                }
                let key = (entry.addr_space, entry.address & !7u64);
                if !current_value.contains_key(&key) {
                    return Err(rvr_error(format!(
                        "delta residual event at timestamp {} has no first-touch seed",
                        entry.timestamp
                    )));
                }
                last_touch.insert(key, entry.timestamp);
                if entry.kind == PREFLIGHT_MEMORY_KIND_WRITE {
                    current_value.insert(key, entry.value);
                }
                *residual += 1;
            }
            Ok(())
        };

    let delta_record_count = delta_bytes.len() / PREFLIGHT_DELTA_RECORD_SIZE;
    for (delta_index, bytes) in delta_bytes
        .chunks_exact(PREFLIGHT_DELTA_RECORD_SIZE)
        .enumerate()
    {
        let record = DeltaRecord {
            from_pc: delta_u32(bytes, 0),
            from_timestamp: delta_u32(bytes, 4),
            v1: delta_u64(bytes, 8),
            v2: delta_u64(bytes, 16),
        };
        if record.from_pc < exe.program.pc_base
            || !(record.from_pc - exe.program.pc_base).is_multiple_of(DEFAULT_PC_STEP)
        {
            return Err(rvr_error(format!(
                "delta record {delta_index}/{delta_record_count} has invalid pc {:#x} at timestamp {}",
                record.from_pc, record.from_timestamp
            )));
        }
        while let Some(&(timestamp, pc, pattern, write_value)) = arena_events.get(arena_event) {
            if timestamp >= record.from_timestamp {
                break;
            }
            merge_residual_before(
                timestamp,
                &mut residual,
                &mut last_touch,
                &mut current_value,
            )?;
            let Some((instruction, _, _)) = instruction_and_air_idx(exe, pc_to_air_idx, pc)? else {
                return Err(rvr_error(format!(
                    "arena-native chronology pc {pc:#x} has no routed AIR"
                )));
            };
            let accesses = delta_accesses(instruction, pattern, 0);
            replay_delta_touches(
                &mut last_touch,
                accesses,
                delta_access_count(pattern),
                timestamp,
            );
            if let Some(slot) = delta_write_slot(instruction, pattern) {
                let key = accesses[slot].ok_or_else(|| {
                    rvr_error(format!(
                        "arena-native chronology pc {pc:#x} omitted its write address"
                    ))
                })?;
                if !current_value.contains_key(&key) {
                    return Err(rvr_error(format!(
                        "arena-native chronology pc {pc:#x} has no first-touch seed"
                    )));
                }
                current_value.insert(key, write_value);
            }
            arena_event += 1;
        }
        merge_residual_before(
            record.from_timestamp,
            &mut residual,
            &mut last_touch,
            &mut current_value,
        )?;

        let Some((instruction, air_idx, _)) =
            instruction_and_air_idx(exe, pc_to_air_idx, record.from_pc)?
        else {
            return Err(rvr_error(format!(
                "delta pc {:#x} has no routed AIR",
                record.from_pc
            )));
        };
        let registered = registry
            .inline_assemblers
            .get(&instruction.opcode)
            .ok_or_else(|| {
                rvr_error(format!(
                    "delta pc {:#x} opcode {:?} has no inline assembler",
                    record.from_pc, instruction.opcode
                ))
            })?;
        let pattern = registered.delta_pattern.ok_or_else(|| {
            rvr_error(format!(
                "delta pc {:#x} opcode {:?} has no access pattern",
                record.from_pc, instruction.opcode
            ))
        })?;

        let accesses = delta_accesses(instruction, pattern, record.v1);
        let prev = replay_delta_touches(
            &mut last_touch,
            accesses,
            delta_access_count(pattern),
            record.from_timestamp,
        );
        let mut write_prev_value = 0;
        if let Some(slot) = delta_write_slot(instruction, pattern) {
            let key = accesses[slot].ok_or_else(|| {
                rvr_error(format!(
                    "delta pc {:#x} omitted its write address",
                    record.from_pc
                ))
            })?;
            write_prev_value = current_value.get(&key).copied().ok_or_else(|| {
                rvr_error(format!(
                    "delta pc {:#x} has no first-touch seed for write {:?}",
                    record.from_pc, key
                ))
            })?;
            let write_value = if pattern == DeltaAccessPattern::Store {
                let width = registered.delta_store_width.ok_or_else(|| {
                    rvr_error(format!(
                        "delta store pc {:#x} has no registered patch width",
                        record.from_pc
                    ))
                })?;
                patch_delta_store(instruction, width, write_prev_value, record.v1, record.v2)
            } else {
                let derive = registry.delta_write_value.ok_or_else(|| {
                    rvr_error(format!(
                        "delta pc {:#x} opcode {:?} has no registered write-value derivation",
                        record.from_pc, instruction.opcode
                    ))
                })?;
                derive(instruction, record.from_pc, record.v1, record.v2).ok_or_else(|| {
                    rvr_error(format!(
                        "delta pc {:#x} opcode {:?} could not derive its post-write value",
                        record.from_pc, instruction.opcode
                    ))
                })?
            };
            current_value.insert(key, write_value);
        }

        let (stride, out) = per_air
            .entry(air_idx)
            .or_insert_with(|| (registered.record_size, Vec::new()));
        if *stride != registered.record_size {
            return Err(rvr_error(format!(
                "delta AIR {air_idx} has conflicting strides {stride} and {}",
                registered.record_size
            )));
        }
        append_u32(out, record.from_pc);
        append_u32(out, record.from_timestamp);
        match pattern {
            DeltaAccessPattern::Alu3
            | DeltaAccessPattern::Alu3Reg
            | DeltaAccessPattern::Load
            | DeltaAccessPattern::Store => {
                append_u32(out, prev[0]);
                append_u32(out, prev[1]);
                append_u32(out, prev[2]);
                append_u64(out, write_prev_value);
                append_u64(out, record.v1);
                append_u64(out, record.v2);
            }
            DeltaAccessPattern::Branch2 => {
                append_u32(out, prev[0]);
                append_u32(out, prev[1]);
                append_u64(out, record.v1);
                append_u64(out, record.v2);
            }
            DeltaAccessPattern::Wr1 | DeltaAccessPattern::Wr1Always => {
                append_u32(out, prev[0]);
                append_u64(out, write_prev_value);
            }
            DeltaAccessPattern::Rw1 => {
                append_u32(out, prev[0]);
                append_u32(out, prev[1]);
                append_u64(out, record.v1);
                append_u64(out, write_prev_value);
            }
        }
        synthetic_program.push(ProgramLogEntry::new(record.from_timestamp, record.from_pc));
    }

    // No later delta record consumes these touches, but replay the suffix to
    // keep the CPU oracle's chronology complete and validate every retained
    // arena-native event against its compiled access pattern.
    while let Some(&(timestamp, pc, pattern, write_value)) = arena_events.get(arena_event) {
        merge_residual_before(
            timestamp,
            &mut residual,
            &mut last_touch,
            &mut current_value,
        )?;
        let Some((instruction, _, _)) = instruction_and_air_idx(exe, pc_to_air_idx, pc)? else {
            return Err(rvr_error(format!(
                "arena-native chronology pc {pc:#x} has no routed AIR"
            )));
        };
        let accesses = delta_accesses(instruction, pattern, 0);
        replay_delta_touches(
            &mut last_touch,
            accesses,
            delta_access_count(pattern),
            timestamp,
        );
        if let Some(slot) = delta_write_slot(instruction, pattern) {
            let key = accesses[slot].ok_or_else(|| {
                rvr_error(format!(
                    "arena-native chronology pc {pc:#x} omitted its write address"
                ))
            })?;
            if !current_value.contains_key(&key) {
                return Err(rvr_error(format!(
                    "arena-native chronology pc {pc:#x} has no first-touch seed"
                )));
            }
            current_value.insert(key, write_value);
        }
        arena_event += 1;
    }

    let mut decoded = per_air
        .into_iter()
        .map(|(air_idx, (record_size, bytes))| RvrInlineChipRecords {
            air_idx,
            record_size,
            bytes,
        })
        .collect::<Vec<_>>();
    output.inline_records.append(&mut decoded);
    output.raw_logs.program_log.extend(synthetic_program);
    output
        .raw_logs
        .program_log
        .sort_unstable_by_key(|entry| entry.timestamp);
    Ok(Some(delta))
}

/// `(instruction, air_idx, program_slot_idx)` for one program-log pc.
type InstructionAirSlot<'a, F> = (&'a Instruction<F>, usize, usize);

fn instruction_and_air_idx<'a, F: PrimeField32>(
    exe: &'a VmExe<F>,
    pc_to_air_idx: &[Option<usize>],
    pc: u32,
) -> Result<Option<InstructionAirSlot<'a, F>>, ExecutionError> {
    if pc < exe.program.pc_base || !(pc - exe.program.pc_base).is_multiple_of(DEFAULT_PC_STEP) {
        return Err(rvr_error(format!(
            "program-log pc {pc:#x} is not a valid program pc"
        )));
    }
    let index = ((pc - exe.program.pc_base) / DEFAULT_PC_STEP) as usize;
    let Some(slot) = exe.program.instructions_and_debug_infos.get(index) else {
        return Err(rvr_error(format!(
            "program-log pc {pc:#x} is out of program bounds"
        )));
    };
    let Some((instruction, _)) = slot else {
        return Ok(None);
    };
    let Some(air_idx) = pc_to_air_idx.get(index).copied().flatten() else {
        return Ok(None);
    };
    Ok(Some((instruction, air_idx, index)))
}

fn rvr_error(message: String) -> ExecutionError {
    ExecutionError::RvrExecution(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::rvr::{DeltaMemoryLogEntry, MemoryLogEntry};

    fn raw_logs(delta_memory_log: Vec<DeltaMemoryLogEntry>) -> PreflightRawLogs {
        PreflightRawLogs {
            program_log: Vec::new(),
            memory_log: Vec::new(),
            delta_memory_log,
            chip_counts: Vec::new(),
            touched: Vec::new(),
        }
    }

    #[test]
    fn compact_residual_memory_completeness_fails_closed() {
        let logs = raw_logs(vec![DeltaMemoryLogEntry {
            timestamp: 9,
            address: 16,
            value: 7,
            kind: PREFLIGHT_MEMORY_KIND_WRITE,
            addr_space: 2,
            width: 8,
            complete: 0,
            _reserved: 0,
        }]);
        let err = residual_memory_event(&logs, 0).unwrap_err();
        assert!(err.to_string().contains("invalid compact delta residual"));
    }

    #[test]
    fn compact_residual_memory_accepts_touch_only_zero_width() {
        let logs = raw_logs(vec![DeltaMemoryLogEntry {
            timestamp: 9,
            address: 16,
            kind: PREFLIGHT_MEMORY_KIND_TOUCH,
            addr_space: 2,
            width: 0,
            complete: 1,
            _reserved: 0,
            ..Default::default()
        }]);
        let event = residual_memory_event(&logs, 0)
            .expect("touch-only compact entry must decode")
            .expect("touch-only compact entry must exist");
        assert_eq!(event.kind, PREFLIGHT_MEMORY_KIND_TOUCH);
        assert_eq!(event.address, 16);
    }

    #[test]
    fn dual_residual_memory_schemas_fail_closed() {
        let mut logs = raw_logs(vec![DeltaMemoryLogEntry {
            complete: 1,
            width: 8,
            ..Default::default()
        }]);
        logs.memory_log.push(MemoryLogEntry::default());
        let err = residual_memory_event(&logs, 0).unwrap_err();
        assert!(err.to_string().contains("both full and compact"));
    }
}
