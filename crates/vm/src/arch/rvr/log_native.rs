//! Extension-extensible record assembly from rvr preflight logs.

use std::{cell::Cell, collections::HashMap};

use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{PreflightMemoryAccessAux, RvrInlineChipRecords, RvrPreflightOutput};
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
    Alu3ArenaFieldOffsets, ArenaNativeGeometry, ArenaNativeLayout, Branch2ArenaFieldOffsets,
    LoadStoreArenaFieldOffsets, Rw1ArenaFieldOffsets, Wr1ArenaFieldOffsets,
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
}

/// Opcode-keyed registry of log-native record assemblers.
pub struct LogNativeAssemblerRegistry<F, RA> {
    assemblers: HashMap<VmOpcode, RegisteredAssembler<F, RA>>,
    inline_assemblers: HashMap<VmOpcode, RegisteredInlineAssembler<F, RA>>,
}

impl<F, RA> LogNativeAssemblerRegistry<F, RA> {
    pub fn new() -> Self {
        Self {
            assemblers: HashMap::new(),
            inline_assemblers: HashMap::new(),
        }
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
                        }
                    )
                    .is_none(),
                "multiple inline record assemblers registered for opcode {opcode:?}"
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
    let access = LogNativeAccessView::new(&output.access_aux)?;

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

    for program_entry in &output.raw_logs.program_log {
        let pc = u32::try_from(program_entry.pc).map_err(|_| {
            rvr_error(format!(
                "program-log pc {:#x} does not fit OpenVM pc width",
                program_entry.pc
            ))
        })?;
        let Some((instruction, air_idx, slot_idx)) =
            instruction_and_air_idx(exe, pc_to_air_idx, pc)?
        else {
            continue;
        };
        let num_arenas = arenas.len();
        let arena = arenas.get_mut(air_idx).ok_or_else(|| {
            rvr_error(format!(
                "pc {:#x} maps to air_idx {} but only {} arenas exist",
                program_entry.pc, air_idx, num_arenas
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
                continue;
            }
            (registered.assemble)(arena, instruction, compact, pc)?;
            continue;
        }
        registry.assemble(arena, &access, instruction, pc, program_entry.timestamp)?;
    }

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

    Ok((arenas, wire_buffers))
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
