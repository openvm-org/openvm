//! Extension-extensible record assembly from rvr preflight logs.

use std::{cell::Cell, collections::HashMap};

use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::{PreflightMemoryAccessAux, RvrPreflightOutput};
use crate::arch::{Arena, ExecutionError, InlineRecordLayout};

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

/// Opcode-keyed registry of log-native record assemblers.
pub struct LogNativeAssemblerRegistry<F, RA> {
    assemblers: HashMap<VmOpcode, RegisteredAssembler<F, RA>>,
    inline_layouts: HashMap<VmOpcode, InlineRecordLayout>,
}

impl<F, RA> LogNativeAssemblerRegistry<F, RA> {
    pub fn new() -> Self {
        Self {
            assemblers: HashMap::new(),
            inline_layouts: HashMap::new(),
        }
    }

    /// Register the packed-record byte layout for opcodes the preflight
    /// codegen can migrate to inline compact records (R3). The assembler for
    /// these opcodes stays registered — it still runs when a library is
    /// compiled without inline records; which path a pc takes is decided by
    /// the compile metadata (`RvrPreflightOutput::inline_pc_slots`).
    pub fn register_inline_layout(
        &mut self,
        opcodes: impl IntoIterator<Item = VmOpcode>,
        layout: InlineRecordLayout,
    ) {
        for opcode in opcodes {
            assert!(
                self.inline_layouts.insert(opcode, layout).is_none(),
                "multiple inline record layouts registered for opcode {opcode:?}"
            );
        }
    }

    fn inline_layout(&self, opcode: &VmOpcode) -> Option<&InlineRecordLayout> {
        self.inline_layouts.get(opcode)
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
}

impl<F, RA> LogNativeOpcodeAdmitter<F> for LogNativeAssemblerRegistry<F, RA> {
    fn has_log_native_assembler(&self, instruction: &Instruction<F>) -> bool {
        self.contains_instruction(instruction)
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
/// entries and no assembler run: their pcs are skipped (per
/// `output.inline_pc_slots`, the compile metadata mirroring the generated C)
/// and each migrated chip's arena instead adopts the C-written record buffer
/// from `output.inline_records` (taken out of `output`). The adopted byte
/// count must equal the program-log record count for that chip, so a record
/// dropped in C (buffer overflow, unmapped chip) fails loudly here rather
/// than surfacing as a bus imbalance at proving.
pub fn generate_record_arenas_from_logs<F: PrimeField32, RA: Arena>(
    registry: &LogNativeAssemblerRegistry<F, RA>,
    exe: &VmExe<F>,
    output: &mut RvrPreflightOutput<F>,
    capacities: &[(usize, usize)],
    pc_to_air_idx: &[Option<usize>],
) -> Result<Vec<RA>, ExecutionError> {
    let inline_records = std::mem::take(&mut output.inline_records);
    let inline_pc_slots = output.inline_pc_slots.clone();
    let mut arenas = capacities
        .iter()
        .map(|&(height, width)| RA::with_capacity(height, width))
        .collect::<Vec<_>>();
    let access = LogNativeAccessView::new(&output.access_aux)?;

    let mut inline_record_counts = vec![0u64; arenas.len()];
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
        if inline_pc_slots.get(slot_idx).copied().unwrap_or(false) {
            let counts = inline_record_counts.get_mut(air_idx).ok_or_else(|| {
                rvr_error(format!(
                    "pc {:#x} maps to air_idx {} but only {} arenas exist",
                    program_entry.pc, air_idx, num_arenas
                ))
            })?;
            *counts += 1;
            continue;
        }
        let arena = arenas.get_mut(air_idx).ok_or_else(|| {
            rvr_error(format!(
                "pc {:#x} maps to air_idx {} but only {} arenas exist",
                program_entry.pc, air_idx, num_arenas
            ))
        })?;
        registry.assemble(arena, &access, instruction, pc, program_entry.timestamp)?;
    }

    let air_layouts = bind_inline_layouts_to_airs(registry, exe, &inline_pc_slots, pc_to_air_idx)?;
    for chip in inline_records {
        let air_idx = chip.air_idx;
        let layout = air_layouts.get(&air_idx).copied().ok_or_else(|| {
            rvr_error(format!(
                "inline record buffer for air_idx {air_idx} has no registered record layout"
            ))
        })?;
        if layout.record_size() != chip.record_size {
            return Err(rvr_error(format!(
                "inline record size mismatch for air_idx {air_idx}: compiled {} vs registered {}",
                chip.record_size,
                layout.record_size()
            )));
        }
        let expected_bytes = inline_record_counts
            .get(air_idx)
            .copied()
            .unwrap_or(0)
            .checked_mul(chip.record_size as u64)
            .expect("inline record bytes overflow u64");
        if chip.records.written as u64 != expected_bytes {
            return Err(rvr_error(format!(
                "inline record bytes for air_idx {air_idx} do not match the program log: \
                 C wrote {} bytes, expected {expected_bytes}",
                chip.records.written
            )));
        }
        let arena = arenas
            .get_mut(air_idx)
            .ok_or_else(|| rvr_error(format!("inline air_idx {air_idx} out of arena range")))?;
        arena
            .adopt_inline_records(chip.records, &layout)
            .map_err(|err| {
                rvr_error(format!(
                    "adopting inline records for air_idx {air_idx} failed: {err}"
                ))
            })?;
    }

    Ok(arenas)
}

/// Bind each inline-record chip (AIR) to its packed-record layout from the
/// registry, using the static program: every pc slot flagged inline maps its
/// opcode's registered layout to its AIR.
fn bind_inline_layouts_to_airs<F: PrimeField32, RA: Arena>(
    registry: &LogNativeAssemblerRegistry<F, RA>,
    exe: &VmExe<F>,
    inline_pc_slots: &[bool],
    pc_to_air_idx: &[Option<usize>],
) -> Result<HashMap<usize, InlineRecordLayout>, ExecutionError> {
    let mut air_layouts = HashMap::new();
    for (slot_idx, _) in inline_pc_slots
        .iter()
        .enumerate()
        .filter(|(_, &inline)| inline)
    {
        let instruction = exe
            .program
            .instructions_and_debug_infos
            .get(slot_idx)
            .and_then(|slot| slot.as_ref().map(|(instruction, _)| instruction))
            .ok_or_else(|| {
                rvr_error(format!(
                    "inline pc slot {slot_idx} has no program instruction"
                ))
            })?;
        let air_idx = pc_to_air_idx
            .get(slot_idx)
            .copied()
            .flatten()
            .ok_or_else(|| rvr_error(format!("inline pc slot {slot_idx} maps to no AIR index")))?;
        let layout = registry.inline_layout(&instruction.opcode).ok_or_else(|| {
            rvr_error(format!(
                "no inline record layout registered for opcode {:?} at inline pc slot {slot_idx}",
                instruction.opcode
            ))
        })?;
        if let Some(previous) = air_layouts.insert(air_idx, *layout) {
            if previous != *layout {
                return Err(rvr_error(format!(
                    "conflicting inline record layouts for air_idx {air_idx}"
                )));
            }
        }
    }
    Ok(air_layouts)
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
