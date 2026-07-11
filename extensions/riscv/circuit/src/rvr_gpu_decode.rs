//! M-GPUDEC (G2): per-exe device operand table + per-segment emission-mode
//! state, shared between the GPU builder (producer side: binds the exe, builds
//! the table, selects per-air modes each segment) and the migrated GPU chips
//! (consumer side: read the mode, upload the table once, launch the compact
//! decode kernels).
//!
//! One-derivation-two-consumers rule: table entries are produced by the same
//! `derive_*_operands` helpers the host inline assemblers use, so the CUDA
//! decoder is the only new derivation surface and the three-way differential
//! pins it.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

#[cfg(feature = "rvr")]
use openvm_instructions::{exe::VmExe, program::DEFAULT_PC_STEP, LocalOpcode};
#[cfg(feature = "rvr")]
use openvm_riscv_transpiler::{BaseAluOpcode, LessThanOpcode, ShiftOpcode};
#[cfg(feature = "rvr")]
use openvm_stark_backend::p3_field::PrimeField32;

#[cfg(feature = "rvr")]
use crate::log_native::derive_base_alu_u16_operands;

/// One 16-byte operand-table entry per program slot, indexed by
/// `(from_pc - pc_base) / DEFAULT_PC_STEP`. Field meanings are per wire
/// format; for alu3 over the BaseAluU16 adapter: `a` = rd_ptr, `b` = rs1_ptr,
/// `c` = rs2 (register ptr or immediate), flags carry rs2_as/imm-sign, and
/// `local_opcode` is the class-local opcode index for the owning AIR.
/// Mirrored by the CUDA `RvrOperandEntry` (static-asserted there).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DeviceOperandEntry {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub flags: u8,
    pub local_opcode: u8,
    pub _reserved: u16,
}

const _: () = assert!(size_of::<DeviceOperandEntry>() == 16);
const _: () = assert!(align_of::<DeviceOperandEntry>() == 4);

/// `rs2` is an immediate (else a register pointer).
pub const OPERAND_FLAG_RS2_IMM: u8 = 1 << 0;
/// The sign-extension bit for a 16-bit-signed immediate (`rs2_imm_sign`).
pub const OPERAND_FLAG_RS2_IMM_SIGN: u8 = 1 << 1;

/// How a migrated AIR's records are fed to the GPU this segment (naming
/// shared with the R4-fuse work).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineEmissionMode {
    /// Compact wire records + on-device operand-table decode (G2).
    CompactWire,
    /// Full in-arena records written by C, fed zero-copy (G1; requires the
    /// R4 batch-2 arena-native emitters).
    ArenaNative,
}

/// Env toggle for the staged G1/G2 measurement matrix. Unset (or any other
/// value) = expanded host records, the gate-validated default path.
pub fn configured_emission_mode() -> Option<InlineEmissionMode> {
    match std::env::var("OPENVM_RVR_GPU_RECORDS").as_deref() {
        Ok("compact") => Some(InlineEmissionMode::CompactWire),
        Ok("arena-native") => Some(InlineEmissionMode::ArenaNative),
        _ => None,
    }
}

struct HostOperandTable {
    /// Identity of the bound exe: (pc_base, program slot count). A VM binds one
    /// exe in practice; this guard rebuilds the table if that ever changes.
    exe_key: (u32, usize),
    pc_base: u32,
    entries: Arc<Vec<DeviceOperandEntry>>,
}

/// Shared producer/consumer state. The builder holds one `Arc` per VM and
/// clones it into every migrated GPU chip at construction.
#[derive(Default)]
pub struct RvrGpuDecodeState {
    table: Mutex<Option<HostOperandTable>>,
    /// Per-SEGMENT emission mode per AIR index; refreshed on every
    /// `generate_rvr_record_arenas_from_logs` call (so a segment produced by a
    /// different route can never be misread). Empty = everything expanded.
    segment_modes: Mutex<HashMap<usize, InlineEmissionMode>>,
}

impl RvrGpuDecodeState {
    /// The emission mode the current segment's arena for `air_idx` was built
    /// with, if any.
    pub fn mode_for(&self, air_idx: usize) -> Option<InlineEmissionMode> {
        self.segment_modes.lock().unwrap().get(&air_idx).copied()
    }

    /// The bound exe's operand table + its pc base (consumer side).
    pub fn operand_table(&self) -> Option<(Arc<Vec<DeviceOperandEntry>>, u32)> {
        let table = self.table.lock().unwrap();
        table.as_ref().map(|t| (Arc::clone(&t.entries), t.pc_base))
    }

    /// Clear all per-segment modes (used when the toggle is off or the route
    /// produced expanded arenas).
    pub fn clear_segment_modes(&self) {
        self.segment_modes.lock().unwrap().clear();
    }

    /// Producer side: bind `exe` (building the alu_u16-family operand table if
    /// this exe isn't already bound) and mark the family's AIRs as
    /// `CompactWire` for the CURRENT segment. Returns the compact AIR set.
    #[cfg(feature = "rvr")]
    pub fn bind_alu_u16_compact_segment<F: PrimeField32>(
        &self,
        exe: &VmExe<F>,
        pc_to_air_idx: &[Option<usize>],
    ) -> HashSet<usize> {
        let exe_key = (
            exe.program.pc_base,
            exe.program.instructions_and_debug_infos.len(),
        );
        let mut airs = HashSet::new();
        {
            let mut table = self.table.lock().unwrap();
            let rebuild = table.as_ref().map(|t| t.exe_key != exe_key).unwrap_or(true);
            if rebuild {
                *table = Some(build_alu_u16_operand_table(exe, exe_key));
            }
        }
        for (slot_idx, slot) in exe.program.instructions_and_debug_infos.iter().enumerate() {
            let Some((instruction, _)) = slot else {
                continue;
            };
            if alu_u16_local_opcode(instruction.opcode.as_usize()).is_none() {
                continue;
            }
            if let Some(air_idx) = pc_to_air_idx.get(slot_idx).copied().flatten() {
                airs.insert(air_idx);
            }
        }
        // The family maps to exactly the four BaseAluU16-adapter AIRs; more
        // means the opcode->air routing changed under us.
        assert!(
            airs.len() <= 4,
            "alu_u16 family mapped to {} AIRs (expected <= 4)",
            airs.len()
        );
        let mut modes = self.segment_modes.lock().unwrap();
        modes.clear();
        for &air in &airs {
            modes.insert(air, InlineEmissionMode::CompactWire);
        }
        airs
    }
}

/// Class-local opcode index if `opcode` belongs to the alu_u16 (BaseAluU16
/// adapter) family: AddSub (ADD/SUB), LessThan (SLT/SLTU), ShiftLogical
/// (SLL/SRL), ShiftRightArithmetic (SRA).
#[cfg(feature = "rvr")]
fn alu_u16_local_opcode(opcode: usize) -> Option<u8> {
    if opcode == BaseAluOpcode::ADD.global_opcode_usize()
        || opcode == BaseAluOpcode::SUB.global_opcode_usize()
    {
        return Some((opcode - BaseAluOpcode::CLASS_OFFSET) as u8);
    }
    if opcode == LessThanOpcode::SLT.global_opcode_usize()
        || opcode == LessThanOpcode::SLTU.global_opcode_usize()
    {
        return Some((opcode - LessThanOpcode::CLASS_OFFSET) as u8);
    }
    if opcode == ShiftOpcode::SLL.global_opcode_usize()
        || opcode == ShiftOpcode::SRL.global_opcode_usize()
        || opcode == ShiftOpcode::SRA.global_opcode_usize()
    {
        return Some((opcode - ShiftOpcode::CLASS_OFFSET) as u8);
    }
    None
}

#[cfg(feature = "rvr")]
fn build_alu_u16_operand_table<F: PrimeField32>(
    exe: &VmExe<F>,
    exe_key: (u32, usize),
) -> HostOperandTable {
    let program = &exe.program;
    let mut entries =
        vec![DeviceOperandEntry::default(); program.instructions_and_debug_infos.len()];
    for (slot_idx, slot) in program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = slot else {
            continue;
        };
        let Some(local_opcode) = alu_u16_local_opcode(instruction.opcode.as_usize()) else {
            continue;
        };
        let operands = derive_base_alu_u16_operands(instruction);
        let mut flags = 0u8;
        if operands.rs2_as != openvm_instructions::riscv::RV64_REGISTER_AS as u8 {
            flags |= OPERAND_FLAG_RS2_IMM;
        }
        if operands.rs2_imm_sign {
            flags |= OPERAND_FLAG_RS2_IMM_SIGN;
        }
        entries[slot_idx] = DeviceOperandEntry {
            a: operands.rd_ptr,
            b: operands.rs1_ptr,
            c: operands.rs2,
            flags,
            local_opcode,
            _reserved: 0,
        };
    }
    let _ = DEFAULT_PC_STEP; // index = (from_pc - pc_base) / DEFAULT_PC_STEP
    HostOperandTable {
        exe_key,
        pc_base: program.pc_base,
        entries: Arc::new(entries),
    }
}
