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
    collections::HashSet,
    sync::{Arc, Mutex},
};

use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
#[cfg(feature = "rvr")]
use openvm_instructions::{
    exe::VmExe, instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode,
};
#[cfg(feature = "rvr")]
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, LessThanOpcode, Rv64AuipcOpcode,
    Rv64JalLuiOpcode, Rv64JalrOpcode, ShiftOpcode,
};
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
/// wr1/rw1: the conditional rd write is enabled (`f != 0`; x0 suppression).
pub const OPERAND_FLAG_WRITE_ENABLED: u8 = 1 << 2;
/// wr1 (JalLui): the opcode is JAL (else LUI).
pub const OPERAND_FLAG_IS_JAL: u8 = 1 << 3;
/// rw1 (Jalr): the immediate sign bit (`g != 0`).
pub const OPERAND_FLAG_JALR_IMM_SIGN: u8 = 1 << 4;

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
    /// Per-SEGMENT emission mode of the decode-kernel formats — all supported
    /// AIRs toggle together. Refreshed on every
    /// `generate_rvr_record_arenas_from_logs` call, so a segment produced by a
    /// different route can never be misread. `None` = expanded.
    compact_mode: Mutex<Option<InlineEmissionMode>>,
    /// Device copy of the operand table, keyed by the host table's identity.
    device_table: Mutex<Option<(usize, Arc<DeviceBuffer<u8>>)>>,
}

impl RvrGpuDecodeState {
    /// The emission mode the current segment's alu_u16-family arenas were
    /// built with, if any.
    pub fn compact_segment_mode(&self) -> Option<InlineEmissionMode> {
        *self.compact_mode.lock().unwrap()
    }

    /// The device operand table (uploaded once per bound exe) + its pc base.
    pub fn device_operand_table(
        &self,
        device_ctx: &GpuDeviceCtx,
    ) -> Option<(Arc<DeviceBuffer<u8>>, u32)> {
        let (entries, pc_base) = self.operand_table()?;
        let key = Arc::as_ptr(&entries) as usize;
        let mut cache = self.device_table.lock().unwrap();
        if let Some((cached_key, buf)) = cache.as_ref() {
            if *cached_key == key {
                return Some((Arc::clone(buf), pc_base));
            }
        }
        // SAFETY: DeviceOperandEntry is repr(C), size 16, no padding invariants
        // beyond the static asserts; reinterpreting as bytes for H2D is sound.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                entries.as_ptr() as *const u8,
                entries.len() * size_of::<DeviceOperandEntry>(),
            )
        };
        let buf = Arc::new(bytes.to_device_on(device_ctx).expect("operand table H2D"));
        *cache = Some((key, Arc::clone(&buf)));
        Some((buf, pc_base))
    }

    /// The bound exe's operand table + its pc base (consumer side).
    pub fn operand_table(&self) -> Option<(Arc<Vec<DeviceOperandEntry>>, u32)> {
        let table = self.table.lock().unwrap();
        table.as_ref().map(|t| (Arc::clone(&t.entries), t.pc_base))
    }

    /// Clear the per-segment mode (used when the toggle is off or the route
    /// produced expanded arenas).
    pub fn clear_segment_modes(&self) {
        *self.compact_mode.lock().unwrap() = None;
    }

    /// Producer side: bind `exe` (building the alu_u16-family operand table if
    /// this exe isn't already bound) and mark the family's AIRs as
    /// `CompactWire` for the CURRENT segment. Returns the compact AIR set.
    #[cfg(feature = "rvr")]
    pub fn bind_compact_segment<F: PrimeField32>(
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
                *table = Some(build_operand_table(exe, exe_key));
            }
        }
        for (slot_idx, slot) in exe.program.instructions_and_debug_infos.iter().enumerate() {
            let Some((instruction, _)) = slot else {
                continue;
            };
            if gpu_decode_entry(instruction).is_none() {
                continue;
            }
            if let Some(air_idx) = pc_to_air_idx.get(slot_idx).copied().flatten() {
                airs.insert(air_idx);
            }
        }
        // The supported formats map to exactly the nine decode-kernel AIRs
        // (AddSub, LessThan, ShiftLogical, SRA, BranchEqual, BranchLessThan,
        // JalLui, Auipc, Jalr); more means opcode->air routing changed.
        assert!(
            airs.len() <= 9,
            "gpu-decode formats mapped to {} AIRs (expected <= 9)",
            airs.len()
        );
        *self.compact_mode.lock().unwrap() = if airs.is_empty() {
            None
        } else {
            Some(InlineEmissionMode::CompactWire)
        };
        airs
    }
}

/// The operand-table entry for one instruction, if its opcode belongs to a
/// wire format with a device decode kernel. THE shared list: the table
/// builder and the per-segment bind both consult this, so the compact air
/// set and the table can never drift apart. Derivations mirror the host
/// inline assemblers exactly (alu3 via `derive_base_alu_u16_operands`).
#[cfg(feature = "rvr")]
fn gpu_decode_entry<F: PrimeField32>(instruction: &Instruction<F>) -> Option<DeviceOperandEntry> {
    use openvm_instructions::riscv::RV64_REGISTER_AS;
    let opcode = instruction.opcode.as_usize();

    // alu3 over the BaseAluU16 adapter: AddSub, LessThan, ShiftLogical, SRA.
    let alu_u16_local = if opcode == BaseAluOpcode::ADD.global_opcode_usize()
        || opcode == BaseAluOpcode::SUB.global_opcode_usize()
    {
        Some((opcode - BaseAluOpcode::CLASS_OFFSET) as u8)
    } else if opcode == LessThanOpcode::SLT.global_opcode_usize()
        || opcode == LessThanOpcode::SLTU.global_opcode_usize()
    {
        Some((opcode - LessThanOpcode::CLASS_OFFSET) as u8)
    } else if opcode == ShiftOpcode::SLL.global_opcode_usize()
        || opcode == ShiftOpcode::SRL.global_opcode_usize()
        || opcode == ShiftOpcode::SRA.global_opcode_usize()
    {
        Some((opcode - ShiftOpcode::CLASS_OFFSET) as u8)
    } else {
        None
    };
    if let Some(local_opcode) = alu_u16_local {
        let operands = derive_base_alu_u16_operands(instruction);
        let mut flags = 0u8;
        if operands.rs2_as != RV64_REGISTER_AS as u8 {
            flags |= OPERAND_FLAG_RS2_IMM;
        }
        if operands.rs2_imm_sign {
            flags |= OPERAND_FLAG_RS2_IMM_SIGN;
        }
        return Some(DeviceOperandEntry {
            a: operands.rd_ptr,
            b: operands.rs1_ptr,
            c: operands.rs2,
            flags,
            local_opcode,
            _reserved: 0,
        });
    }

    // branch2: BranchEqual (BEQ/BNE), BranchLessThan (BLT/BLTU/BGE/BGEU).
    let branch_local = if opcode == BranchEqualOpcode::BEQ.global_opcode_usize()
        || opcode == BranchEqualOpcode::BNE.global_opcode_usize()
    {
        Some((opcode - BranchEqualOpcode::CLASS_OFFSET) as u8)
    } else if opcode == BranchLessThanOpcode::BLT.global_opcode_usize()
        || opcode == BranchLessThanOpcode::BLTU.global_opcode_usize()
        || opcode == BranchLessThanOpcode::BGE.global_opcode_usize()
        || opcode == BranchLessThanOpcode::BGEU.global_opcode_usize()
    {
        Some((opcode - BranchLessThanOpcode::CLASS_OFFSET) as u8)
    } else {
        None
    };
    if let Some(local_opcode) = branch_local {
        return Some(DeviceOperandEntry {
            a: instruction.a.as_canonical_u32(),
            b: instruction.b.as_canonical_u32(),
            c: instruction.c.as_canonical_u32(),
            flags: 0,
            local_opcode,
            _reserved: 0,
        });
    }

    // wr1: JalLui (conditional write via f) and Auipc (always written).
    if opcode == Rv64JalLuiOpcode::JAL.global_opcode_usize()
        || opcode == Rv64JalLuiOpcode::LUI.global_opcode_usize()
    {
        let mut flags = 0u8;
        if !instruction.f.is_zero() {
            flags |= OPERAND_FLAG_WRITE_ENABLED;
        }
        if opcode == Rv64JalLuiOpcode::JAL.global_opcode_usize() {
            flags |= OPERAND_FLAG_IS_JAL;
        }
        return Some(DeviceOperandEntry {
            a: instruction.a.as_canonical_u32(),
            b: 0,
            c: instruction.c.as_canonical_u32(),
            flags,
            local_opcode: 0,
            _reserved: 0,
        });
    }
    if opcode == Rv64AuipcOpcode::AUIPC.global_opcode_usize() {
        return Some(DeviceOperandEntry {
            a: instruction.a.as_canonical_u32(),
            b: 0,
            c: instruction.c.as_canonical_u32(),
            flags: OPERAND_FLAG_WRITE_ENABLED,
            local_opcode: 0,
            _reserved: 0,
        });
    }

    // rw1: Jalr (conditional write via f, imm sign via g).
    if opcode == Rv64JalrOpcode::JALR.global_opcode_usize() {
        let mut flags = 0u8;
        if !instruction.f.is_zero() {
            flags |= OPERAND_FLAG_WRITE_ENABLED;
        }
        if !instruction.g.is_zero() {
            flags |= OPERAND_FLAG_JALR_IMM_SIGN;
        }
        return Some(DeviceOperandEntry {
            a: instruction.a.as_canonical_u32(),
            b: instruction.b.as_canonical_u32(),
            c: instruction.c.as_canonical_u32(),
            flags,
            local_opcode: 0,
            _reserved: 0,
        });
    }

    None
}

#[cfg(feature = "rvr")]
fn build_operand_table<F: PrimeField32>(exe: &VmExe<F>, exe_key: (u32, usize)) -> HostOperandTable {
    let program = &exe.program;
    let mut entries =
        vec![DeviceOperandEntry::default(); program.instructions_and_debug_infos.len()];
    for (slot_idx, slot) in program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = slot else {
            continue;
        };
        if let Some(entry) = gpu_decode_entry(instruction) {
            entries[slot_idx] = entry;
        }
    }
    let _ = DEFAULT_PC_STEP; // index = (from_pc - pc_base) / DEFAULT_PC_STEP
    HostOperandTable {
        exe_key,
        pc_base: program.pc_base,
        entries: Arc::new(entries),
    }
}
