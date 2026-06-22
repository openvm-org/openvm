//! `VmExe<F>` -> `Vec<LiftedInstr>` conversion.

use std::collections::HashSet;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{LiftedInstr, SourceLoc};

use crate::{extension::ExtensionRegistry, opcode::lift_instruction};

/// Error during VmExe to IR conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("unrecognized opcode {opcode} at pc {pc:#x}")]
    UnrecognizedOpcode { opcode: usize, pc: u32 },
}

/// Convert a VmExe to a vector of lifted IR instructions.
pub fn convert_vmexe_to_ir<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &crate::extension::ExtensionRegistry<F>,
) -> Result<Vec<LiftedInstr>, ConvertError> {
    convert_vmexe_to_ir_with_debug(exe, extensions, |_| None)
}

/// Convert a VmExe to a vector of lifted IR instructions, optionally
/// attaching source locations from a caller-provided PC lookup.
///
/// This remains public because `rvr-openvm` consumes it across the crate
/// boundary when guest debug info is available.
pub fn convert_vmexe_to_ir_with_debug<F, G>(
    exe: &VmExe<F>,
    extensions: &ExtensionRegistry<F>,
    mut source_lookup: G,
) -> Result<Vec<LiftedInstr>, ConvertError>
where
    F: PrimeField32,
    G: FnMut(u32) -> Option<SourceLoc>,
{
    let mut lifted = Vec::new();

    for (pc, insn, _debug_info) in exe.program.enumerate_by_pc() {
        match lift_instruction(&insn, u64::from(pc), extensions) {
            Some(mut li) => {
                if let Some(loc) = source_lookup(pc) {
                    match &mut li {
                        LiftedInstr::Body(instr_at) => {
                            instr_at.source_loc = Some(loc.clone());
                        }
                        LiftedInstr::Term { source_loc, .. } => {
                            *source_loc = Some(loc);
                        }
                    }
                }
                lifted.push(li);
            }
            None => {
                return Err(ConvertError::UnrecognizedOpcode {
                    opcode: insn.opcode.as_usize(),
                    pc,
                });
            }
        }
    }

    Ok(lifted)
}

/// Scan the VmExe's init_memory for values that look like valid instruction PCs.
/// Candidates are read at 4-byte-aligned addresses (RISC-V instruction width),
/// not memory word (u64) boundaries. This discovers switch table entries,
/// function pointer arrays, and other code pointers embedded in read-only data.
pub fn scan_init_memory_for_code_pointers<F: PrimeField32>(
    exe: &VmExe<F>,
    valid_pcs: &HashSet<u64>,
) -> Vec<u64> {
    // Collect all bytes in address space 2 (main memory).
    let mut mem_bytes: std::collections::BTreeMap<u32, u8> = std::collections::BTreeMap::new();
    for (&(addr_space, addr), &byte) in &exe.init_memory {
        if addr_space == 2 {
            mem_bytes.insert(addr, byte);
        }
    }

    // Scan 4-byte-aligned addresses (RISC-V instruction width) for potential code pointers.
    let mut targets = Vec::new();
    let addrs: Vec<u32> = mem_bytes.keys().copied().collect();
    for &addr in &addrs {
        if addr % 4 != 0 {
            continue;
        }
        // Try to reconstruct a little-endian u32.
        if let (Some(&b0), Some(&b1), Some(&b2), Some(&b3)) = (
            mem_bytes.get(&addr),
            mem_bytes.get(&(addr + 1)),
            mem_bytes.get(&(addr + 2)),
            mem_bytes.get(&(addr + 3)),
        ) {
            let val = u64::from(u32::from_le_bytes([b0, b1, b2, b3]));
            if valid_pcs.contains(&val) {
                targets.push(val);
            }
        }
    }

    targets
}
