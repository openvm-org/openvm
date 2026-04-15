//! VmExe<F> -> Vec<LiftedInstr> conversion.

use std::collections::HashSet;

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{LiftedInstr, SourceLoc};

use crate::extension::ExtensionRegistry;
use crate::opcode::lift_instruction;

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
        match lift_instruction(&insn, pc, extensions) {
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

/// Scan the VmExe's init_memory for 4-byte-aligned values that look like
/// valid instruction PCs. This discovers switch table entries, function
/// pointer arrays, and other code pointers embedded in read-only data.
pub fn scan_init_memory_for_code_pointers<F: PrimeField32>(
    exe: &VmExe<F>,
    valid_pcs: &HashSet<u32>,
) -> Vec<u32> {
    // Collect all bytes in address space 2 (main memory).
    let mut mem_bytes: std::collections::BTreeMap<u32, u8> = std::collections::BTreeMap::new();
    for (&(addr_space, addr), &byte) in &exe.init_memory {
        if addr_space == 2 {
            mem_bytes.insert(addr, byte);
        }
    }

    // Scan 4-byte-aligned addresses for potential code pointers.
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
            let val = u32::from_le_bytes([b0, b1, b2, b3]);
            if valid_pcs.contains(&val) {
                targets.push(val);
            }
        }
    }

    targets
}
