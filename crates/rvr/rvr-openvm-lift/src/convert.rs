//! `VmExe` -> `Vec<LiftedInstr>` conversion.

use openvm_instructions::exe::VmExe;
use rvr_openvm_ir::{LiftedInstr, SourceLoc};

use crate::{extension::ExtensionRegistry, opcode::lift_instruction, ExtensionError};

/// Error during VmExe to IR conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("unrecognized opcode {opcode} at pc {pc:#x}")]
    UnrecognizedOpcode { opcode: usize, pc: u64 },
    #[error(transparent)]
    Extension(#[from] ExtensionError),
}

/// Convert a VmExe to a vector of lifted IR instructions.
pub fn convert_vmexe_to_ir(
    exe: &VmExe,
    extensions: &ExtensionRegistry,
) -> Result<Vec<LiftedInstr>, ConvertError> {
    convert_vmexe_to_ir_with_debug(exe, extensions, |_| None)
}

/// Convert a VmExe to a vector of lifted IR instructions, optionally
/// attaching source locations from a caller-provided PC lookup.
///
/// This remains public because `rvr-openvm` consumes it across the crate
/// boundary when guest debug info is available.
pub fn convert_vmexe_to_ir_with_debug<G>(
    exe: &VmExe,
    extensions: &ExtensionRegistry,
    mut source_lookup: G,
) -> Result<Vec<LiftedInstr>, ConvertError>
where
    G: FnMut(u32) -> Option<SourceLoc>,
{
    let mut lifted = Vec::new();
    for (pc, insn, _debug_info) in exe.program.enumerate_by_pc() {
        match lift_instruction(&insn, u64::from(pc), extensions)? {
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
                    pc: u64::from(pc),
                });
            }
        }
    }

    Ok(lifted)
}
