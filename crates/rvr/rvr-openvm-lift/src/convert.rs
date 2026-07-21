//! `VmExe<F>` -> `Vec<LiftedInstr>` conversion.

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{LiftedInstr, SourceLoc};

use crate::{extension::ExtensionRegistry, opcode::lift_instruction, RvrInstruction};

/// Error during VmExe to IR conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("unrecognized opcode {opcode} at pc {pc:#x}")]
    UnrecognizedOpcode { opcode: usize, pc: u64 },
    #[error(transparent)]
    Extension(#[from] crate::ExtensionError),
}

/// Convert a VmExe to a vector of lifted IR instructions.
pub fn convert_vmexe_to_ir<F: PrimeField32>(
    exe: &VmExe<F>,
    extensions: &crate::extension::ExtensionRegistry,
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
    extensions: &ExtensionRegistry,
    mut source_lookup: G,
) -> Result<Vec<LiftedInstr>, ConvertError>
where
    F: PrimeField32,
    G: FnMut(u32) -> Option<SourceLoc>,
{
    let mut lifted = Vec::new();
    for (pc, insn, _debug_info) in exe.program.enumerate_by_pc() {
        let insn = RvrInstruction::from_field(&insn);
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
