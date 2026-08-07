//! OpenVM system instruction lifting and RVR extension dispatch.

use openvm_instructions::{
    instruction::{Instruction, InstructionOperand},
    LocalOpcode, SysPhantom, SystemOpcode,
};
use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, Terminator};

use crate::{ExtensionError, ExtensionRegistry};

/// Lift one OpenVM instruction.
///
/// System instructions are handled here. All remaining instructions are offered
/// to the registered extensions, and duplicate claims return an error.
pub fn lift_instruction(
    insn: &Instruction,
    pc: u64,
    extensions: &ExtensionRegistry,
) -> Result<Option<LiftedInstr>, ExtensionError> {
    let opcode = insn.opcode.as_usize();

    if opcode == SystemOpcode::TERMINATE.global_opcode_usize() {
        let Some(code) = insn.c.checked_as_u32() else {
            return Err(ExtensionError::InvalidInstruction {
                opcode: insn.opcode,
                pc,
            });
        };
        return Ok(Some(LiftedInstr::Term {
            pc,
            terminator: Terminator::Exit { code },
            source_loc: None,
        }));
    }

    if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
        if let Some(phantom) = decode_system_phantom(insn, pc)? {
            return Ok(Some(lift_system_phantom(pc, phantom)));
        }
    }

    extensions.try_lift(insn, pc)
}

fn decode_system_phantom(
    insn: &Instruction,
    pc: u64,
) -> Result<Option<SysPhantom>, ExtensionError> {
    let invalid = || ExtensionError::InvalidInstruction {
        opcode: insn.opcode,
        pc,
    };
    let require_u32 = |operand: InstructionOperand| operand.checked_as_u32().ok_or_else(&invalid);

    if [insn.e, insn.f, insn.g]
        .into_iter()
        .any(|operand| !operand.is_zero())
    {
        return Err(invalid());
    }

    require_u32(insn.a)?;
    require_u32(insn.b)?;

    let discriminant = u16::try_from(require_u32(insn.c)?).map_err(|_| invalid())?;
    let _c_upper = u16::try_from(require_u32(insn.d)?).map_err(|_| invalid())?;

    Ok(SysPhantom::from_repr(discriminant))
}

fn lift_system_phantom(pc: u64, phantom: SysPhantom) -> LiftedInstr {
    match phantom {
        SysPhantom::Nop | SysPhantom::CtStart | SysPhantom::CtEnd => LiftedInstr::Body(InstrAt {
            pc,
            instr: Box::new(NopInstr),
            source_loc: None,
        }),
        SysPhantom::DebugPanic => LiftedInstr::Term {
            pc,
            terminator: Terminator::Trap {
                message: "PHANTOM DebugPanic".to_string(),
            },
            source_loc: None,
        },
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NopInstr;

impl ExtInstr for NopInstr {
    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        ctx.advance_timestamp(1);
    }

    fn opname(&self) -> &str {
        "nop"
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn supports_preflight(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::VmOpcode;

    use super::*;

    #[test]
    fn unknown_phantom_is_not_silently_accepted() {
        let insn = Instruction::from_usize(
            VmOpcode::from_usize(SystemOpcode::PHANTOM.global_opcode_usize()),
            [0, 0, usize::from(u16::MAX)],
        );
        assert!(lift_instruction(&insn, 0, &ExtensionRegistry::new())
            .unwrap()
            .is_none());
    }

    #[test]
    fn terminate_rejects_negative_exit_code() {
        let insn = Instruction::from_isize(
            VmOpcode::from_usize(SystemOpcode::TERMINATE.global_opcode_usize()),
            0,
            0,
            -1,
            0,
            0,
        );

        assert!(matches!(
            lift_instruction(&insn, 0, &ExtensionRegistry::new()),
            Err(ExtensionError::InvalidInstruction { .. })
        ));
    }

    #[test]
    fn phantom_rejects_invalid_shape() {
        let opcode = VmOpcode::from_usize(SystemOpcode::PHANTOM.global_opcode_usize());
        let discriminant = SysPhantom::Nop as usize;
        let invalid = [
            Instruction::from_isize(opcode, -1, 0, discriminant as isize, 0, 0),
            Instruction::from_isize(opcode, 0, -1, discriminant as isize, 0, 0),
            Instruction::from_isize(opcode, 0, 0, -1, 0, 0),
            Instruction::from_isize(opcode, 0, 0, discriminant as isize, -1, 0),
            Instruction::from_usize(opcode, [0, 0, usize::from(u16::MAX) + 1]),
            Instruction::from_usize(opcode, [0, 0, discriminant, usize::from(u16::MAX) + 1]),
            Instruction::from_usize(opcode, [0, 0, discriminant, 0, 1]),
        ];

        for insn in invalid {
            assert!(matches!(
                lift_instruction(&insn, 0, &ExtensionRegistry::new()),
                Err(ExtensionError::InvalidInstruction { .. })
            ));
        }
    }
}
