//! OpenVM system instruction lifting and RVR extension dispatch.

use openvm_instructions::{LocalOpcode, SysPhantom, SystemOpcode};
use rvr_openvm_ir::{CfgEffect, EmitCtx, Instr, InstrAt, LiftedInstr, Terminator};

use crate::{ExtensionError, ExtensionRegistry, RvrInstruction};

pub fn lift_instruction(
    insn: &RvrInstruction,
    pc: u64,
    extensions: &ExtensionRegistry,
) -> Result<Option<LiftedInstr>, ExtensionError> {
    let opcode = insn.opcode.as_usize();

    if opcode == SystemOpcode::TERMINATE.global_opcode_usize() {
        return Ok(Some(LiftedInstr::Term {
            pc,
            terminator: Terminator::Exit { code: insn.c },
            source_loc: None,
        }));
    }

    if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
        let discriminant = (insn.c & 0xffff) as u16;
        if let Some(phantom) = SysPhantom::from_repr(discriminant) {
            return Ok(Some(lift_system_phantom(pc, phantom)));
        }
    }

    extensions.try_lift(insn, pc)
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
pub struct NopInstr;

impl Instr for NopInstr {
    fn emit_c(&self, _ctx: &mut dyn EmitCtx) {}

    fn opname(&self) -> &str {
        "nop"
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn Instr> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::VmOpcode;

    use super::*;

    #[test]
    fn unknown_phantom_is_not_silently_accepted() {
        let insn = RvrInstruction::from_canonical(
            VmOpcode::from_usize(SystemOpcode::PHANTOM.global_opcode_usize()),
            [0, 0, u32::from(u16::MAX), 0, 0, 0, 0],
            2_013_265_921,
        );
        assert!(lift_instruction(&insn, 0, &ExtensionRegistry::new())
            .unwrap()
            .is_none());
    }
}
