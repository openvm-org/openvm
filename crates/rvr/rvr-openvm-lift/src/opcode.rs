//! OpenVM system instruction lifting and RVR extension dispatch.

use openvm_instructions::{LocalOpcode, SysPhantom, SystemOpcode};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, InlineRecordShape, InstrAt, LiftedInstr, Terminator,
};

use crate::{ExtensionError, ExtensionRegistry, RvrInstruction};

/// Lift one OpenVM instruction.
///
/// System instructions are handled here. All remaining instructions are offered
/// to the registered extensions, and duplicate claims return an error.
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
        let operands = [insn.a, insn.b, insn.c];
        let discriminant = (insn.c & 0xffff) as u16;
        if let Some(phantom) = SysPhantom::from_repr(discriminant) {
            return Ok(Some(lift_system_phantom(pc, phantom, operands)));
        }
    }

    extensions.try_lift(insn, pc)
}

fn lift_system_phantom(pc: u64, phantom: SysPhantom, operands: [u32; 3]) -> LiftedInstr {
    match phantom {
        SysPhantom::Nop | SysPhantom::CtStart | SysPhantom::CtEnd => LiftedInstr::Body(InstrAt {
            pc,
            instr: Box::new(SystemPhantomInstr { operands }),
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
struct SystemPhantomInstr {
    operands: [u32; 3],
}

impl ExtInstr for SystemPhantomInstr {
    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        ctx.trace_phantom_record(self.operands);
    }

    fn opname(&self) -> &str {
        "phantom"
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom { record_size: 20 })
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NopInstr;

impl ExtInstr for NopInstr {
    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        ctx.trace_timestamp();
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
