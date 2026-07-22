//! Shared RV64 instruction decoding and IR helpers.

use openvm_instructions::riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES};
use rvr_openvm_ir::{CfgEffect, CfgOperand, ExtEmitCtx, ExtInstr, ValueSlot};
use rvr_openvm_lift::{decode_value_slot, RvrInstruction};

/// An RV64 integer register represented by the target-neutral IR slot handle.
pub(crate) type Reg = ValueSlot;

pub(crate) const ZERO: Reg = Reg::new(0);
pub(crate) const RA: Reg = Reg::new(1);

/// Decode an OpenVM RV64 register operand into its architectural register.
pub(crate) fn decode_reg(value: u32) -> Reg {
    decode_value_slot(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

/// Decode a sign-extended immediate from the register-width limb encoding.
pub(crate) fn decode_imm_cg(insn: &RvrInstruction) -> u32 {
    let low16 = insn.c & 0xffff;
    low16.wrapping_add(if insn.g != 0 { 0xffff_0000 } else { 0 })
}

pub(crate) const fn reg_operand(reg: Reg) -> CfgOperand {
    if reg.index() == 0 {
        CfgOperand::Const(0)
    } else {
        CfgOperand::Slot(reg)
    }
}

pub(crate) fn hex_u64(value: u64) -> String {
    format!("0x{value:016x}ull")
}

/// Architectural no-op used when an instruction writes to `x0`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NopInstr;

impl ExtInstr for NopInstr {
    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

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
        Box::new(*self)
    }
}
