//! rvr lifter for the rv32im I/O sub-extension (HINT_STOREW, HINT_BUFFER,
//! REVEAL).
//!
//! TODO: check if other RV32IM instructions/opcodes can be separated into
//! extensions.

use std::path::Path;

use openvm_instructions::{instruction::Instruction, riscv::RV32_MEMORY_AS, LocalOpcode};
use openvm_rv32im_transpiler::{Rv32HintStoreOpcode, Rv32LoadStoreOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_ffi_common::AS_PUBLIC_VALUES;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    decode_imm_cg, decode_reg, ExtensionError, RvrExtension, RvrExtensionCtx, NO_CHIP,
};

/// HINT_STOREW: pop 4 bytes from the hint stream into `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub struct HintStoreWInstr {
    pub ptr_reg: Reg,
}

impl ExtInstr for HintStoreWInstr {
    fn opname(&self) -> &str {
        "hint_storew"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_reg(self.ptr_reg);
        ctx.write_line(&format!(
            "trace_mem_access(state, {ptr}, {RV32_MEMORY_AS}u);"
        ));
        ctx.write_line(&format!("openvm_hint_storew({ptr});"));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_BUFFER: pop `4 * reg[num_words_reg]` bytes from the hint stream and
/// write them sequentially starting at `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub struct HintBufferInstr {
    pub ptr_reg: Reg,
    pub num_words_reg: Reg,
    pub chip_idx: u32,
}

impl ExtInstr for HintBufferInstr {
    fn opname(&self) -> &str {
        "hint_buffer"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_reg(self.ptr_reg);
        let n = ctx.read_reg(self.num_words_reg);
        if self.chip_idx != NO_CHIP {
            // Block-entry already credits a static +1; emit the runtime
            // `(n - 1)` correction only when there is more than one row.
            ctx.write_line(&format!("if ({n} > 1) {{"));
            ctx.write_line(&format!(
                "  trace_chip(state, {}u, {n} - 1);",
                self.chip_idx
            ));
            ctx.write_line("}");
        }
        ctx.write_line(&format!("if ({n} > 0) {{"));
        ctx.write_line(&format!(
            "  trace_mem_access_u32_range(state, {ptr}, {n}, {RV32_MEMORY_AS}u);"
        ));
        ctx.write_line("}");
        ctx.write_line(&format!("openvm_hint_buffer({ptr}, {n});"));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// REVEAL: write `reg[src_reg]` (4 bytes, little-endian) to user public-output
/// address space at `reg[ptr_reg] + offset`.
#[derive(Debug, Clone)]
pub struct RevealInstr {
    pub src_reg: Reg,
    pub ptr_reg: Reg,
    pub offset: u32,
}

impl ExtInstr for RevealInstr {
    fn opname(&self) -> &str {
        "reveal"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let src = ctx.read_reg(self.src_reg);
        let ptr = ctx.read_reg(self.ptr_reg);
        ctx.write_line(&format!(
            "trace_mem_access(state, {ptr}, {AS_PUBLIC_VALUES}u);"
        ));
        ctx.write_line(&format!(
            "openvm_reveal({src}, {ptr}, 0x{:08x}u);",
            self.offset
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// rvr extension for the rv32im I/O instructions HINT_STOREW, HINT_BUFFER, and
/// REVEAL.
pub struct Rv32IoExtension {
    hint_store_chip_idx: u32,
}

impl Rv32IoExtension {
    pub fn new_pure() -> Self {
        Self {
            hint_store_chip_idx: NO_CHIP,
        }
    }

    pub fn new(ctx: &RvrExtensionCtx) -> Result<Self, ExtensionError> {
        let opcode = Rv32HintStoreOpcode::HINT_STOREW.global_opcode();
        let hint_store_chip_idx = match ctx.resolve_opcode_air_idx(opcode) {
            Some(idx) => idx,
            None => NO_CHIP,
        };
        Ok(Self {
            hint_store_chip_idx,
        })
    }
}

impl<F: PrimeField32> RvrExtension<F> for Rv32IoExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == Rv32HintStoreOpcode::HINT_STOREW.global_opcode_usize() {
            let ptr_reg = decode_reg(insn.b);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(HintStoreWInstr { ptr_reg })),
                source_loc: None,
            }));
        }

        if opcode == Rv32HintStoreOpcode::HINT_BUFFER.global_opcode_usize() {
            let num_words_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(HintBufferInstr {
                    ptr_reg,
                    num_words_reg,
                    chip_idx: self.hint_store_chip_idx,
                })),
                source_loc: None,
            }));
        }

        // REVEAL: STOREW with address-space e = AS_PUBLIC_VALUES.
        if opcode == Rv32LoadStoreOpcode::STOREW.global_opcode_usize()
            && insn.e.as_canonical_u32() == AS_PUBLIC_VALUES
        {
            let src_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            let offset = decode_imm_cg(insn);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(RevealInstr {
                    src_reg,
                    ptr_reg,
                    offset,
                })),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn staticlib_paths(&self) -> Vec<&Path> {
        Vec::new()
    }

    fn staticlib_path(&self) -> &Path {
        // Unused: we override `staticlib_paths()` to return an empty list
        // because this extension has no native side-car library.
        Path::new("")
    }
}
