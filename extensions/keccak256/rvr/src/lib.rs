//! Keccak-256 extension: IR nodes for KECCAKF + XORIN and the
//! `KeccakExtension` that lifts and emits them.
//!
//! The keccak-f permutation runs in C against the keccak-ffi staticlib; the
//! `.c` shim is emitted alongside generated code so clang can inline the
//! tracer helpers across the call boundary.

use openvm_instructions::{
    riscv::{RV64_NUM_REGISTERS, RV64_REGISTER_BYTES},
    LocalOpcode,
};
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, FixedTraceRows, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    decode_variable, fixed_trace_rows_for_chip, max_main_memory_pages_for_contiguous_range,
    opcode_air_idx, AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction,
};

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, RV64_REGISTER_BYTES as u32, RV64_NUM_REGISTERS as u32)
}

const KECCAK_NUM_ROUNDS: u32 = p3_keccak_air::NUM_ROUNDS as u32;
const _: () = assert!(KECCAK_NUM_ROUNDS as usize == p3_keccak_air::NUM_ROUNDS);
// XORIN reads one 136-byte rate buffer, writes it back, and separately reads its input.
const XORIN_MAX_PAGES: usize = 3 * max_main_memory_pages_for_contiguous_range(136);
// KECCAKF reads and writes the 200-byte state in place.
const KECCAKF_MAX_PAGES: usize = 2 * max_main_memory_pages_for_contiguous_range(200);
const KECCAK_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize = if XORIN_MAX_PAGES > KECCAKF_MAX_PAGES {
    XORIN_MAX_PAGES
} else {
    KECCAKF_MAX_PAGES
};

/// keccak-f\[1600\]: read 200 bytes via `buffer_ptr_reg`, permute in place.
#[derive(Debug, Clone)]
pub struct KeccakfInstr {
    pub buffer_ptr_reg: Variable,
    /// KeccakfPerm chip (24 rows per instruction).
    pub perm_chip_idx: Option<AirIndex>,
}

impl ExtInstr for KeccakfInstr {
    fn opname(&self) -> &str {
        "keccakf"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf = ctx.read_var(self.buffer_ptr_reg);
        ctx.emit_call("rvr_ext_keccakf", &["state", &buf]);
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        fixed_trace_rows_for_chip(self.perm_chip_idx, KECCAK_NUM_ROUNDS)
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// XORIN: XOR `len_reg` bytes from `input_ptr_reg` into `buffer_ptr_reg` in place.
#[derive(Debug, Clone)]
pub struct XorinInstr {
    pub buffer_ptr_reg: Variable,
    pub input_ptr_reg: Variable,
    pub len_reg: Variable,
}

impl ExtInstr for XorinInstr {
    fn opname(&self) -> &str {
        "xorin"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf_ptr = ctx.read_var(self.buffer_ptr_reg);
        let input = ctx.read_var(self.input_ptr_reg);
        let len = ctx.read_var(self.len_reg);
        ctx.emit_checked_call("rvr_ext_xorin", &["state", &buf_ptr, &input, &len]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// Keccak-256 extension. Register with the `ExtensionRegistry`.
pub struct KeccakExtension {
    keccakf_perm_chip_idx: Option<AirIndex>,
}

impl KeccakExtension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        opcode_air_idx(ctx, XorinOpcode::XORIN)?;
        let keccakf_op_chip_idx = opcode_air_idx(ctx, KeccakfOpcode::KECCAKF)?;
        // KeccakfPerm is registered adjacent to KeccakfOp and assigned the next
        // AIR index (keccakf_op_chip_idx + 1) due to reverse registration order.
        let keccakf_perm_chip_idx = keccakf_op_chip_idx.map(AirIndex::next);

        Ok(Self {
            keccakf_perm_chip_idx,
        })
    }
}

impl RvrExtension for KeccakExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == KeccakfOpcode::KECCAKF.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(KeccakfInstr {
                    buffer_ptr_reg,
                    perm_chip_idx: self.keccakf_perm_chip_idx,
                }),
                source_loc: None,
            }));
        }

        if opcode == XorinOpcode::XORIN.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            let input_ptr_reg = decode_reg(insn.b);
            let len_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(XorinInstr {
                    buffer_ptr_reg,
                    input_ptr_reg,
                    len_reg,
                }),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_keccak.h", include_str!("../c/rvr_ext_keccak.h"))]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_keccak.c", include_str!("../c/rvr_ext_keccak.c"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_keccak_ffi.a",
            include_bytes!(env!("RVR_KECCAK_FFI_STATICLIB")),
        )]
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        KECCAK_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}
