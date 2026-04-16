//! Keccak-256 extension: IR nodes for KECCAKF + XORIN and the
//! `KeccakExtension` that lifts and emits them.
//!
//! The keccak-f permutation runs in C against the keccak-ffi staticlib; the
//! `.c` shim is emitted alongside generated code so clang can inline the
//! tracer helpers across the call boundary.

use std::path::{Path, PathBuf};

use openvm_instructions::instruction::Instruction;
use openvm_instructions::LocalOpcode;
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{decode_reg, resolve_opcode_air_idx, RvrExtension, RvrExtensionCtx};

/// keccak-f[1600]: read 200 bytes via `buffer_ptr_reg`, permute in place.
#[derive(Debug, Clone)]
pub struct KeccakfInstr {
    pub buffer_ptr_reg: Reg,
    /// KeccakfOp chip (1 row per instruction).
    pub op_chip_idx: u32,
    /// KeccakfPerm chip (24 rows per instruction).
    pub perm_chip_idx: u32,
}

impl ExtInstr for KeccakfInstr {
    fn opname(&self) -> &str {
        "keccakf"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf = ctx.read_reg(self.buffer_ptr_reg);
        ctx.write_line(&format!(
            "rvr_ext_keccakf(state, {buf}, {}u, {}u);",
            self.op_chip_idx, self.perm_chip_idx
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// XORIN: XOR `len_reg` bytes from `input_ptr_reg` into `buffer_ptr_reg` in place.
#[derive(Debug, Clone)]
pub struct XorinInstr {
    pub buffer_ptr_reg: Reg,
    pub input_ptr_reg: Reg,
    pub len_reg: Reg,
    /// Xorin chip (1 row per instruction).
    pub chip_idx: u32,
}

impl ExtInstr for XorinInstr {
    fn opname(&self) -> &str {
        "xorin"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf_ptr = ctx.read_reg(self.buffer_ptr_reg);
        let input = ctx.read_reg(self.input_ptr_reg);
        let len = ctx.read_reg(self.len_reg);
        ctx.write_line(&format!(
            "rvr_ext_xorin(state, {buf_ptr}, {input}, {len}, {}u);",
            self.chip_idx
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// Keccak-256 extension. Register with the `ExtensionRegistry`.
pub struct KeccakExtension {
    xorin_chip_idx: u32,
    keccakf_op_chip_idx: u32,
    keccakf_perm_chip_idx: u32,
    /// Path to the keccak-ffi staticlib that exports `rvr_keccak_f1600`.
    asm_staticlib_path: PathBuf,
}

impl KeccakExtension {
    /// Pure-mode constructor; chip indices are unused (`trace_chip` is no-op).
    pub fn new_pure(asm_staticlib_path: PathBuf) -> Self {
        Self {
            xorin_chip_idx: u32::MAX,
            keccakf_op_chip_idx: u32::MAX,
            keccakf_perm_chip_idx: u32::MAX,
            asm_staticlib_path,
        }
    }

    /// Resolves chip indices from the VM config.
    pub fn new(ctx: &RvrExtensionCtx, asm_staticlib_path: PathBuf) -> Self {
        let xorin_chip_idx = resolve_opcode_air_idx(XorinOpcode::XORIN.global_opcode(), ctx);
        let keccakf_op_chip_idx = resolve_opcode_air_idx(KeccakfOpcode::KECCAKF.global_opcode(), ctx);
        // KeccakfPermAir is inserted right before KeccakfOpAir in
        // Keccak256Rv32::extend_circuit, and the chip indices are set in
        // reverse order.
        let keccakf_perm_chip_idx = keccakf_op_chip_idx + 1;

        Self {
            xorin_chip_idx,
            keccakf_op_chip_idx,
            keccakf_perm_chip_idx,
            asm_staticlib_path,
        }
    }
}

impl<F: PrimeField32> RvrExtension<F> for KeccakExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == KeccakfOpcode::KECCAKF.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(KeccakfInstr {
                    buffer_ptr_reg,
                    op_chip_idx: self.keccakf_op_chip_idx,
                    perm_chip_idx: self.keccakf_perm_chip_idx,
                })),
                source_loc: None,
            }));
        }

        if opcode == XorinOpcode::XORIN.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            let input_ptr_reg = decode_reg(insn.b);
            let len_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(XorinInstr {
                    buffer_ptr_reg,
                    input_ptr_reg,
                    len_reg,
                    chip_idx: self.xorin_chip_idx,
                })),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_keccak.h", include_str!("../c/rvr_ext_keccak.h"))]
    }

    fn c_sources(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_keccak.c", include_str!("../c/rvr_ext_keccak.c"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.asm_staticlib_path
    }
}
