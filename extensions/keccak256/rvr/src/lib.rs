//! Keccak-256 extension: IR nodes for KECCAKF + XORIN and the
//! `KeccakExtension` that lifts and emits them.
//!
//! The keccak-f permutation runs in C against the keccak-ffi staticlib; the
//! `.c` shim is emitted alongside generated code so clang can inline the
//! tracer helpers across the call boundary.

use openvm_instructions::LocalOpcode;
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_to_c, decode_reg, opcode_air_idx, AirIndex, ExtensionError, RvrExtension,
    RvrExtensionCtx, RvrInstruction,
};

/// keccak-f\[1600\]: read 200 bytes via `buffer_ptr_reg`, permute in place.
#[derive(Debug, Clone)]
pub struct KeccakfInstr {
    pub buffer_ptr_reg: Reg,
    /// KeccakfOp chip (1 row per instruction).
    pub op_chip_idx: Option<AirIndex>,
    /// KeccakfPerm chip (24 rows per instruction).
    pub perm_chip_idx: Option<AirIndex>,
}

impl ExtInstr for KeccakfInstr {
    fn opname(&self) -> &str {
        "keccakf"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf = ctx.read_reg(self.buffer_ptr_reg);
        let op = air_index_to_c(self.op_chip_idx);
        let perm = air_index_to_c(self.perm_chip_idx);
        let op = format!("{op}u");
        let perm = format!("{perm}u");
        ctx.emit_call("rvr_ext_keccakf", &["state", &buf, &op, &perm]);
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
    pub chip_idx: Option<AirIndex>,
}

impl ExtInstr for XorinInstr {
    fn opname(&self) -> &str {
        "xorin"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let buf_ptr = ctx.read_reg(self.buffer_ptr_reg);
        let input = ctx.read_reg(self.input_ptr_reg);
        let len = ctx.read_reg(self.len_reg);
        let chip = air_index_to_c(self.chip_idx);
        let chip = format!("{chip}u");
        ctx.emit_call("rvr_ext_xorin", &["state", &buf_ptr, &input, &len, &chip]);
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
    xorin_chip_idx: Option<AirIndex>,
    keccakf_op_chip_idx: Option<AirIndex>,
    keccakf_perm_chip_idx: Option<AirIndex>,
}

impl KeccakExtension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        let xorin_chip_idx = opcode_air_idx(ctx, XorinOpcode::XORIN)?;
        let keccakf_op_chip_idx = opcode_air_idx(ctx, KeccakfOpcode::KECCAKF)?;
        // KeccakfPerm is registered adjacent to KeccakfOp and assigned the next
        // AIR index (keccakf_op_chip_idx + 1) due to reverse registration order.
        let keccakf_perm_chip_idx = keccakf_op_chip_idx.map(AirIndex::next);

        Ok(Self {
            xorin_chip_idx,
            keccakf_op_chip_idx,
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
}
