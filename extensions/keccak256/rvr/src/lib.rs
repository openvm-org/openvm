//! Keccak-256 extension: IR nodes for KECCAKF + XORIN and the
//! `KeccakExtension` that lifts and emits them.
//!
//! The keccak-f permutation runs in C against the keccak-ffi staticlib; the
//! `.c` shim is emitted alongside generated code so clang can inline the
//! tracer helpers across the call boundary.

use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_keccak256_transpiler::{KeccakfOpcode, XorinOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, InlineRecordShape, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_codegen_fingerprint, air_index_to_c, decode_reg, opcode_air_idx, AirIndex,
    ExtensionError, RvrExtension, RvrExtensionCtx,
};

/// Byte size shared by the generated-C, host-arena, and CUDA KeccakF record
/// ABIs. The C and CUDA mirrors each carry a compile-time size assertion.
pub const KECCAKF_DIRECT_RECORD_SIZE: usize = 320;
pub const XORIN_DIRECT_RECORD_SIZE: usize = 656;

/// keccak-f\[1600\]: read 200 bytes via `buffer_ptr_reg`, permute in place.
#[derive(Debug, Clone)]
pub struct KeccakfInstr {
    pub from_pc: u32,
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
        let (buf, from_timestamp, rd_prev_timestamp) = ctx.read_reg_with_trace(self.buffer_ptr_reg);
        let from_pc = format!("{}u", self.from_pc);
        let rd_ptr = format!("{}u", u32::from(self.buffer_ptr_reg) * 8);
        let op = if ctx.inline_record_enabled() {
            air_index_to_c(self.op_chip_idx)
        } else {
            u32::MAX
        };
        let perm = air_index_to_c(self.perm_chip_idx);
        let op = format!("{op}u");
        let perm = format!("{perm}u");
        ctx.extern_call(
            "rvr_ext_keccakf",
            &[
                "state",
                &buf,
                &from_pc,
                &from_timestamp,
                &rd_ptr,
                &rd_prev_timestamp,
                &op,
                &perm,
            ],
        );
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom {
            record_size: KECCAKF_DIRECT_RECORD_SIZE,
        })
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
    pub from_pc: u32,
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
        let (buf_ptr, from_timestamp, buffer_prev_timestamp) =
            ctx.read_reg_with_trace(self.buffer_ptr_reg);
        let (input, _, input_prev_timestamp) = ctx.read_reg_with_trace(self.input_ptr_reg);
        let (len, _, len_prev_timestamp) = ctx.read_reg_with_trace(self.len_reg);
        let from_pc = format!("{}u", self.from_pc);
        let buffer_ptr_reg = format!("{}u", u32::from(self.buffer_ptr_reg) * 8);
        let input_ptr_reg = format!("{}u", u32::from(self.input_ptr_reg) * 8);
        let len_reg = format!("{}u", u32::from(self.len_reg) * 8);
        let chip = if ctx.inline_record_enabled() {
            air_index_to_c(self.chip_idx)
        } else {
            u32::MAX
        };
        let chip = format!("{chip}u");
        ctx.extern_call(
            "rvr_ext_xorin",
            &[
                "state",
                &buf_ptr,
                &input,
                &len,
                &from_pc,
                &from_timestamp,
                &buffer_ptr_reg,
                &input_ptr_reg,
                &len_reg,
                &buffer_prev_timestamp,
                &input_prev_timestamp,
                &len_prev_timestamp,
                &chip,
            ],
        );
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom {
            record_size: XORIN_DIRECT_RECORD_SIZE,
        })
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

impl<F: PrimeField32> RvrExtension<F> for KeccakExtension {
    fn codegen_fingerprint(&self) -> Option<Vec<u8>> {
        Some(air_index_codegen_fingerprint(
            b"openvm-keccak-rvr-v1",
            &[
                self.xorin_chip_idx,
                self.keccakf_op_chip_idx,
                self.keccakf_perm_chip_idx,
            ],
        ))
    }

    fn try_lift(&self, insn: &Instruction<F>, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == KeccakfOpcode::KECCAKF.global_opcode_usize() {
            let buffer_ptr_reg = decode_reg(insn.a);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(KeccakfInstr {
                    from_pc: pc as u32,
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
                    from_pc: pc as u32,
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
