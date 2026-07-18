//! SHA-2 extension for rvr-openvm.
//!
//! Provides IR nodes for the SHA-256 and SHA-512 opcodes and the
//! `Sha2Extension` for lifting and executing them via double FFI.

use openvm_instructions::LocalOpcode;
use openvm_sha2_transpiler::Rv64Sha2Opcode;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_to_c, decode_reg, opcode_air_idx, AirIndex, ExtensionError, RvrExtension,
    RvrExtensionCtx, RvrInstruction,
};

/// IR node for a SHA-256 compress instruction.
///
/// Reads 32 bytes of state and 64 bytes of input block, applies SHA-256
/// compression, writes 32 bytes of new state to the destination pointer.
#[derive(Debug, Clone)]
pub struct Sha256Instr {
    /// Register index holding destination pointer (where new state is written).
    pub dst_ptr_reg: Reg,
    /// Register index holding state pointer (previous hash state).
    pub state_ptr_reg: Reg,
    /// Register index holding input pointer (message block).
    pub input_ptr_reg: Reg,
    /// AIR index of the SHA-256 main chip (1 row per instruction).
    pub main_chip_idx: Option<AirIndex>,
    /// AIR index of the SHA-256 block hasher chip (ROWS_PER_BLOCK rows per instruction).
    pub block_hasher_chip_idx: Option<AirIndex>,
}

impl ExtInstr for Sha256Instr {
    fn opname(&self) -> &str {
        "sha256"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let dst = ctx.read_reg(self.dst_ptr_reg);
        let st = ctx.read_reg(self.state_ptr_reg);
        let inp = ctx.read_reg(self.input_ptr_reg);
        let main = air_index_to_c(self.main_chip_idx);
        let block = air_index_to_c(self.block_hasher_chip_idx);
        let main = format!("{main}u");
        let block = format!("{block}u");
        ctx.emit_call("rvr_ext_sha256", &["state", &dst, &st, &inp, &main, &block]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for a SHA-512 compress instruction.
///
/// Reads 64 bytes of state and 128 bytes of input block, applies SHA-512
/// compression, writes 64 bytes of new state to the destination pointer.
#[derive(Debug, Clone)]
pub struct Sha512Instr {
    /// Register index holding destination pointer (where new state is written).
    pub dst_ptr_reg: Reg,
    /// Register index holding state pointer (previous hash state).
    pub state_ptr_reg: Reg,
    /// Register index holding input pointer (message block).
    pub input_ptr_reg: Reg,
    /// AIR index of the SHA-512 main chip (1 row per instruction).
    pub main_chip_idx: Option<AirIndex>,
    /// AIR index of the SHA-512 block hasher chip (ROWS_PER_BLOCK rows per instruction).
    pub block_hasher_chip_idx: Option<AirIndex>,
}

impl ExtInstr for Sha512Instr {
    fn opname(&self) -> &str {
        "sha512"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let dst = ctx.read_reg(self.dst_ptr_reg);
        let st = ctx.read_reg(self.state_ptr_reg);
        let inp = ctx.read_reg(self.input_ptr_reg);
        let main = air_index_to_c(self.main_chip_idx);
        let block = air_index_to_c(self.block_hasher_chip_idx);
        let main = format!("{main}u");
        let block = format!("{block}u");
        ctx.emit_call("rvr_ext_sha512", &["state", &dst, &st, &inp, &main, &block]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// The SHA-2 extension (SHA-256 + SHA-512 opcodes).
/// Register this with the `ExtensionRegistry`.
pub struct Sha2Extension {
    sha256_main_chip_idx: Option<AirIndex>,
    sha256_block_hasher_chip_idx: Option<AirIndex>,
    sha512_main_chip_idx: Option<AirIndex>,
    sha512_block_hasher_chip_idx: Option<AirIndex>,
}

impl Sha2Extension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        let sha256_main_chip_idx = opcode_air_idx(ctx, Rv64Sha2Opcode::SHA256)?;
        // The SHA-256 block hasher is registered adjacent to the main chip and
        // assigned the next AIR index (main_air_idx + 1) due to reverse registration order.
        let sha256_block_hasher_chip_idx = sha256_main_chip_idx.map(AirIndex::next);

        let sha512_main_chip_idx = opcode_air_idx(ctx, Rv64Sha2Opcode::SHA512)?;
        let sha512_block_hasher_chip_idx = sha512_main_chip_idx.map(AirIndex::next);

        Ok(Self {
            sha256_main_chip_idx,
            sha256_block_hasher_chip_idx,
            sha512_main_chip_idx,
            sha512_block_hasher_chip_idx,
        })
    }
}

impl RvrExtension for Sha2Extension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == Rv64Sha2Opcode::SHA256.global_opcode_usize() {
            let dst_ptr_reg = decode_reg(insn.a);
            let state_ptr_reg = decode_reg(insn.b);
            let input_ptr_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(Sha256Instr {
                    dst_ptr_reg,
                    state_ptr_reg,
                    input_ptr_reg,
                    main_chip_idx: self.sha256_main_chip_idx,
                    block_hasher_chip_idx: self.sha256_block_hasher_chip_idx,
                })),
                source_loc: None,
            }));
        }

        if opcode == Rv64Sha2Opcode::SHA512.global_opcode_usize() {
            let dst_ptr_reg = decode_reg(insn.a);
            let state_ptr_reg = decode_reg(insn.b);
            let input_ptr_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(Sha512Instr {
                    dst_ptr_reg,
                    state_ptr_reg,
                    input_ptr_reg,
                    main_chip_idx: self.sha512_main_chip_idx,
                    block_hasher_chip_idx: self.sha512_block_hasher_chip_idx,
                })),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_sha2.h", include_str!("../c/rvr_ext_sha2.h"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_sha2_ffi.a",
            include_bytes!(env!("RVR_SHA2_FFI_STATICLIB")),
        )]
    }
}
