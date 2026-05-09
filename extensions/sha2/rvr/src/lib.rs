//! SHA-2 extension for rvr-openvm.
//!
//! Provides IR nodes for the SHA-256 and SHA-512 opcodes and the
//! `Sha2Extension` for lifting and executing them via double FFI.

use std::path::{Path, PathBuf};

use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{decode_reg, ExtensionError, RvrExtension, RvrExtensionCtx, NO_CHIP};

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
    /// SHA-256 main chip AIR index (1 row per instruction).
    pub main_chip_idx: u32,
    /// SHA-256 block hasher chip AIR index (ROWS_PER_BLOCK rows per instruction).
    pub block_hasher_chip_idx: u32,
}

impl ExtInstr for Sha256Instr {
    fn opname(&self) -> &str {
        "sha256"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let dst = ctx.read_reg(self.dst_ptr_reg);
        let st = ctx.read_reg(self.state_ptr_reg);
        let inp = ctx.read_reg(self.input_ptr_reg);
        ctx.write_line(&format!(
            "rvr_ext_sha256(state, {dst}, {st}, {inp}, {}u, {}u);",
            self.main_chip_idx, self.block_hasher_chip_idx
        ));
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
    /// SHA-512 main chip AIR index (1 row per instruction).
    pub main_chip_idx: u32,
    /// SHA-512 block hasher chip AIR index (ROWS_PER_BLOCK rows per instruction).
    pub block_hasher_chip_idx: u32,
}

impl ExtInstr for Sha512Instr {
    fn opname(&self) -> &str {
        "sha512"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let dst = ctx.read_reg(self.dst_ptr_reg);
        let st = ctx.read_reg(self.state_ptr_reg);
        let inp = ctx.read_reg(self.input_ptr_reg);
        ctx.write_line(&format!(
            "rvr_ext_sha512(state, {dst}, {st}, {inp}, {}u, {}u);",
            self.main_chip_idx, self.block_hasher_chip_idx
        ));
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
    sha256_main_chip_idx: u32,
    sha256_block_hasher_chip_idx: u32,
    sha512_main_chip_idx: u32,
    sha512_block_hasher_chip_idx: u32,
    staticlib_path: PathBuf,
}

impl Sha2Extension {
    /// Create a `Sha2Extension` for pure execution where chip indices
    /// don't matter (trace_chip is a no-op in pure mode).
    pub fn new_pure() -> Self {
        Self {
            sha256_main_chip_idx: u32::MAX,
            sha256_block_hasher_chip_idx: u32::MAX,
            sha512_main_chip_idx: u32::MAX,
            sha512_block_hasher_chip_idx: u32::MAX,
            staticlib_path: default_staticlib_path(),
        }
    }

    /// Create a new `Sha2Extension`, resolving chip indices from the VM config.
    pub fn new(ctx: &RvrExtensionCtx) -> Result<Self, ExtensionError> {
        // SHA-256 main chip AIR index
        let sha256_main_chip_idx =
            ctx.require_opcode_air_idx(Rv32Sha2Opcode::SHA256.global_opcode())?;

        // SHA-256 block hasher: in extend_circuit, the block hasher is added right before
        // the main chip. Due to reverse ordering of AIR indices,
        // block_hasher_air_idx = main_air_idx + 1. Stay NO_CHIP-safe in case
        // pure execution sneaks a NO_CHIP `main_chip_idx` through here.
        let sha256_block_hasher_chip_idx = if sha256_main_chip_idx == NO_CHIP {
            NO_CHIP
        } else {
            sha256_main_chip_idx + 1
        };

        // SHA-512 main chip AIR index
        let sha512_main_chip_idx =
            ctx.require_opcode_air_idx(Rv32Sha2Opcode::SHA512.global_opcode())?;

        // SHA-512 block hasher: same pattern as SHA-256.
        let sha512_block_hasher_chip_idx = if sha512_main_chip_idx == NO_CHIP {
            NO_CHIP
        } else {
            sha512_main_chip_idx + 1
        };

        Ok(Self {
            sha256_main_chip_idx,
            sha256_block_hasher_chip_idx,
            sha512_main_chip_idx,
            sha512_block_hasher_chip_idx,
            staticlib_path: default_staticlib_path(),
        })
    }
}

fn default_staticlib_path() -> PathBuf {
    PathBuf::from(env!("RVR_SHA2_FFI_STATICLIB"))
}

impl<F: PrimeField32> RvrExtension<F> for Sha2Extension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == Rv32Sha2Opcode::SHA256.global_opcode_usize() {
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

        if opcode == Rv32Sha2Opcode::SHA512.global_opcode_usize() {
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

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![("rvr_ext_sha2.h", include_str!("../c/rvr_ext_sha2.h"))]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }
}
