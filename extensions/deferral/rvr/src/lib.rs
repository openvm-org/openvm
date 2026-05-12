//! Deferral extension for rvr-openvm: IR nodes for CALL/OUTPUT and the
//! `DeferralRvrExtension` for lifting them via double FFI.

use std::path::{Path, PathBuf};

use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{decode_reg, ExtensionError, RvrExtension, RvrExtensionCtx, NO_CHIP};

// ── IR Nodes ──────────────────────────────────────────────────────────────────

/// IR node for a deferral CALL instruction.
#[derive(Debug, Clone)]
pub struct DeferralCallInstr {
    pub rd_reg: Reg,
    pub rs_reg: Reg,
    pub def_idx: u32,
    pub poseidon2_chip_idx: u32,
}

impl ExtInstr for DeferralCallInstr {
    fn opname(&self) -> &str {
        "def_call"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs = ctx.read_reg(self.rs_reg);
        ctx.write_line(&format!(
            "rvr_ext_deferral_call(state, {rd}, {rs}, {}u, {}u);",
            self.def_idx, self.poseidon2_chip_idx
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for a deferral OUTPUT instruction.
#[derive(Debug, Clone)]
pub struct DeferralOutputInstr {
    pub rd_reg: Reg,
    pub rs_reg: Reg,
    pub def_idx: u32,
    pub output_chip_idx: u32,
    pub poseidon2_chip_idx: u32,
}

impl ExtInstr for DeferralOutputInstr {
    fn opname(&self) -> &str {
        "def_output"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rd = ctx.read_reg(self.rd_reg);
        let rs = ctx.read_reg(self.rs_reg);
        ctx.write_line(&format!(
            "rvr_ext_deferral_output(state, {rd}, {rs}, {}u, {}u, {}u);",
            self.def_idx, self.output_chip_idx, self.poseidon2_chip_idx
        ));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

// ── Extension ─────────────────────────────────────────────────────────────────

/// The Deferral extension (CALL + OUTPUT opcodes).
pub struct DeferralRvrExtension {
    #[allow(dead_code)]
    call_chip_idx: u32,
    output_chip_idx: u32,
    poseidon2_chip_idx: u32,
    staticlib_path: PathBuf,
}

impl DeferralRvrExtension {
    /// Create for pure execution (chip indices don't matter).
    pub fn new_pure() -> Self {
        Self {
            call_chip_idx: NO_CHIP,
            output_chip_idx: NO_CHIP,
            poseidon2_chip_idx: NO_CHIP,
            staticlib_path: default_staticlib_path(),
        }
    }

    /// Create with chip indices resolved from the VM config.
    pub fn new(ctx: &RvrExtensionCtx) -> Result<Self, ExtensionError> {
        let call_chip_idx = ctx.require_opcode_air_idx(DeferralOpcode::CALL.global_opcode())?;
        let output_chip_idx = ctx.require_opcode_air_idx(DeferralOpcode::OUTPUT.global_opcode())?;
        // Poseidon2 periphery chip: in extend_circuit, the hasher is added
        // right before the CALL chip. Due to reverse ordering of AIR indices,
        // poseidon2_air_idx = call_air_idx + 1. Stay NO_CHIP-safe in case
        // pure execution sneaks a NO_CHIP `call_chip_idx` through here.
        let poseidon2_chip_idx = if call_chip_idx == NO_CHIP {
            NO_CHIP
        } else {
            call_chip_idx + 1
        };

        Ok(Self {
            call_chip_idx,
            output_chip_idx,
            poseidon2_chip_idx,
            staticlib_path: default_staticlib_path(),
        })
    }
}

fn default_staticlib_path() -> PathBuf {
    PathBuf::from(env!("RVR_DEFERRAL_FFI_STATICLIB"))
}

impl<F: PrimeField32> RvrExtension<F> for DeferralRvrExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == DeferralOpcode::CALL.global_opcode_usize() {
            let rd_reg = decode_reg(insn.a);
            let rs_reg = decode_reg(insn.b);
            let def_idx = insn.c.as_canonical_u32();
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(DeferralCallInstr {
                    rd_reg,
                    rs_reg,
                    def_idx,
                    poseidon2_chip_idx: self.poseidon2_chip_idx,
                })),
                source_loc: None,
            }));
        }

        if opcode == DeferralOpcode::OUTPUT.global_opcode_usize() {
            let rd_reg = decode_reg(insn.a);
            let rs_reg = decode_reg(insn.b);
            let def_idx = insn.c.as_canonical_u32();
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(DeferralOutputInstr {
                    rd_reg,
                    rs_reg,
                    def_idx,
                    output_chip_idx: self.output_chip_idx,
                    poseidon2_chip_idx: self.poseidon2_chip_idx,
                })),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&str, &str)> {
        vec![(
            "rvr_ext_deferral.h",
            include_str!("../c/rvr_ext_deferral.h"),
        )]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }
}
