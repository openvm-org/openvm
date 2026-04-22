//! Deferral extension for rvr-openvm.
//!
//! Provides IR nodes for the two deferral opcodes (CALL and OUTPUT) and the
//! `DeferralRvrExtension` for lifting and executing them via double FFI.
//!
//! Pre-computed deferral results are produced by [`precompute`] and stored in
//! `OpenVmIoState.deferral` before execution.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use openvm_circuit::arch::VmField;
use openvm_deferral_circuit::{
    generate_deferral_results, poseidon2::deferral_poseidon2_chip, RawDeferralResult,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_ffi_common::{DEFERRAL_COMMIT_NUM_BYTES, DEFERRAL_OUTPUT_KEY_BYTES};
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{decode_reg, ExtensionError, RvrExtension, RvrExtensionCtx};

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

// ── Pre-computed data ─────────────────────────────────────────────────────────

/// Pre-computed deferral results, matching the layout in `OpenVmIoState.deferral`.
pub struct DeferralPrecomputedData {
    /// (def_idx, input_commit) → output_key
    pub call_entries:
        HashMap<(u32, [u8; DEFERRAL_COMMIT_NUM_BYTES]), [u8; DEFERRAL_OUTPUT_KEY_BYTES]>,
    /// output_commit → output_raw
    pub output_entries: HashMap<[u8; DEFERRAL_COMMIT_NUM_BYTES], Vec<u8>>,
}

/// Deferral function type: maps raw input bytes to raw output bytes.
pub type DeferralFnBox = Box<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

/// Per-circuit deferral inputs for pre-computation.
pub struct DeferralCircuitInputs {
    /// The deferral function: maps raw input to raw output.
    pub func: DeferralFnBox,
    /// Known (input_commit, input_raw) pairs.
    pub inputs: Vec<(Vec<u8>, Vec<u8>)>,
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
    pub fn new_pure(staticlib_path: PathBuf) -> Self {
        Self {
            call_chip_idx: u32::MAX,
            output_chip_idx: u32::MAX,
            poseidon2_chip_idx: u32::MAX,
            staticlib_path,
        }
    }

    /// Create with chip indices resolved from the VM config.
    pub fn new<F: VmField>(
        ctx: &RvrExtensionCtx,
        staticlib_path: PathBuf,
    ) -> Result<Self, ExtensionError> {
        let call_chip_idx = ctx.require_opcode_air_idx(DeferralOpcode::CALL.global_opcode())?;
        let output_chip_idx = ctx.require_opcode_air_idx(DeferralOpcode::OUTPUT.global_opcode())?;
        // Poseidon2 periphery chip: in extend_circuit, the hasher is added
        // right before the CALL chip. Due to reverse ordering of AIR indices,
        // poseidon2_air_idx = call_air_idx + 1.
        let poseidon2_chip_idx = call_chip_idx + 1;

        Ok(Self {
            call_chip_idx,
            output_chip_idx,
            poseidon2_chip_idx,
            staticlib_path,
        })
    }
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

// ── Pre-computation ───────────────────────────────────────────────────────────

pub fn precompute<F: VmField>(circuit_inputs: &[DeferralCircuitInputs]) -> DeferralPrecomputedData {
    let hasher = deferral_poseidon2_chip::<F>();

    let mut call_entries = HashMap::new();
    let mut output_entries = HashMap::new();

    for (def_idx, ci) in circuit_inputs.iter().enumerate() {
        let raw_results: Vec<RawDeferralResult> = ci
            .inputs
            .iter()
            .map(|(input_commit, input_raw)| {
                let output_raw = (ci.func)(input_raw);
                RawDeferralResult::new(input_commit.clone(), output_raw)
            })
            .collect();

        let results = generate_deferral_results::<F>(raw_results, def_idx as u32, &hasher);

        for result in &results {
            let input_commit: [u8; DEFERRAL_COMMIT_NUM_BYTES] =
                result.input.as_slice().try_into().unwrap();
            let output_commit: [u8; DEFERRAL_COMMIT_NUM_BYTES] =
                result.output_commit.as_slice().try_into().unwrap();
            let output_len = result.output_raw.len() as u64;

            let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
            output_key[..DEFERRAL_COMMIT_NUM_BYTES].copy_from_slice(&output_commit);
            output_key[DEFERRAL_COMMIT_NUM_BYTES..].copy_from_slice(&output_len.to_le_bytes());

            call_entries.insert((def_idx as u32, input_commit), output_key);
            output_entries.insert(output_commit, result.output_raw.clone());
        }
    }

    DeferralPrecomputedData {
        call_entries,
        output_entries,
    }
}
