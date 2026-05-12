//! Deferral extension for rvr-openvm: IR nodes for CALL/OUTPUT and the
//! `DeferralRvrExtension` for lifting them via double FFI.
//!
//! Also owns the host-side deferral runtime: thread-local storage for the
//! registered deferral closures and output hasher, populated by
//! `DeferralExtension::extend_rvr` and consumed during rvr execution by
//! `openvm-circuit`'s `host_deferral_call_lookup` (on cache miss).

use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::Arc,
};

use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_ffi_common::DEFERRAL_COMMIT_NUM_BYTES;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{decode_reg, ExtensionError, RvrExtension, RvrExtensionCtx, NO_CHIP};

// ── Host-side deferral runtime ────────────────────────────────────────────────

/// `input_raw → output_raw` closure registered by the host.
pub type DeferralFnPtr = Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

/// `(def_idx, output_raw) → output_commit` hasher registered by the host.
pub type DeferralHashFn = Arc<dyn Fn(u32, &[u8]) -> [u8; DEFERRAL_COMMIT_NUM_BYTES] + Send + Sync>;

#[derive(Default)]
struct DeferralRuntime {
    fns: Vec<DeferralFnPtr>,
    hash: Option<DeferralHashFn>,
}

thread_local! {
    static DEFERRAL_RUNTIME: RefCell<DeferralRuntime> = RefCell::new(DeferralRuntime::default());
}

/// Install the host-side deferral closures and output hasher into this
/// thread's TLS. Overwrites any prior installation. Called by
/// `DeferralExtension::extend_rvr` while the rvr `ExtensionRegistry` is being
/// built; the values stay live until the next install (or thread exit).
pub fn install_deferral_runtime(fns: Vec<DeferralFnPtr>, hash: DeferralHashFn) {
    DEFERRAL_RUNTIME.with(|r| {
        let mut r = r.borrow_mut();
        r.fns = fns;
        r.hash = Some(hash);
    });
}

/// Evaluate a registered deferral closure: returns `(output_commit, output_raw)`.
///
/// Called from openvm-circuit's `host_deferral_call_lookup` on cache miss
/// (when the cache holds `Raw(input_raw)` but no resolved output yet).
///
/// # Panics
///
/// Panics if `def_idx` has no registered closure, or no hasher was installed.
pub fn eval_deferral_call(
    def_idx: u32,
    input_raw: &[u8],
) -> ([u8; DEFERRAL_COMMIT_NUM_BYTES], Vec<u8>) {
    DEFERRAL_RUNTIME.with(|r| {
        let r = r.borrow();
        let func = r
            .fns
            .get(def_idx as usize)
            .expect("deferral CALL: def_idx has no closure registered");
        let hash = r
            .hash
            .as_ref()
            .expect("deferral CALL: hash_output not configured");
        let output_raw = func(input_raw);
        let output_commit = hash(def_idx, &output_raw);
        (output_commit, output_raw)
    })
}

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
