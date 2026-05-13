//! Deferral extension for rvr-openvm: IR nodes for CALL/OUTPUT and the
//! `DeferralRvrExtension` for lifting them via double FFI.
#![cfg(feature = "rvr")]

use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    sync::Arc,
};

use openvm_circuit::arch::{
    deferral::{DeferralFn, InputMapVal},
    rvr::io::OpenVmIoState,
};
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_ffi_common::{DEFERRAL_COMMIT_NUM_BYTES, DEFERRAL_OUTPUT_KEY_BYTES};
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{decode_reg, ExtensionError, RvrExtension, RvrExtensionCtx, NO_CHIP};

/// `(def_idx, output_raw) → output_commit` hasher registered by the host.
pub type DeferralHashFn = Arc<dyn Fn(u32, &[u8]) -> [u8; DEFERRAL_COMMIT_NUM_BYTES] + Send + Sync>;

pub struct DeferralCtx {
    pub fns: Vec<Arc<DeferralFn>>,
    pub hash: DeferralHashFn,
}

impl DeferralCtx {
    pub fn new(fns: Vec<Arc<DeferralFn>>, hash: DeferralHashFn) -> Self {
        Self { fns, hash }
    }
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
    deferral_ctx: DeferralCtx,
}

impl DeferralRvrExtension {
    /// Create for pure execution (chip indices don't matter).
    pub fn new_pure(fns: Vec<Arc<DeferralFn>>, hash: DeferralHashFn) -> Self {
        Self {
            call_chip_idx: NO_CHIP,
            output_chip_idx: NO_CHIP,
            poseidon2_chip_idx: NO_CHIP,
            staticlib_path: default_staticlib_path(),
            deferral_ctx: DeferralCtx::new(fns, hash),
        }
    }

    /// Create with chip indices resolved from the VM config.
    pub fn new(
        ctx: &RvrExtensionCtx,
        fns: Vec<Arc<DeferralFn>>,
        hash: DeferralHashFn,
    ) -> Result<Self, ExtensionError> {
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
            deferral_ctx: DeferralCtx::new(fns, hash),
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

    fn c_sources(&self) -> Vec<(&str, &str)> {
        vec![(
            "ext_deferral_lookup.c",
            include_str!("../c/ext_deferral_lookup.c"),
        )]
    }

    fn staticlib_path(&self) -> &Path {
        &self.staticlib_path
    }

    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        type RegisterFn = unsafe extern "C" fn(*const DeferralHostCallbacks);
        let register_fn: RegisterFn = unsafe {
            let sym = lib
                .get::<RegisterFn>(b"register_deferral_callbacks")
                .map_err(|e| ExtensionError::HostCallbackRegistration(e.to_string()))?;
            *sym
        };
        // `ctx` aliases `self.deferral_ctx`; the C side must not outlive `self`.
        let callbacks = DeferralHostCallbacks {
            ctx: &self.deferral_ctx as *const DeferralCtx as *mut c_void,
            call_lookup: host_deferral_call_lookup::<F>,
            output_lookup: host_deferral_output_lookup::<F>,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

// ── Host callbacks ──────────────────────────────────────────────────────────

/// Must match the C `DeferralHostCallbacks` layout in `ext_deferral_lookup.c`.
#[repr(C)]
pub struct DeferralHostCallbacks {
    pub ctx: *mut c_void,
    pub call_lookup: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8) -> i32,
    pub output_lookup:
        unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const u8, *mut u8, u32) -> i32,
}

/// Deferral CALL lookup. Returns 1 on hit, 0 on miss.
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a valid
/// `OpenVmIoState<'_, F>`.
/// `input_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_key_out` must point to `DEFERRAL_OUTPUT_KEY_BYTES` writable bytes.
pub unsafe extern "C" fn host_deferral_call_lookup<F: PrimeField32>(
    d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    input_commit_raw: *const u8,
    output_key_out: *mut u8,
) -> i32 {
    let dc = unsafe { &*(d_ctx as *const DeferralCtx) };
    let io = unsafe { &mut *(io_ctx as *mut OpenVmIoState<'_, F>) };

    let input_commit: Vec<u8> =
        unsafe { std::slice::from_raw_parts(input_commit_raw, DEFERRAL_COMMIT_NUM_BYTES).to_vec() };

    let Some(state) = io.deferrals.get_mut(def_idx as usize) else {
        return 0;
    };

    let (output_commit, output_len) = match state.get_input(&input_commit).clone() {
        InputMapVal::Output(commit) => {
            let len = state.get_output(&commit).len() as u64;
            let arr: [u8; DEFERRAL_COMMIT_NUM_BYTES] = commit.as_slice().try_into().unwrap();
            (arr, len)
        }
        InputMapVal::Raw(input_raw) => {
            let func = dc
                .fns
                .get(def_idx as usize)
                .expect("deferral CALL: def_idx has no closure registered");
            let output_raw = func.call_raw(&input_raw);
            let commit = (dc.hash)(def_idx, &output_raw);
            let len = output_raw.len() as u64;
            io.deferrals[def_idx as usize].store_output(&input_commit, commit.to_vec(), output_raw);
            (commit, len)
        }
    };

    let mut output_key = [0u8; DEFERRAL_OUTPUT_KEY_BYTES];
    output_key[..DEFERRAL_COMMIT_NUM_BYTES].copy_from_slice(&output_commit);
    output_key[DEFERRAL_COMMIT_NUM_BYTES..].copy_from_slice(&output_len.to_le_bytes());
    unsafe {
        std::ptr::copy_nonoverlapping(
            output_key.as_ptr(),
            output_key_out,
            DEFERRAL_OUTPUT_KEY_BYTES,
        );
    }
    1
}

/// Deferral OUTPUT lookup: `deferrals[def_idx].output_map[output_commit]`.
/// Returns 1 on hit, 0 on miss.
///
/// # Safety
///
/// `d_ctx` must point to a valid `DeferralCtx`. `io_ctx` must point to a
/// valid `OpenVmIoState<'_, F>`.
/// `output_commit_raw` must point to `DEFERRAL_COMMIT_NUM_BYTES` readable bytes.
/// `output_raw_out` must point to at least `expected_len` writable bytes.
pub unsafe extern "C" fn host_deferral_output_lookup<F: PrimeField32>(
    _d_ctx: *mut c_void,
    io_ctx: *mut c_void,
    def_idx: u32,
    output_commit_raw: *const u8,
    output_raw_out: *mut u8,
    expected_len: u32,
) -> i32 {
    let io = unsafe { &*(io_ctx as *const OpenVmIoState<'_, F>) };

    let output_commit: Vec<u8> = unsafe {
        std::slice::from_raw_parts(output_commit_raw, DEFERRAL_COMMIT_NUM_BYTES).to_vec()
    };
    let Some(state) = io.deferrals.get(def_idx as usize) else {
        return 0;
    };
    let raw = state.get_output(&output_commit);
    // TODO: change these panics to something better to handle across the FFI boundary.
    assert_eq!(raw.len(), expected_len as usize);
    unsafe { std::ptr::copy_nonoverlapping(raw.as_ptr(), output_raw_out, raw.len()) };
    1
}
