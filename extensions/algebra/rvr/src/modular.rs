//! Modular arithmetic IR nodes, phantom hints, and the
//! [`ModularRvrExtension`] lifter.

use num_bigint::BigUint;
use openvm_algebra_transpiler::{ModularPhantom, Rv64ModularArithmeticOpcode};
use openvm_algebra_utils::{find_non_qr, NQR_RNG_SEED};
use openvm_instructions::{instruction::Instruction, LocalOpcode, SystemOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::{rngs::StdRng, SeedableRng};
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{helpers::decode_reg, RvrExtension};
use strum::EnumCount;

use crate::{detect_known_field, format_c_byte_array, ModOp};

include!(concat!(env!("OUT_DIR"), "/secp256k1_files.rs"));

/// Per-modulus info for the modular extension. Includes a precomputed non-QR
/// for the `HintNonQr` / `HintSqrt` phantoms.
struct ModulusInfo {
    modulus_bytes: Vec<u8>,
    non_qr_bytes: Vec<u8>,
    num_limbs: u32,
}

fn make_moduli(moduli: Vec<BigUint>) -> Vec<ModulusInfo> {
    // Use the same deterministic seed as the circuit-side `NonQrHintSubEx::new`
    // (single rng across the full modulus list), so rvr-emitted NQRs match
    // what the circuit would compute.
    let mut rng = StdRng::from_seed(NQR_RNG_SEED);
    moduli
        .into_iter()
        .map(|m| make_modulus_info(&m, &mut rng))
        .collect()
}

fn make_modulus_info(modulus: &BigUint, rng: &mut StdRng) -> ModulusInfo {
    let bytes = modulus.bits().div_ceil(8) as usize;
    assert!(
        bytes <= 48,
        "modulus exceeds maximum supported size of 384 bits"
    );
    let num_limbs = if bytes <= 32 { 32u32 } else { 48u32 };
    let mut modulus_bytes = modulus.to_bytes_le();
    modulus_bytes.resize(num_limbs as usize, 0);
    let non_qr = find_non_qr(modulus, rng);
    let mut non_qr_bytes = non_qr.to_bytes_le();
    non_qr_bytes.resize(num_limbs as usize, 0);
    ModulusInfo {
        modulus_bytes,
        non_qr_bytes,
        num_limbs,
    }
}

// ── Modular arithmetic IR ────────────────────────────────────────────────────

/// IR node for modular arithmetic (ADD, SUB, MUL, DIV).
#[derive(Debug, Clone)]
pub struct ModArithInstr {
    pub op: ModOp,
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl ExtInstr for ModArithInstr {
    fn opname(&self) -> &str {
        "mod_arith"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        // Match Rv64VecHeapAdapter's memory-bus order: rs1, rs2, then rd.
        let rd = ctx.read_reg(self.rd_reg);
        let op_name = self.op.c_name();
        if let Some(field) = detect_known_field(&self.modulus) {
            let suffix = field.c_suffix();
            let name = format!("rvr_ext_mod_{op_name}_{suffix}");
            ctx.extern_call(&name, &["state", &rd, &rs1, &rs2]);
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            let name = format!("rvr_ext_mod_{op_name}");
            let num_limbs = format!("{}u", self.num_limbs);
            ctx.extern_call(&name, &["state", &rd, &rs1, &rs2, &num_limbs, "mod_"]);
            ctx.write_line("}");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for modular IS_EQ.
#[derive(Debug, Clone)]
pub struct ModIsEqInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
}

impl ExtInstr for ModIsEqInstr {
    fn opname(&self) -> &str {
        "mod_iseq"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        if let Some(field) = detect_known_field(&self.modulus) {
            let suffix = field.c_suffix();
            let name = format!("rvr_ext_mod_iseq_{suffix}");
            let val = ctx.extern_call_expr("uint32_t", &name, &["state", &rs1, &rs2]);
            ctx.write_reg(self.rd_reg, &val);
        } else {
            let mod_literal = format_c_byte_array(&self.modulus);
            ctx.write_line("{");
            ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
            let num_limbs = format!("{}u", self.num_limbs);
            let val = ctx.extern_call_expr(
                "uint32_t",
                "rvr_ext_mod_iseq",
                &["state", &rs1, &rs2, &num_limbs, "mod_"],
            );
            ctx.write_reg(self.rd_reg, &val);
            ctx.write_line("}");
        }
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for modular SETUP (SETUP_ADDSUB, SETUP_MULDIV, SETUP_ISEQ).
#[derive(Debug, Clone)]
pub struct ModSetupInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
}

/// IR node for modular SETUP_ISEQ, whose circuit writes the comparison
/// result to `rd` instead of treating `rd` as a heap pointer.
#[derive(Debug, Clone)]
pub struct ModIsEqSetupInstr {
    pub rd_reg: Reg,
    pub rs1_reg: Reg,
    pub rs2_reg: Reg,
    pub num_limbs: u32,
}

impl ExtInstr for ModIsEqSetupInstr {
    fn opname(&self) -> &str {
        "mod_iseq_setup"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        let num_limbs = format!("{}u", self.num_limbs);
        let val = ctx.extern_call_expr(
            "uint32_t",
            "rvr_ext_mod_setup_iseq",
            &["state", &rs1, &rs2, &num_limbs],
        );
        ctx.write_reg(self.rd_reg, &val);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

impl ExtInstr for ModSetupInstr {
    fn opname(&self) -> &str {
        "mod_setup"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let rs1 = ctx.read_reg(self.rs1_reg);
        let rs2 = ctx.read_reg(self.rs2_reg);
        // Match Rv64VecHeapAdapter's memory-bus order: rs1, rs2, then rd.
        let rd = ctx.read_reg(self.rd_reg);
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.extern_call("rvr_ext_mod_setup", &["state", &rd, &rs1, &rs2, &num_limbs]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

// ── Phantom instructions (HintNonQr, HintSqrt) ──────────────────────────────

/// IR node for HintNonQr phantom instruction.
#[derive(Debug, Clone)]
pub struct HintNonQrInstr {
    pub non_qr_bytes: Vec<u8>,
}

impl ExtInstr for HintNonQrInstr {
    fn opname(&self) -> &str {
        "hint_nonqr"
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static const uint8_t nqr[] = {literal};"));
        let len = format!("{}u", self.non_qr_bytes.len());
        ctx.extern_call("ext_hint_stream_set", &["nqr", &len]);
        ctx.write_line("}");
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

/// IR node for HintSqrt phantom instruction.
#[derive(Debug, Clone)]
pub struct HintSqrtInstr {
    pub rs1_reg: Reg,
    pub num_limbs: u32,
    pub modulus: Vec<u8>,
    pub non_qr_bytes: Vec<u8>,
}

impl ExtInstr for HintSqrtInstr {
    fn opname(&self) -> &str {
        "hint_sqrt"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        // HintSqrt is a phantom: its `rs1` pointer read must NOT be traced. The
        // reference executor reads it via untraced `GuestMemory` and advances the
        // clock by a single `increment_timestamp()`, so no memory-bus interaction
        // exists for the PhantomAir to consume. Using the tracing `read_reg` here
        // logged an orphan register access, breaking MemoryBus balance (the read
        // sat unconsumed between two AddSub writes to the same register). Mirror
        // the sibling phantoms (PrintStr, HintRandom): untraced read + a bare tick.
        let rs1 = ctx.read_reg_raw(self.rs1_reg);
        let mod_literal = format_c_byte_array(&self.modulus);
        let nqr_literal = format_c_byte_array(&self.non_qr_bytes);
        ctx.write_line("{");
        ctx.write_line(&format!("static const uint8_t mod_[] = {mod_literal};"));
        ctx.write_line(&format!("static const uint8_t nqr[] = {nqr_literal};"));
        let num_limbs = format!("{}u", self.num_limbs);
        ctx.extern_call(
            "rvr_ext_algebra_hint_sqrt",
            &["state", &rs1, &num_limbs, "mod_", "nqr"],
        );
        ctx.write_line("}");
        ctx.trace_timestamp();
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn is_block_end(&self) -> bool {
        false
    }
}

// ── Modular extension ────────────────────────────────────────────────────────

/// Modular arithmetic + phantom hints. Owns the modular Rust FFI staticlib
/// (built at repo build time) and registers `rvr_ext_modular.c` plus its
/// libsecp256k1 inputs for lift-time compilation. Independent of
/// [`crate::Fp2RvrExtension`].
pub struct ModularRvrExtension {
    moduli: Vec<ModulusInfo>,
}

impl ModularRvrExtension {
    pub fn new(moduli: Vec<BigUint>) -> Self {
        Self {
            moduli: make_moduli(moduli),
        }
    }
}

impl<F: PrimeField32> RvrExtension<F> for ModularRvrExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if let Some(lifted) = self.try_lift_modular(insn, pc, opcode) {
            return Some(lifted);
        }

        if opcode == SystemOpcode::PHANTOM.global_opcode_usize() {
            if let Some(lifted) = self.try_lift_phantom(insn, pc) {
                return Some(lifted);
            }
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_mod.h", include_str!("../c/rvr_ext_mod.h"))]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rvr_ext_modular.c",
            include_str!("../ffi/modular/c/rvr_ext_modular.c"),
        )]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_algebra_modular_ffi.a",
            include_bytes!(env!("RVR_ALGEBRA_MODULAR_FFI_STATICLIB")),
        )]
    }

    fn extra_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "precomputed_ecmult.c",
                include_str!("../ffi/modular/secp256k1/src/precomputed_ecmult.c"),
            ),
            (
                "precomputed_ecmult_gen.c",
                include_str!("../ffi/modular/secp256k1/src/precomputed_ecmult_gen.c"),
            ),
        ]
    }

    fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        SECP256K1_C_FILES.to_vec()
    }

    fn extra_cflags(&self) -> Vec<String> {
        vec![
            "-Isecp256k1/src".to_string(),
            "-Isecp256k1".to_string(),
            // ENABLE_MODULE_RECOVERY keeps the ECC modules compiled in so the
            // k256 EC ops in rvr_ext_modular.c can call into libsecp256k1.
            // (-DSECP256K1_BUILD is not set here — secp256k1.c defines it
            // internally.)
            "-DENABLE_MODULE_RECOVERY".to_string(),
            "-Wno-unused-function".to_string(),
            "-Wno-unused-parameter".to_string(),
            "-Wno-unused-variable".to_string(),
            "-Wno-strict-prototypes".to_string(),
        ]
    }
}

impl ModularRvrExtension {
    fn try_lift_modular<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u64,
        opcode: usize,
    ) -> Option<LiftedInstr> {
        let base_offset = Rv64ModularArithmeticOpcode::CLASS_OFFSET;
        let count = Rv64ModularArithmeticOpcode::COUNT;

        if opcode < base_offset {
            return None;
        }
        let relative = opcode - base_offset;
        let mod_idx = relative / count;
        let local = relative % count;

        if mod_idx >= self.moduli.len() {
            return None;
        }

        let info = &self.moduli[mod_idx];
        let rd_reg = decode_reg(insn.a);
        let rs1_reg = decode_reg(insn.b);
        let rs2_reg = decode_reg(insn.c);

        let instr: Instr = match local {
            x if x == Rv64ModularArithmeticOpcode::ADD as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Add,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::SUB as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Sub,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::SETUP_ADDSUB as usize => {
                Instr::Ext(Box::new(ModSetupInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::MUL as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Mul,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::DIV as usize => {
                Instr::Ext(Box::new(ModArithInstr {
                    op: ModOp::Div,
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::SETUP_MULDIV as usize => {
                Instr::Ext(Box::new(ModSetupInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::IS_EQ as usize => {
                Instr::Ext(Box::new(ModIsEqInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                    modulus: info.modulus_bytes.clone(),
                }))
            }
            x if x == Rv64ModularArithmeticOpcode::SETUP_ISEQ as usize => {
                Instr::Ext(Box::new(ModIsEqSetupInstr {
                    rd_reg,
                    rs1_reg,
                    rs2_reg,
                    num_limbs: info.num_limbs,
                }))
            }
            _ => return None,
        };

        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr,
            source_loc: None,
        }))
    }

    fn try_lift_phantom<F: PrimeField32>(
        &self,
        insn: &Instruction<F>,
        pc: u64,
    ) -> Option<LiftedInstr> {
        let c_val = insn.c.as_canonical_u32();
        let discriminant = (c_val & 0xffff) as u16;
        let mod_idx = (c_val >> 16) as usize;

        match ModularPhantom::from_repr(discriminant) {
            Some(ModularPhantom::HintNonQr) => {
                let info = self.moduli.get(mod_idx)?;
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Instr::Ext(Box::new(HintNonQrInstr {
                        non_qr_bytes: info.non_qr_bytes.clone(),
                    })),
                    source_loc: None,
                }))
            }
            Some(ModularPhantom::HintSqrt) => {
                let info = self.moduli.get(mod_idx)?;
                let rs1_reg = decode_reg(insn.a);
                Some(LiftedInstr::Body(InstrAt {
                    pc,
                    instr: Instr::Ext(Box::new(HintSqrtInstr {
                        rs1_reg,
                        num_limbs: info.num_limbs,
                        modulus: info.modulus_bytes.clone(),
                        non_qr_bytes: info.non_qr_bytes.clone(),
                    })),
                    source_loc: None,
                }))
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal capturing [`ExtEmitCtx`]. A *traced* register read (`read_reg`)
    /// records a `trace_reg_read(...)` line — in generated C this emits a
    /// MemoryBus event; an *untraced* read (`read_reg_raw`) records nothing.
    /// Tests can therefore assert whether a phantom logged a bus-visible
    /// register access.
    #[derive(Default)]
    struct TestEmitCtx {
        lines: Vec<String>,
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn read_reg(&mut self, idx: u8) -> String {
            self.lines.push(format!("trace_reg_read(state, {idx});"));
            format!("r{idx}")
        }
        fn read_reg_raw(&mut self, idx: u8) -> String {
            format!("r{idx}")
        }
        fn write_reg(&mut self, _idx: u8, _val: &str) {}
        fn write_reg_raw(&mut self, _idx: u8, _val: &str) {}
        fn write_line(&mut self, s: &str) {
            self.lines.push(s.to_string());
        }
        fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!(
                "uint32_t {tmp} = read_mem({base}, {offset}, {width}, {signed});"
            ));
            tmp
        }
        fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
            self.write_line(&format!("write_mem({base}, {offset}, {val}, {width});"));
        }
        fn extern_call(&mut self, name: &str, args: &[&str]) {
            self.write_line(&format!("{name}({});", args.join(", ")));
        }
        fn extern_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!("{ret_ty} {tmp} = {name}({});", args.join(", ")));
            tmp
        }
        fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
            self.write_line(&format!("trace_chip(state, {chip_idx}u, {count_expr});"));
        }
        fn trace_mem_access(&mut self, addr: &str, addr_space: u32) {
            self.write_line(&format!("trace_mem_access(state, {addr}, {addr_space}u);"));
        }
        fn trace_mem_access_u64_range(
            &mut self,
            base_addr: &str,
            num_dwords: &str,
            addr_space: u32,
        ) {
            self.write_line(&format!(
                "trace_mem_access_u64_range(state, {base_addr}, {num_dwords}, {addr_space}u);"
            ));
        }
        fn trace_wr_as_u64(&mut self, addr: &str, val: &str, addr_space: u32) {
            self.write_line(&format!(
                "trace_wr_as_u64(state, {addr}, {val}, {addr_space}u);"
            ));
        }
        fn trace_timestamp(&mut self) {
            self.write_line("trace_timestamp(state);");
        }
    }

    /// Regression guard for bug #2 (reth CPU seg-28 MemoryBus imbalance).
    ///
    /// `HintSqrt` is a phantom: the reference executor reads its pointer register
    /// through untraced `GuestMemory` and advances the clock with a single
    /// `increment_timestamp()`, so no MemoryBus interaction exists for the
    /// memory-unconstrained `PhantomAir` to consume. Emitting a *traced* read
    /// (`read_reg`) here logged an orphan register access that sat unconsumed
    /// between two AddSub writes to the same register, leaving bus 1 one-sided
    /// (+1 send at T, -1 receive at T+gap). The phantom must instead read
    /// untraced and emit one bare timestamp tick, exactly like every other
    /// phantom (PrintStr, HintRandom, HintFinalExp).
    #[test]
    fn hint_sqrt_phantom_reads_pointer_untraced_and_ticks_once() {
        let mut ctx = TestEmitCtx::default();
        HintSqrtInstr {
            rs1_reg: 10,
            num_limbs: 4,
            modulus: vec![0u8; 32],
            non_qr_bytes: vec![0u8; 32],
        }
        .emit_c(&mut ctx);

        // A phantom must emit no traced (bus-visible) register access.
        assert!(
            !ctx.lines.iter().any(|l| l.contains("trace_reg_read")),
            "HintSqrt emitted a traced register read (orphan MemoryBus access); \
             phantoms must use read_reg_raw. lines: {:#?}",
            ctx.lines
        );
        // The pointer still reaches the host callback via the untraced value r10.
        assert!(
            ctx.lines
                .iter()
                .any(|l| l.contains("rvr_ext_algebra_hint_sqrt(state, r10,")),
            "expected untraced pointer r10 passed to hint_sqrt callback. lines: {:#?}",
            ctx.lines
        );
        // The phantom still advances the clock by exactly one bare tick, matching
        // the reference `increment_timestamp()` so downstream timestamps are
        // byte-identical.
        assert_eq!(
            ctx.lines
                .iter()
                .filter(|l| l.as_str() == "trace_timestamp(state);")
                .count(),
            1,
            "HintSqrt must emit exactly one bare timestamp tick. lines: {:#?}",
            ctx.lines
        );
    }
}
