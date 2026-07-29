//! Compiler lowering a finalized [`FieldExpr`] into [`TracegenIr`]: validates CUDA
//! capabilities, emits the value and limb operation tapes, and derives the field,
//! Montgomery, and carry-bound constants.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use super::{abi::*, ConstraintMeta, EvalOp, EvalOpcode, TracegenIr, WitnessOp, WitnessOpcode};
use crate::{ExprBuilder, FieldExpr, SymbolicExpr};

fn biguint_to_u32s(x: &BigUint, k: usize) -> Vec<u32> {
    let mut v = x.to_u32_digits();
    assert!(v.len() <= k, "value too wide");
    v.resize(k, 0);
    v
}

struct TracegenCompiler<'a> {
    builder: &'a ExprBuilder,
    k: usize,
    eval_ops: Vec<EvalOp>,
    next_slot: usize,
    mont_payload: Vec<u32>,
    witness_ops: Vec<WitnessOp>,
    scratch_top: u32,
    const_limbs_payload: Vec<i32>,
    r: BigUint, // 2^(32K) mod p
}

impl<'a> TracegenCompiler<'a> {
    fn mont(&self, x: &BigUint) -> Vec<u32> {
        biguint_to_u32s(&((x * &self.r) % &self.builder.prime), self.k)
    }

    fn push_mont_payload(&mut self, x: &BigUint) -> u32 {
        let idx = (self.mont_payload.len() / self.k) as u32;
        let limbs = self.mont(x);
        self.mont_payload.extend(limbs);
        idx
    }

    fn imm_to_field(&self, s: isize) -> BigUint {
        if s >= 0 {
            BigUint::from(s as u64) % &self.builder.prime
        } else {
            &self.builder.prime - BigUint::from(s.unsigned_abs() as u64) % &self.builder.prime
        }
    }

    /// Emit evaluation ops computing `node`, returning the slot holding the result.
    fn emit_eval(&mut self, node: &SymbolicExpr) -> u32 {
        let alloc = |s: &mut Self| {
            let slot = s.next_slot;
            s.next_slot += 1;
            slot as u32
        };
        match node {
            // Input and Var slots are preassigned: inputs at [0, num_input) (loaded once
            // at tape start), vars at [num_input, num_input + num_vars).
            SymbolicExpr::Input(i) => *i as u32,
            SymbolicExpr::Var(i) => (self.builder.num_input + i) as u32,
            SymbolicExpr::Const(i, _, _) => {
                let val = self.builder.constants[*i].0.clone();
                let payload = self.push_mont_payload(&val);
                let dst = alloc(self);
                self.eval_ops.push(EvalOp {
                    opcode: EvalOpcode::Constant,
                    dst,
                    a: payload,
                    ..Default::default()
                });
                dst
            }
            SymbolicExpr::Add(l, r2)
            | SymbolicExpr::Sub(l, r2)
            | SymbolicExpr::Mul(l, r2)
            | SymbolicExpr::Div(l, r2) => {
                let a = self.emit_eval(l);
                let b = self.emit_eval(r2);
                let opcode = match node {
                    SymbolicExpr::Add(..) => EvalOpcode::Add,
                    SymbolicExpr::Sub(..) => EvalOpcode::Sub,
                    SymbolicExpr::Mul(..) => EvalOpcode::Mul,
                    _ => EvalOpcode::Div,
                };
                let dst = alloc(self);
                self.eval_ops.push(EvalOp {
                    opcode,
                    dst,
                    a,
                    b,
                    ..Default::default()
                });
                dst
            }
            SymbolicExpr::IntAdd(l, s) | SymbolicExpr::IntMul(l, s) => {
                let a = self.emit_eval(l);
                let imm = self.imm_to_field(*s);
                let payload = self.push_mont_payload(&imm);
                let opcode = if matches!(node, SymbolicExpr::IntAdd(..)) {
                    EvalOpcode::IntAdd
                } else {
                    EvalOpcode::IntMul
                };
                let dst = alloc(self);
                self.eval_ops.push(EvalOp {
                    opcode,
                    dst,
                    a,
                    b: payload,
                    ..Default::default()
                });
                dst
            }
            SymbolicExpr::Select(flag, l, r2) => {
                let a = self.emit_eval(l);
                let b = self.emit_eval(r2);
                let dst = alloc(self);
                self.eval_ops.push(EvalOp {
                    opcode: EvalOpcode::Select,
                    flag: *flag as u32,
                    dst,
                    a,
                    b,
                });
                dst
            }
        }
    }

    /// Emit witness ops computing `node`; returns (scratch_off, len).
    fn emit_witness(&mut self, node: &SymbolicExpr) -> (u32, u32) {
        let num_limbs = self.builder.num_limbs;
        let alloc = |s: &mut Self, len: u32| {
            let off = s.scratch_top;
            s.scratch_top += len;
            off
        };
        match node {
            SymbolicExpr::Input(i) => {
                let off = alloc(self, num_limbs as u32);
                self.witness_ops.push(WitnessOp {
                    opcode: WitnessOpcode::Input,
                    dst_off: off,
                    dst_len: num_limbs as u32,
                    a_off: *i as u32,
                    ..Default::default()
                });
                (off, num_limbs as u32)
            }
            SymbolicExpr::Var(i) => {
                let off = alloc(self, num_limbs as u32);
                self.witness_ops.push(WitnessOp {
                    opcode: WitnessOpcode::Var,
                    dst_off: off,
                    dst_len: num_limbs as u32,
                    a_off: *i as u32,
                    ..Default::default()
                });
                (off, num_limbs as u32)
            }
            SymbolicExpr::Const(i, _, nl) => {
                let limbs = &self.builder.constants[*i].1;
                assert_eq!(limbs.len(), *nl);
                let payload = self.const_limbs_payload.len() as u32;
                self.const_limbs_payload
                    .extend(limbs.iter().map(|&x| x as i32));
                let off = alloc(self, *nl as u32);
                self.witness_ops.push(WitnessOp {
                    opcode: WitnessOpcode::Constant,
                    dst_off: off,
                    dst_len: *nl as u32,
                    a_off: payload,
                    ..Default::default()
                });
                (off, *nl as u32)
            }
            SymbolicExpr::Add(l, r) | SymbolicExpr::Sub(l, r) => {
                let (ao, al) = self.emit_witness(l);
                let (bo, bl) = self.emit_witness(r);
                let len = al.max(bl);
                let off = alloc(self, len);
                self.witness_ops.push(WitnessOp {
                    opcode: if matches!(node, SymbolicExpr::Add(..)) {
                        WitnessOpcode::Add
                    } else {
                        WitnessOpcode::Sub
                    },
                    dst_off: off,
                    dst_len: len,
                    a_off: ao,
                    a_len: al,
                    b_off: bo,
                    b_len: bl,
                    ..Default::default()
                });
                (off, len)
            }
            SymbolicExpr::Mul(l, r) => {
                let (ao, al) = self.emit_witness(l);
                let (bo, bl) = self.emit_witness(r);
                let len = al + bl - 1;
                let off = alloc(self, len);
                self.witness_ops.push(WitnessOp {
                    opcode: WitnessOpcode::Mul,
                    dst_off: off,
                    dst_len: len,
                    a_off: ao,
                    a_len: al,
                    b_off: bo,
                    b_len: bl,
                    ..Default::default()
                });
                (off, len)
            }
            SymbolicExpr::IntAdd(l, s) | SymbolicExpr::IntMul(l, s) => {
                let (ao, al) = self.emit_witness(l);
                let off = alloc(self, al);
                self.witness_ops.push(WitnessOp {
                    opcode: if matches!(node, SymbolicExpr::IntAdd(..)) {
                        WitnessOpcode::IntAdd
                    } else {
                        WitnessOpcode::IntMul
                    },
                    dst_off: off,
                    dst_len: al,
                    a_off: ao,
                    a_len: al,
                    imm: i32::try_from(*s).expect("imm fits i32"),
                    ..Default::default()
                });
                (off, al)
            }
            SymbolicExpr::Select(flag, l, r) => {
                let (ao, al) = self.emit_witness(l);
                let (bo, bl) = self.emit_witness(r);
                let len = al.max(bl);
                let off = alloc(self, len);
                self.witness_ops.push(WitnessOp {
                    opcode: WitnessOpcode::Select,
                    flag: *flag as u32,
                    dst_off: off,
                    dst_len: len,
                    a_off: ao,
                    a_len: al,
                    b_off: bo,
                    b_len: bl,
                    ..Default::default()
                });
                (off, len)
            }
            SymbolicExpr::Div(..) => unreachable!("Div not allowed in constraints"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TracegenCompileError(String);

impl std::fmt::Display for TracegenCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TracegenCompileError {}

fn require_device_capabilities(
    expr: &FieldExpr,
    local_opcode_idx: &[usize],
    opcode_flag_idx: &[usize],
    width: usize,
) -> Result<(), TracegenCompileError> {
    let b = expr.program().builder();
    let k = (b.num_limbs * b.limb_bits).div_ceil(32);
    let reject = |message: String| Err(TracegenCompileError(message));

    if !b.is_finalized() {
        return reject("tracegen IR requires a finalized FieldExpr".to_owned());
    }
    if b.limb_bits != 8 {
        return reject(format!(
            "CUDA tracegen requires 8-bit limbs, got {}",
            b.limb_bits
        ));
    }
    if k > TRACEGEN_MAX_K as usize {
        return reject(format!(
            "field uses {k} u32 words; CUDA tracegen supports at most {TRACEGEN_MAX_K}"
        ));
    }
    if b.num_flags > TRACEGEN_MAX_FLAGS as usize {
        return reject(format!(
            "expression uses {} flags; CUDA tracegen supports at most {TRACEGEN_MAX_FLAGS}",
            b.num_flags
        ));
    }
    if let Some(&q_limbs) = b
        .q_limbs
        .iter()
        .find(|&&q| q > TRACEGEN_MAX_Q_LIMBS as usize)
    {
        return reject(format!(
            "constraint uses {q_limbs} quotient limbs; CUDA tracegen supports at most \
             {TRACEGEN_MAX_Q_LIMBS}"
        ));
    }
    if width != expr.program().width() {
        return reject(format!(
            "trace width {width} does not match FieldExpr width {}",
            expr.program().width()
        ));
    }
    if local_opcode_idx
        .iter()
        .any(|&opcode| opcode > u8::MAX as usize)
    {
        return reject("local opcode does not fit in the record's opcode byte".to_owned());
    }
    if let Some(&flag) = opcode_flag_idx.iter().find(|&&flag| flag >= b.num_flags) {
        return reject(format!(
            "opcode table references flag {flag}, but expression has {} flags",
            b.num_flags
        ));
    }
    Ok(())
}

/// Compiles a finalized expression into trace-generation IR validated for the CUDA interpreter.
pub fn compile_tracegen_ir(
    expr: &FieldExpr,
    local_opcode_idx: Vec<usize>,
    opcode_flag_idx: Vec<usize>,
    width: usize,
) -> Result<TracegenIr, TracegenCompileError> {
    require_device_capabilities(expr, &local_opcode_idx, &opcode_flag_idx, width)?;
    let b = expr.program().builder();
    let k = (b.num_limbs * b.limb_bits).div_ceil(32);
    let r = (BigUint::one() << (32 * k)) % &b.prime;

    let mut ser = TracegenCompiler {
        builder: b,
        k,
        eval_ops: vec![],
        next_slot: b.num_input + b.num_variables,
        mont_payload: vec![],
        witness_ops: vec![],
        scratch_top: 0,
        const_limbs_payload: vec![],
        r: r.clone(),
    };

    // Load inputs once into their preassigned slots.
    for i in 0..b.num_input {
        ser.eval_ops.push(EvalOp {
            opcode: EvalOpcode::LoadInput,
            dst: i as u32,
            a: i as u32,
            ..Default::default()
        });
    }
    // Evaluation phase: compute each variable in order (computes[i] may reference vars < i).
    for (i, compute) in b.computes.iter().enumerate() {
        let src = ser.emit_eval(compute);
        ser.eval_ops.push(EvalOp {
            opcode: EvalOpcode::SaveVar,
            dst: (b.num_input + i) as u32,
            a: i as u32,
            b: src,
            ..Default::default()
        });
    }

    // Witness phase tapes + carry params via data-independent bound propagation.
    use openvm_circuit_primitives::bigint::{
        check_carry_to_zero::get_carry_max_abs_and_bits, OverflowInt,
    };
    let zero_inputs: Vec<OverflowInt<isize>> = (0..b.num_input)
        .map(|_| {
            OverflowInt::<isize>::from_biguint(&BigUint::zero(), b.limb_bits, Some(b.num_limbs))
        })
        .collect();
    let zero_vars: Vec<OverflowInt<isize>> = (0..b.num_variables)
        .map(|_| {
            OverflowInt::<isize>::from_biguint(&BigUint::zero(), b.limb_bits, Some(b.num_limbs))
        })
        .collect();
    let zero_consts: Vec<OverflowInt<isize>> = b
        .constants
        .iter()
        .map(|(_, limbs)| {
            OverflowInt::<isize>::from_unsigned_limbs(vec![0; limbs.len()], b.limb_bits)
        })
        .collect();
    let flags = vec![false; b.num_flags];
    let prime_overflow = OverflowInt::<isize>::from_biguint(&b.prime, b.limb_bits, None);

    let mut constraints = vec![];
    let mut scratch_len = 0;
    for (i, constraint) in b.constraints.iter().enumerate() {
        // Constraint tapes execute sequentially, so they can share the same scratch arena.
        ser.scratch_top = 0;
        let tape_start = ser.witness_ops.len();
        let (result_off, result_len) = ser.emit_witness(constraint);
        let tape_len = ser.witness_ops.len() - tape_start;
        scratch_len = scratch_len.max(ser.scratch_top as usize);

        // Bound propagation only (limb values are zeros; bounds are data independent).
        // NOTE: for Select nodes the two sides must have identical static bounds, which
        // the builder enforces ("same structure").
        let expr_bound =
            constraint.evaluate_overflow_isize(&zero_inputs, &zero_vars, &zero_consts, &flags);
        let q_bound = OverflowInt::<isize>::from_signed_limbs(vec![0; b.q_limbs[i]], b.limb_bits);
        let total = expr_bound - q_bound * prime_overflow.clone();
        let (carry_min_abs, carry_bits) =
            get_carry_max_abs_and_bits(total.max_overflow_bits(), b.limb_bits);
        assert_eq!(total.num_limbs(), b.carry_limbs[i]);
        // Exact division works mod 2^(32*2K); only |q| needs signed headroom in the
        // 2K-word accumulator (N may wrap; q = N * p^{-1} mod 2^(32*2K) is still exact).
        assert!(
            b.q_limbs[i] * b.limb_bits + 1 < 32 * 2 * k,
            "q too wide for 2K-limb exact division accumulator"
        );

        constraints.push(ConstraintMeta {
            tape_start,
            tape_len,
            result_off,
            result_len,
            q_limbs: b.q_limbs[i],
            carry_limbs: b.carry_limbs[i],
            carry_min_abs: carry_min_abs as u32,
            carry_bits: carry_bits as u32,
        });
    }

    // Field constants.
    let p_u32 = biguint_to_u32s(&b.prime, k);
    let mut x = 1u32;
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(p_u32[0].wrapping_mul(x)));
    }
    let mprime = x.wrapping_neg();
    let r2_u32 = biguint_to_u32s(&((&r * &r) % &b.prime), k);
    let pm2_u32 = biguint_to_u32s(&(&b.prime - BigUint::from(2u32)), k);
    let m2k = BigInt::one() << (32 * 2 * k);
    let pinv = BigInt::from_biguint(Sign::Plus, b.prime.clone())
        .modinv(&m2k)
        .expect("p odd");
    let (_, pinv_digits) = pinv.to_u32_digits();
    let mut pinv_u32 = pinv_digits;
    pinv_u32.resize(2 * k, 0);
    let p8: Vec<i32> = b.prime_limbs.iter().map(|&x| x as i32).collect();

    Ok(TracegenIr {
        num_limbs: b.num_limbs,
        limb_bits: b.limb_bits,
        k,
        num_input: b.num_input,
        num_vars: b.num_variables,
        num_flags: b.num_flags,
        needs_setup: b.needs_setup(),
        width,
        num_eval_slots: ser.next_slot,
        eval_ops: ser.eval_ops,
        witness_ops: ser.witness_ops,
        scratch_len,
        constraints,
        p_u32,
        mprime,
        r2_u32,
        pm2_u32,
        pinv_u32,
        p8,
        mont_payload: ser.mont_payload,
        const_limbs_payload: ser.const_limbs_payload,
        local_opcode_idx,
        opcode_flag_idx,
    })
}
