//! Serialization of a finalized [`FieldExpr`] into a flat "device program" that a GPU
//! kernel can interpret to perform trace generation (one thread per row), plus a CPU
//! reference interpreter that defines the exact semantics the CUDA kernel must match.
//!
//! Layout of the work per row (mirrors `FieldExpressionFiller::fill_trace_row`):
//! 1. Decode record: opcode byte + input limbs. Map opcode -> flags.
//! 2. Value phase: evaluate `computes` in Montgomery form to obtain each variable (Div -> Fermat
//!    inversion), then store canonical limbs.
//! 3. Constraint phase: for each constraint, evaluate the expression in the *limb* domain (signed
//!    limbs, no carry propagation), derive the integer value N from the limbs, compute q = N / p by
//!    exact division (multiply by p^{-1} mod 2^(32*2K)), subtract conv(q, p_limbs) and run the
//!    carry chain.
//! 4. Emit range checks: var limbs (limb_bits), q limbs shifted (limb_bits + 1), carries shifted
//!    (per-constraint carry_bits).
//! 5. Write the sub-row: [is_valid, inputs, vars, qs, carries, flags].

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};

use crate::{FieldExpr, SymbolicExpr};

/// Value-phase opcodes (field arithmetic over slots of K u32 limbs, Montgomery form).
pub const VOP_LOAD_INPUT: u32 = 0; // dst <- mont(input[a])
pub const VOP_CONST: u32 = 1; // dst <- mont_const[payload a]
pub const VOP_ADD: u32 = 2; // dst <- a + b
pub const VOP_SUB: u32 = 3; // dst <- a - b
pub const VOP_MUL: u32 = 4; // dst <- a * b
pub const VOP_DIV: u32 = 5; // dst <- a * b^(p-2)
pub const VOP_INTADD: u32 = 6; // dst <- a + mont_imm[payload b]
pub const VOP_INTMUL: u32 = 7; // dst <- a * mont_imm[payload b]
pub const VOP_SELECT: u32 = 8; // dst <- flag ? a : b
pub const VOP_SAVE_VAR: u32 = 9; // var[a] <- canonical(slot b)

/// Limb-phase opcodes (signed limb vectors in an i32 scratch arena).
pub const LOP_INPUT: u32 = 0; // scratch[dst..dst+n] <- input limbs (unsigned)
pub const LOP_VAR: u32 = 1; // scratch[dst..dst+n] <- var canonical limbs
pub const LOP_CONST: u32 = 2; // scratch[dst..dst+n] <- const limbs (payload)
pub const LOP_ADD: u32 = 3;
pub const LOP_SUB: u32 = 4;
pub const LOP_MUL: u32 = 5; // convolution, dst_len = a_len + b_len - 1
pub const LOP_INTADD: u32 = 6; // limb 0 += imm
pub const LOP_INTMUL: u32 = 7; // all limbs *= imm
pub const LOP_SELECT: u32 = 8; // flag ? a : b, zero-padded to dst_len

#[derive(Clone, Copy, Debug, Default)]
pub struct ValueOp {
    pub opcode: u32,
    pub flag: u32,
    pub dst: u32,
    pub a: u32,
    pub b: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LimbOp {
    pub opcode: u32,
    pub flag: u32,
    pub dst_off: u32,
    pub dst_len: u32,
    pub a_off: u32,
    pub a_len: u32,
    pub b_off: u32,
    pub b_len: u32,
    pub imm: i32,
}

#[derive(Clone, Debug)]
pub struct ConstraintMeta {
    pub tape_start: usize,
    pub tape_len: usize,
    /// Scratch offset/len of the evaluated constraint expression limbs.
    pub result_off: u32,
    pub result_len: u32,
    pub q_limbs: usize,
    pub carry_limbs: usize,
    /// Range-check shift/bits for carries, from `get_carry_max_abs_and_bits` on the
    /// bound-propagated `expr - q * p` overflow int (data independent).
    pub carry_min_abs: u32,
    pub carry_bits: u32,
}

/// Host-side representation. `to_blob` flattens it for the device.
#[derive(Clone, Debug)]
pub struct DeviceFieldExprProgram {
    pub num_limbs: usize,
    pub limb_bits: usize,
    /// K: number of u32 limbs per field element.
    pub k: usize,
    pub num_input: usize,
    pub num_vars: usize,
    pub num_flags: usize,
    pub needs_setup: bool,
    /// Trace sub-row width (must equal `BaseAir::width(&expr)`).
    pub width: usize,

    pub value_ops: Vec<ValueOp>,
    pub num_value_slots: usize,
    pub limb_ops: Vec<LimbOp>,
    pub scratch_len: usize,
    pub constraints: Vec<ConstraintMeta>,

    // Field constants (u32 little-endian limbs)
    pub p_u32: Vec<u32>,
    pub mprime: u32,
    pub r2_u32: Vec<u32>,
    pub pm2_u32: Vec<u32>,
    /// p^{-1} mod 2^(32*2K), 2K limbs (for exact division).
    pub pinv_u32: Vec<u32>,
    /// Prime as `ceil(p.bits()/limb_bits)` canonical limbs (matches `prime_overflow`).
    pub p8: Vec<i32>,

    /// Montgomery-form payload for VOP_CONST / VOP_INTADD / VOP_INTMUL (K limbs each).
    pub mont_payload: Vec<u32>,
    /// Limb payload for LOP_CONST (concatenated, offsets stored in op.b_off).
    pub const_limbs_payload: Vec<i32>,

    /// Opcode -> flags mapping (from FieldExpressionFiller): position of the record's
    /// local opcode in `local_opcode_idx`; if < opcode_flag_idx.len(), that flag is set.
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
}

fn biguint_to_u32s(x: &BigUint, k: usize) -> Vec<u32> {
    let mut v = x.to_u32_digits();
    assert!(v.len() <= k, "value too wide");
    v.resize(k, 0);
    v
}

struct Serializer<'a> {
    expr: &'a FieldExpr,
    k: usize,
    value_ops: Vec<ValueOp>,
    next_slot: usize,
    mont_payload: Vec<u32>,
    limb_ops: Vec<LimbOp>,
    scratch_top: u32,
    const_limbs_payload: Vec<i32>,
    r: BigUint, // 2^(32K) mod p
}

impl<'a> Serializer<'a> {
    fn mont(&self, x: &BigUint) -> Vec<u32> {
        biguint_to_u32s(&((x * &self.r) % self.expr.program().prime()), self.k)
    }

    fn push_mont_payload(&mut self, x: &BigUint) -> u32 {
        let idx = (self.mont_payload.len() / self.k) as u32;
        let limbs = self.mont(x);
        self.mont_payload.extend(limbs);
        idx
    }

    fn imm_to_field(&self, s: isize) -> BigUint {
        if s >= 0 {
            BigUint::from(s as u64) % self.expr.program().prime()
        } else {
            self.expr.program().prime()
                - BigUint::from(s.unsigned_abs() as u64) % self.expr.program().prime()
        }
    }

    /// Emit value ops computing `node`, returning the slot holding the result.
    fn emit_value(&mut self, node: &SymbolicExpr) -> u32 {
        let alloc = |s: &mut Self| {
            let slot = s.next_slot;
            s.next_slot += 1;
            slot as u32
        };
        match node {
            // Input and Var slots are preassigned: inputs at [0, num_input) (loaded once
            // at tape start), vars at [num_input, num_input + num_vars).
            SymbolicExpr::Input(i) => *i as u32,
            SymbolicExpr::Var(i) => (self.expr.program().builder().num_input + i) as u32,
            SymbolicExpr::Const(i, _, _) => {
                let val = self.expr.program().builder().constants[*i].0.clone();
                let payload = self.push_mont_payload(&val);
                let dst = alloc(self);
                self.value_ops.push(ValueOp {
                    opcode: VOP_CONST,
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
                let a = self.emit_value(l);
                let b = self.emit_value(r2);
                let opcode = match node {
                    SymbolicExpr::Add(..) => VOP_ADD,
                    SymbolicExpr::Sub(..) => VOP_SUB,
                    SymbolicExpr::Mul(..) => VOP_MUL,
                    _ => VOP_DIV,
                };
                let dst = alloc(self);
                self.value_ops.push(ValueOp {
                    opcode,
                    dst,
                    a,
                    b,
                    ..Default::default()
                });
                dst
            }
            SymbolicExpr::IntAdd(l, s) | SymbolicExpr::IntMul(l, s) => {
                let a = self.emit_value(l);
                let imm = self.imm_to_field(*s);
                let payload = self.push_mont_payload(&imm);
                let opcode = if matches!(node, SymbolicExpr::IntAdd(..)) {
                    VOP_INTADD
                } else {
                    VOP_INTMUL
                };
                let dst = alloc(self);
                self.value_ops.push(ValueOp {
                    opcode,
                    dst,
                    a,
                    b: payload,
                    ..Default::default()
                });
                dst
            }
            SymbolicExpr::Select(flag, l, r2) => {
                let a = self.emit_value(l);
                let b = self.emit_value(r2);
                let dst = alloc(self);
                self.value_ops.push(ValueOp {
                    opcode: VOP_SELECT,
                    flag: *flag as u32,
                    dst,
                    a,
                    b,
                });
                dst
            }
        }
    }

    /// Emit limb ops computing `node`; returns (scratch_off, len).
    fn emit_limb(&mut self, node: &SymbolicExpr) -> (u32, u32) {
        let num_limbs = self.expr.program().builder().num_limbs;
        let alloc = |s: &mut Self, len: u32| {
            let off = s.scratch_top;
            s.scratch_top += len;
            off
        };
        match node {
            SymbolicExpr::Input(i) => {
                let off = alloc(self, num_limbs as u32);
                self.limb_ops.push(LimbOp {
                    opcode: LOP_INPUT,
                    dst_off: off,
                    dst_len: num_limbs as u32,
                    a_off: *i as u32,
                    ..Default::default()
                });
                (off, num_limbs as u32)
            }
            SymbolicExpr::Var(i) => {
                let off = alloc(self, num_limbs as u32);
                self.limb_ops.push(LimbOp {
                    opcode: LOP_VAR,
                    dst_off: off,
                    dst_len: num_limbs as u32,
                    a_off: *i as u32,
                    ..Default::default()
                });
                (off, num_limbs as u32)
            }
            SymbolicExpr::Const(i, _, nl) => {
                let limbs = &self.expr.program().builder().constants[*i].1;
                assert_eq!(limbs.len(), *nl);
                let payload = self.const_limbs_payload.len() as u32;
                self.const_limbs_payload
                    .extend(limbs.iter().map(|&x| x as i32));
                let off = alloc(self, *nl as u32);
                self.limb_ops.push(LimbOp {
                    opcode: LOP_CONST,
                    dst_off: off,
                    dst_len: *nl as u32,
                    a_off: payload,
                    ..Default::default()
                });
                (off, *nl as u32)
            }
            SymbolicExpr::Add(l, r) | SymbolicExpr::Sub(l, r) => {
                let (ao, al) = self.emit_limb(l);
                let (bo, bl) = self.emit_limb(r);
                let len = al.max(bl);
                let off = alloc(self, len);
                self.limb_ops.push(LimbOp {
                    opcode: if matches!(node, SymbolicExpr::Add(..)) {
                        LOP_ADD
                    } else {
                        LOP_SUB
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
                let (ao, al) = self.emit_limb(l);
                let (bo, bl) = self.emit_limb(r);
                let len = al + bl - 1;
                let off = alloc(self, len);
                self.limb_ops.push(LimbOp {
                    opcode: LOP_MUL,
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
                let (ao, al) = self.emit_limb(l);
                let off = alloc(self, al);
                self.limb_ops.push(LimbOp {
                    opcode: if matches!(node, SymbolicExpr::IntAdd(..)) {
                        LOP_INTADD
                    } else {
                        LOP_INTMUL
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
                let (ao, al) = self.emit_limb(l);
                let (bo, bl) = self.emit_limb(r);
                let len = al.max(bl);
                let off = alloc(self, len);
                self.limb_ops.push(LimbOp {
                    opcode: LOP_SELECT,
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

pub fn serialize_field_expr(
    expr: &FieldExpr,
    local_opcode_idx: Vec<usize>,
    opcode_flag_idx: Vec<usize>,
    width: usize,
) -> DeviceFieldExprProgram {
    let b = expr.program().builder();
    assert!(b.is_finalized());
    let k = (b.num_limbs * b.limb_bits).div_ceil(32);
    let r = (BigUint::one() << (32 * k)) % &b.prime;

    let mut ser = Serializer {
        expr,
        k,
        value_ops: vec![],
        next_slot: b.num_input + b.num_variables,
        mont_payload: vec![],
        limb_ops: vec![],
        scratch_top: 0,
        const_limbs_payload: vec![],
        r: r.clone(),
    };

    // Load inputs once into their preassigned slots.
    for i in 0..b.num_input {
        ser.value_ops.push(ValueOp {
            opcode: VOP_LOAD_INPUT,
            dst: i as u32,
            a: i as u32,
            ..Default::default()
        });
    }
    // Value phase: compute each variable in order (computes[i] may reference vars < i).
    for (i, compute) in b.computes.iter().enumerate() {
        let src = ser.emit_value(compute);
        ser.value_ops.push(ValueOp {
            opcode: VOP_SAVE_VAR,
            dst: (b.num_input + i) as u32,
            a: i as u32,
            b: src,
            ..Default::default()
        });
    }

    // Constraint phase tapes + carry params via data-independent bound propagation.
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
    for (i, constraint) in b.constraints.iter().enumerate() {
        let tape_start = ser.limb_ops.len();
        let (result_off, result_len) = ser.emit_limb(constraint);
        let tape_len = ser.limb_ops.len() - tape_start;

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

    DeviceFieldExprProgram {
        num_limbs: b.num_limbs,
        limb_bits: b.limb_bits,
        k,
        num_input: b.num_input,
        num_vars: b.num_variables,
        num_flags: b.num_flags,
        needs_setup: b.needs_setup(),
        width,
        num_value_slots: ser.next_slot,
        value_ops: ser.value_ops,
        limb_ops: ser.limb_ops,
        scratch_len: ser.scratch_top as usize,
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
    }
}

// ---------------------------------------------------------------------------
// Flat blob encoding (shared with the CUDA kernel).
// Header (u32 words):
//  0: num_limbs   1: limb_bits  2: k          3: num_input
//  4: num_vars    5: num_flags  6: needs_setup 7: width
//  8: num_value_slots  9: n_value_ops  10: n_limb_ops  11: n_constraints
//  12: scratch_len 13: p8_len  14: n_local_opcodes  15: n_opcode_flags
//  16..: section offsets (in u32 words from blob start):
//  16: value_ops  17: limb_ops  18: constraint_meta  19: p_u32  20: r2  21: pm2
//  22: pinv  23: p8  24: mont_payload  25: const_limbs  26: opcode_tables  27: mprime
// ---------------------------------------------------------------------------
impl DeviceFieldExprProgram {
    pub fn to_blob(&self) -> Vec<u32> {
        let mut blob = vec![0u32; 28];
        blob[0] = self.num_limbs as u32;
        blob[1] = self.limb_bits as u32;
        blob[2] = self.k as u32;
        blob[3] = self.num_input as u32;
        blob[4] = self.num_vars as u32;
        blob[5] = self.num_flags as u32;
        blob[6] = self.needs_setup as u32;
        blob[7] = self.width as u32;
        blob[8] = self.num_value_slots as u32;
        blob[9] = self.value_ops.len() as u32;
        blob[10] = self.limb_ops.len() as u32;
        blob[11] = self.constraints.len() as u32;
        blob[12] = self.scratch_len as u32;
        blob[13] = self.p8.len() as u32;
        blob[14] = self.local_opcode_idx.len() as u32;
        blob[15] = self.opcode_flag_idx.len() as u32;

        blob[16] = blob.len() as u32;
        for op in &self.value_ops {
            blob.extend([op.opcode, op.flag, op.dst, op.a, op.b]);
        }
        blob[17] = blob.len() as u32;
        for op in &self.limb_ops {
            blob.extend([
                op.opcode,
                op.flag,
                op.dst_off,
                op.dst_len,
                op.a_off,
                op.a_len,
                op.b_off,
                op.b_len,
                op.imm as u32,
            ]);
        }
        blob[18] = blob.len() as u32;
        for c in &self.constraints {
            blob.extend([
                c.tape_start as u32,
                c.tape_len as u32,
                c.result_off,
                c.result_len,
                c.q_limbs as u32,
                c.carry_limbs as u32,
                c.carry_min_abs,
                c.carry_bits,
            ]);
        }
        blob[19] = blob.len() as u32;
        blob.extend(&self.p_u32);
        blob[20] = blob.len() as u32;
        blob.extend(&self.r2_u32);
        blob[21] = blob.len() as u32;
        blob.extend(&self.pm2_u32);
        blob[22] = blob.len() as u32;
        blob.extend(&self.pinv_u32);
        blob[23] = blob.len() as u32;
        blob.extend(self.p8.iter().map(|&x| x as u32));
        blob[24] = blob.len() as u32;
        blob.extend(&self.mont_payload);
        blob[25] = blob.len() as u32;
        blob.extend(self.const_limbs_payload.iter().map(|&x| x as u32));
        blob[26] = blob.len() as u32;
        blob.extend(self.local_opcode_idx.iter().map(|&x| x as u32));
        blob.extend(self.opcode_flag_idx.iter().map(|&x| x as u32));
        blob[27] = self.mprime;
        blob
    }
}

// ---------------------------------------------------------------------------
// CPU reference interpreter: defines the exact semantics for the CUDA kernel.
// Outputs canonical BabyBear u32 values and (value, bits) range-check pairs.
// ---------------------------------------------------------------------------

const F_P: u64 = 0x78000001; // BabyBear

fn f_of_i64(v: i64) -> u32 {
    (((v % F_P as i64) + F_P as i64) % F_P as i64) as u32
}

pub struct ReferenceInterpreter<'a> {
    pub prog: &'a DeviceFieldExprProgram,
}

impl<'a> ReferenceInterpreter<'a> {
    fn mont_mul(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
        let k = self.prog.k;
        let p = &self.prog.p_u32;
        let mut t = vec![0u64; k + 2];
        for &ai in a.iter().take(k) {
            let mut carry = 0u64;
            for j in 0..k {
                let cur = t[j] + ai as u64 * b[j] as u64 + carry;
                t[j] = cur & 0xffffffff;
                carry = cur >> 32;
            }
            let cur = t[k] + carry;
            t[k] = cur & 0xffffffff;
            t[k + 1] = cur >> 32;
            let m = (t[0] as u32).wrapping_mul(self.prog.mprime) as u64;
            let mut carry = (t[0] + m * p[0] as u64) >> 32;
            for j in 1..k {
                let cur = t[j] + m * p[j] as u64 + carry;
                t[j - 1] = cur & 0xffffffff;
                carry = cur >> 32;
            }
            let cur = t[k] + carry;
            t[k - 1] = cur & 0xffffffff;
            t[k] = t[k + 1] + (cur >> 32);
            t[k + 1] = 0;
        }
        // conditional subtract
        let mut s = vec![0u32; k];
        let mut borrow = 0i64;
        for j in 0..k {
            let cur = t[j] as i64 - p[j] as i64 - borrow;
            s[j] = cur as u32;
            borrow = if cur < 0 { 1 } else { 0 };
        }
        if t[k] != 0 || borrow == 0 {
            s
        } else {
            t[..k].iter().map(|&x| x as u32).collect()
        }
    }

    fn add_mod(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
        let k = self.prog.k;
        let p = &self.prog.p_u32;
        let mut t = vec![0u32; k];
        let mut carry = 0u64;
        for j in 0..k {
            let cur = a[j] as u64 + b[j] as u64 + carry;
            t[j] = cur as u32;
            carry = cur >> 32;
        }
        let mut s = vec![0u32; k];
        let mut borrow = 0i64;
        for j in 0..k {
            let cur = t[j] as i64 - p[j] as i64 - borrow;
            s[j] = cur as u32;
            borrow = if cur < 0 { 1 } else { 0 };
        }
        if carry != 0 || borrow == 0 {
            s
        } else {
            t
        }
    }

    fn sub_mod(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
        let k = self.prog.k;
        let p = &self.prog.p_u32;
        let mut t = vec![0u32; k];
        let mut borrow = 0i64;
        for j in 0..k {
            let cur = a[j] as i64 - b[j] as i64 - borrow;
            t[j] = cur as u32;
            borrow = if cur < 0 { 1 } else { 0 };
        }
        if borrow != 0 {
            let mut carry = 0u64;
            for j in 0..k {
                let cur = t[j] as u64 + p[j] as u64 + carry;
                t[j] = cur as u32;
                carry = cur >> 32;
            }
        }
        t
    }

    fn mont_inv(&self, a: &[u32]) -> Vec<u32> {
        // a^(p-2), square-and-multiply MSB->LSB. inv(0) = 0 by convention.
        let k = self.prog.k;
        let mut acc: Option<Vec<u32>> = None;
        for bit in (0..32 * k).rev() {
            if let Some(v) = &acc {
                let sq = self.mont_mul(v, v);
                acc = Some(sq);
            }
            if (self.prog.pm2_u32[bit / 32] >> (bit % 32)) & 1 == 1 {
                acc = Some(match acc {
                    Some(v) => self.mont_mul(&v, a),
                    None => a.to_vec(),
                });
            }
        }
        acc.unwrap_or_else(|| vec![0; k])
    }

    /// Fill one sub-row. `input_limbs`: num_input*num_limbs bytes; `opcode`: record opcode.
    /// Returns (row of canonical BabyBear u32, range-check (value, bits) pairs).
    pub fn fill_row(&self, opcode: usize, input_limbs: &[u8]) -> (Vec<u32>, Vec<(u32, u32)>) {
        let prog = self.prog;
        let (k, nl, lb) = (prog.k, prog.num_limbs, prog.limb_bits);
        assert_eq!(lb, 8, "reference interpreter assumes 8-bit limbs");
        assert_eq!(input_limbs.len(), prog.num_input * nl);

        // flags from opcode
        let mut flags = vec![false; prog.num_flags];
        if prog.needs_setup {
            if let Some(pos) = prog.local_opcode_idx.iter().position(|&x| x == opcode) {
                if pos < prog.opcode_flag_idx.len() {
                    flags[prog.opcode_flag_idx[pos]] = true;
                }
            }
        }

        // ---- value phase (Montgomery) ----
        let one = {
            let mut v = vec![0u32; k];
            v[0] = 1;
            v
        };
        let mut slots = vec![vec![0u32; k]; prog.num_value_slots];
        let mut var_canon = vec![vec![0u32; k]; prog.num_vars];
        for op in &prog.value_ops {
            let payload = |idx: u32| -> Vec<u32> {
                prog.mont_payload[idx as usize * k..(idx as usize + 1) * k].to_vec()
            };
            match op.opcode {
                VOP_LOAD_INPUT => {
                    let mut canon = vec![0u32; k];
                    let base = op.a as usize * nl;
                    for (i, &byte) in input_limbs[base..base + nl].iter().enumerate() {
                        canon[i * lb / 32] |= (byte as u32) << ((i * lb) % 32);
                    }
                    slots[op.dst as usize] = self.mont_mul(&canon, &prog.r2_u32);
                }
                VOP_CONST => slots[op.dst as usize] = payload(op.a),
                VOP_ADD => {
                    slots[op.dst as usize] =
                        self.add_mod(&slots[op.a as usize], &slots[op.b as usize])
                }
                VOP_SUB => {
                    slots[op.dst as usize] =
                        self.sub_mod(&slots[op.a as usize], &slots[op.b as usize])
                }
                VOP_MUL => {
                    slots[op.dst as usize] =
                        self.mont_mul(&slots[op.a as usize], &slots[op.b as usize])
                }
                VOP_DIV => {
                    let inv = self.mont_inv(&slots[op.b as usize]);
                    slots[op.dst as usize] = self.mont_mul(&slots[op.a as usize], &inv);
                }
                VOP_INTADD => {
                    slots[op.dst as usize] = self.add_mod(&slots[op.a as usize], &payload(op.b))
                }
                VOP_INTMUL => {
                    slots[op.dst as usize] = self.mont_mul(&slots[op.a as usize], &payload(op.b))
                }
                VOP_SELECT => {
                    slots[op.dst as usize] = if flags[op.flag as usize] {
                        slots[op.a as usize].clone()
                    } else {
                        slots[op.b as usize].clone()
                    }
                }
                VOP_SAVE_VAR => {
                    var_canon[op.a as usize] = self.mont_mul(&slots[op.b as usize], &one);
                    // Var slot keeps the Montgomery form for later computes.
                    slots[op.dst as usize] = slots[op.b as usize].clone();
                }
                _ => unreachable!(),
            }
        }

        let unpack8 = |v: &[u32]| -> Vec<i32> {
            (0..nl)
                .map(|i| ((v[i / 4] >> ((i % 4) * 8)) & 0xff) as i32)
                .collect()
        };
        let var_limbs: Vec<Vec<i32>> = var_canon.iter().map(|v| unpack8(v)).collect();

        // ---- constraint phase ----
        let mut rc: Vec<(u32, u32)> = vec![];
        let mut scratch = vec![0i64; prog.scratch_len];
        let mut all_q: Vec<Vec<i64>> = vec![];
        let mut all_carry: Vec<Vec<i64>> = vec![];
        for c in &prog.constraints {
            for op in &prog.limb_ops[c.tape_start..c.tape_start + c.tape_len] {
                let (d, dl) = (op.dst_off as usize, op.dst_len as usize);
                match op.opcode {
                    LOP_INPUT => {
                        let base = op.a_off as usize * nl;
                        for i in 0..dl {
                            scratch[d + i] = input_limbs[base + i] as i64;
                        }
                    }
                    LOP_VAR => {
                        for i in 0..dl {
                            scratch[d + i] = var_limbs[op.a_off as usize][i] as i64;
                        }
                    }
                    LOP_CONST => {
                        for i in 0..dl {
                            scratch[d + i] = prog.const_limbs_payload[op.a_off as usize + i] as i64;
                        }
                    }
                    LOP_ADD | LOP_SUB => {
                        for i in 0..dl {
                            let a = if i < op.a_len as usize {
                                scratch[op.a_off as usize + i]
                            } else {
                                0
                            };
                            let b = if i < op.b_len as usize {
                                scratch[op.b_off as usize + i]
                            } else {
                                0
                            };
                            scratch[d + i] = if op.opcode == LOP_ADD { a + b } else { a - b };
                        }
                    }
                    LOP_MUL => {
                        for i in 0..dl {
                            let mut acc = 0i64;
                            let lo = (i + 1).saturating_sub(op.b_len as usize);
                            let hi = i.min(op.a_len as usize - 1);
                            for j in lo..=hi {
                                acc += scratch[op.a_off as usize + j]
                                    * scratch[op.b_off as usize + (i - j)];
                            }
                            scratch[d + i] = acc;
                        }
                    }
                    LOP_INTADD => {
                        for i in 0..dl {
                            scratch[d + i] = scratch[op.a_off as usize + i];
                        }
                        scratch[d] += op.imm as i64;
                    }
                    LOP_INTMUL => {
                        for i in 0..dl {
                            scratch[d + i] = scratch[op.a_off as usize + i] * op.imm as i64;
                        }
                    }
                    LOP_SELECT => {
                        let (src, sl) = if flags[op.flag as usize] {
                            (op.a_off as usize, op.a_len as usize)
                        } else {
                            (op.b_off as usize, op.b_len as usize)
                        };
                        for i in 0..dl {
                            scratch[d + i] = if i < sl { scratch[src + i] } else { 0 };
                        }
                    }
                    _ => unreachable!(),
                }
            }

            // N mod 2^(64K) from result limbs (signed, two's complement in 2K u32 words).
            let res = &scratch[c.result_off as usize..(c.result_off + c.result_len) as usize];
            let mut n = vec![0u32; 2 * k];
            let add_signed_shifted = |n: &mut Vec<u32>, v: i64, byte_off: usize| {
                // add v * 2^(8*byte_off) into n (two's complement)
                let (word, shift) = (byte_off / 4, (byte_off % 4) * 8);
                let mag = v.unsigned_abs() as u128;
                let val = mag << shift;
                if v >= 0 {
                    let mut carry = 0u64;
                    let mut w = word;
                    let mut rem = val;
                    while (rem != 0 || carry != 0) && w < 2 * k {
                        let cur = n[w] as u64 + (rem & 0xffffffff) as u64 + carry;
                        n[w] = cur as u32;
                        carry = cur >> 32;
                        rem >>= 32;
                        w += 1;
                    }
                } else {
                    let mut borrow = 0i64;
                    let mut w = word;
                    let mut rem = val;
                    while (rem != 0 || borrow != 0) && w < 2 * k {
                        let cur = n[w] as i64 - (rem & 0xffffffff) as i64 - borrow;
                        n[w] = cur as u32;
                        borrow = if cur < 0 { 1 } else { 0 };
                        rem >>= 32;
                        w += 1;
                    }
                }
            };
            for (i, &v) in res.iter().enumerate() {
                add_signed_shifted(&mut n, v, i);
            }
            // exact division q = n * pinv mod 2^(64K)
            let mut q512 = vec![0u32; 2 * k];
            for i in 0..2 * k {
                let mut carry = 0u64;
                for j in 0..2 * k - i {
                    let prod = n[i] as u64 * prog.pinv_u32[j] as u64;
                    let cur = q512[i + j] as u64 + (prod & 0xffffffff) + carry;
                    q512[i + j] = cur as u32;
                    carry = (cur >> 32) + (prod >> 32);
                }
            }
            // signed q limbs
            let neg = q512[2 * k - 1] >> 31 != 0;
            let mut mag = q512.clone();
            if neg {
                let mut carry = 1u64;
                for w in mag.iter_mut() {
                    let cur = (!*w) as u64 + carry;
                    *w = cur as u32;
                    carry = cur >> 32;
                }
            }
            let q_limbs: Vec<i64> = (0..c.q_limbs)
                .map(|i| {
                    let byte = (mag[i / 4] >> ((i % 4) * 8)) & 0xff;
                    if neg {
                        -(byte as i64)
                    } else {
                        byte as i64
                    }
                })
                .collect();
            for &q in &q_limbs {
                rc.push(((q + (1 << lb)) as u32, (lb + 1) as u32));
            }

            // carries of expr - q*p
            let p8 = &prog.p8;
            let mut carry_acc = 0i64;
            let mut carries = Vec::with_capacity(c.carry_limbs);
            for i in 0..c.carry_limbs {
                let mut e = if i < res.len() { res[i] } else { 0 };
                let lo = (i + 1).saturating_sub(p8.len());
                let hi = i.min(c.q_limbs - 1);
                for j in lo..=hi {
                    e -= q_limbs[j] * p8[i - j] as i64;
                }
                carry_acc = (e + carry_acc) >> lb;
                carries.push(carry_acc);
            }
            for &cv in &carries {
                rc.push(((cv + c.carry_min_abs as i64) as u32, c.carry_bits));
            }
            all_q.push(q_limbs);
            all_carry.push(carries);
        }

        for vl in &var_limbs {
            for &l in vl {
                rc.push((l as u32, lb as u32));
            }
        }

        // ---- write row ----
        let mut row = Vec::with_capacity(prog.width);
        row.push(1u32);
        row.extend(input_limbs.iter().map(|&x| x as u32));
        for vl in &var_limbs {
            row.extend(vl.iter().map(|&x| x as u32));
        }
        for q in &all_q {
            row.extend(q.iter().map(|&x| f_of_i64(x)));
        }
        for cs in &all_carry {
            row.extend(cs.iter().map(|&x| f_of_i64(x)));
        }
        for &f in &flags {
            row.push(f as u32);
        }
        assert_eq!(row.len(), prog.width);
        (row, rc)
    }
}

// ---------------------------------------------------------------------------
// Tests: the reference interpreter must reproduce generate_subrow bit-for-bit.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod device_program_tests {
    use std::sync::{atomic::Ordering, Arc};

    use num_bigint::BigUint;
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime, var_range::VariableRangeCheckerChip,
        TraceSubRowGenerator,
    };
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_field::{PrimeCharacteristicRing, PrimeField32},
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;
    use crate::{test_utils::*, utils::biguint_to_limbs_vec, ExprBuilder, FieldVariable};

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    /// rows: (opcode, optional explicit inputs); None means random inputs mod p.
    fn check_equivalence(
        expr: crate::FieldExpr,
        range_checker: Arc<VariableRangeCheckerChip>,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        rows: &[(usize, Option<Vec<BigUint>>)],
        n_random_repeats: usize,
    ) {
        let width = BaseAir::<BabyBear>::width(&expr);
        let prog = serialize_field_expr(
            &expr,
            local_opcode_idx.clone(),
            opcode_flag_idx.clone(),
            width,
        );
        assert_eq!(prog.width, width);
        let interp = ReferenceInterpreter { prog: &prog };
        let ref_checker = Arc::new(VariableRangeCheckerChip::new(range_checker.bus()));

        let nl = expr.canonical_num_limbs();
        let prime = expr.program().builder().prime.clone();
        let mut state = 0xdeadbeef12345678u64;
        for rep in 0..n_random_repeats {
            for (row_i, (opcode, explicit)) in rows.iter().enumerate() {
                let inputs: Vec<BigUint> = match explicit {
                    Some(v) => v.clone(),
                    None => (0..expr.program().builder().num_input)
                        .map(|_| {
                            let bytes: Vec<u8> = (0..nl).map(|_| lcg(&mut state)).collect();
                            BigUint::from_bytes_le(&bytes) % &prime
                        })
                        .collect(),
                };
                let bytes: Vec<u8> = inputs
                    .iter()
                    .flat_map(|x| biguint_to_limbs_vec(x, nl))
                    .collect();

                // Flags exactly as FieldExpressionFiller derives them.
                let mut flags = vec![false; expr.program().builder().num_flags];
                if expr.needs_setup() {
                    if let Some(pos) = local_opcode_idx.iter().position(|&x| x == *opcode) {
                        if pos < opcode_flag_idx.len() {
                            flags[opcode_flag_idx[pos]] = true;
                        }
                    }
                }

                let mut cpu_row = BabyBear::zero_vec(width);
                expr.generate_subrow((range_checker.as_ref(), inputs, flags), &mut cpu_row);
                let cpu_u32: Vec<u32> = cpu_row.iter().map(|x| x.as_canonical_u32()).collect();

                let (ref_row, rc) = interp.fill_row(*opcode, &bytes);
                assert_eq!(
                    cpu_u32, ref_row,
                    "row mismatch (rep {rep}, row {row_i}, opcode {opcode})"
                );
                for (v, b) in rc {
                    ref_checker.add_count(v, b as usize);
                }
            }
        }
        let a: Vec<u32> = range_checker
            .count
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect();
        let b: Vec<u32> = ref_checker
            .count
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect();
        assert_eq!(a, b, "range checker counts mismatch");
    }

    #[test]
    fn device_program_matches_subrow_ec_add_ne() {
        // Same expression as ec_add_ne_expr (weierstrass chip).
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let x1 = ExprBuilder::new_input(builder.clone());
        let y1 = ExprBuilder::new_input(builder.clone());
        let x2 = ExprBuilder::new_input(builder.clone());
        let y2 = ExprBuilder::new_input(builder.clone());
        let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
        let mut x3 = lambda.square() - x1.clone() - x2;
        x3.save_output();
        let mut y3 = lambda * (x1 - x3.clone()) - y1;
        y3.save_output();
        let builder = (*builder).borrow().clone();
        let expr = crate::FieldExpr::new(builder, range_checker.bus(), true);

        // 1-op chip that needs setup: local ops [0, 2], default flag 0 for op 0.
        check_equivalence(expr, range_checker, vec![0, 2], vec![0], &[(0, None)], 25);
    }

    #[test]
    fn device_program_matches_subrow_muldiv_flags() {
        // Same expression as modular muldiv_expr: Select in constraint and compute,
        // Div under Select, two flags + setup rows.
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let x = ExprBuilder::new_input(builder.clone());
        let y = ExprBuilder::new_input(builder.clone());
        let (z_idx, z) = (*builder).borrow_mut().new_var();
        let mut z = FieldVariable::from_var(builder.clone(), z);
        let is_mul_flag = (*builder).borrow_mut().new_flag();
        let is_div_flag = (*builder).borrow_mut().new_flag();
        let lvar = FieldVariable::select(is_mul_flag, &x, &z);
        let rvar = FieldVariable::select(is_mul_flag, &z, &x);
        let constraint = lvar * y.clone() - rvar;
        (*builder)
            .borrow_mut()
            .set_constraint(z_idx, constraint.expr);
        let compute = SymbolicExpr::Select(
            is_mul_flag,
            Box::new(x.expr.clone() * y.expr.clone()),
            Box::new(SymbolicExpr::Select(
                is_div_flag,
                Box::new(x.expr.clone() / y.expr.clone()),
                Box::new(x.expr.clone()),
            )),
        );
        (*builder).borrow_mut().set_compute(z_idx, compute);
        z.save_output();
        let builder = (*builder).borrow().clone();
        let expr = crate::FieldExpr::new(builder, range_checker.bus(), true);

        // Ops: mul (local 2, flag 0), div (local 3, flag 1), setup (local 4).
        let setup_inputs = vec![prime.clone(), BigUint::from(0u32)];
        check_equivalence(
            expr,
            range_checker,
            vec![2, 3, 4],
            vec![0, 1],
            &[(2, None), (3, None), (4, Some(setup_inputs))],
            10,
        );
    }

    #[test]
    fn device_program_matches_subrow_int_ops() {
        // EcDouble-flavored expression covering IntMul and IntAdd.
        let prime = secp256k1_coord_prime();
        let (range_checker, builder) = setup(&prime);
        let mut x1 = ExprBuilder::new_input(builder.clone());
        let mut y1 = ExprBuilder::new_input(builder.clone());
        let mut nom = x1.square().int_mul(3);
        let mut denom = y1.int_mul(2);
        let mut lambda = nom.div(&mut denom);
        let mut x3 = lambda.square() - x1.int_mul(2);
        x3.save_output();
        let mut y3 = lambda * (x1.clone() - x3.clone()) - y1.clone();
        y3.save_output();
        let mut w = x1.int_add(-7) + y1.int_add(11);
        w.save_output();
        let builder = (*builder).borrow().clone();
        let expr = crate::FieldExpr::new(builder, range_checker.bus(), true);

        check_equivalence(expr, range_checker, vec![0, 2], vec![0], &[(0, None)], 25);
    }
}

#[cfg(test)]
mod device_program_dump {
    //! Dumps GPU validation vectors: blob, records, expected rows, rc counts.
    //! Run with DEVICE_PROGRAM_DUMP_DIR=/path cargo test ... -- --ignored dump_gpu_vectors
    use std::{fs, io::Write, sync::atomic::Ordering};

    use num_bigint::BigUint;
    use openvm_circuit_primitives::{
        bigint::utils::secp256k1_coord_prime, var_range::VariableRangeCheckerChip,
        TraceSubRowGenerator,
    };
    use openvm_stark_backend::{
        p3_air::BaseAir,
        p3_field::{PrimeCharacteristicRing, PrimeField32},
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;
    use crate::{test_utils::*, utils::biguint_to_limbs_vec, ExprBuilder, FieldVariable};

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    fn write_u32s(path: &str, data: &[u32]) {
        let mut f = fs::File::create(path).unwrap();
        for &x in data {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }

    fn dump_case(
        dir: &str,
        name: &str,
        expr: crate::FieldExpr,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        opcodes: &[usize],
        rows: usize,
    ) {
        let width = BaseAir::<BabyBear>::width(&expr);
        let prog = serialize_field_expr(
            &expr,
            local_opcode_idx.clone(),
            opcode_flag_idx.clone(),
            width,
        );
        let blob = prog.to_blob();

        let nl = expr.canonical_num_limbs();
        let prime = expr.program().builder().prime.clone();
        let rec_stride = 1 + expr.program().builder().num_input * nl;
        let range_checker = std::sync::Arc::new(VariableRangeCheckerChip::new(expr.range_bus));

        let mut state = 0x0123456789abcdefu64;
        let mut records = Vec::with_capacity(rows * rec_stride);
        let mut expected = Vec::with_capacity(rows * width);
        for r in 0..rows {
            let opcode = opcodes[r % opcodes.len()];
            let inputs: Vec<BigUint> = (0..expr.program().builder().num_input)
                .map(|_| {
                    let bytes: Vec<u8> = (0..nl).map(|_| lcg(&mut state)).collect();
                    BigUint::from_bytes_le(&bytes) % &prime
                })
                .collect();
            records.push(opcode as u8);
            for x in &inputs {
                records.extend(biguint_to_limbs_vec(x, nl));
            }
            let mut flags = vec![false; expr.program().builder().num_flags];
            if expr.needs_setup() {
                if let Some(pos) = local_opcode_idx.iter().position(|&x| x == opcode) {
                    if pos < opcode_flag_idx.len() {
                        flags[opcode_flag_idx[pos]] = true;
                    }
                }
            }
            let mut row = BabyBear::zero_vec(width);
            expr.generate_subrow((range_checker.as_ref(), inputs, flags), &mut row);
            expected.extend(row.iter().map(|x| x.as_canonical_u32()));
        }
        let rc: Vec<u32> = range_checker
            .count
            .iter()
            .map(|x| x.load(Ordering::Relaxed))
            .collect();

        write_u32s(&format!("{dir}/{name}.blob"), &blob);
        fs::write(format!("{dir}/{name}.records"), &records).unwrap();
        write_u32s(&format!("{dir}/{name}.expected"), &expected);
        write_u32s(&format!("{dir}/{name}.rc"), &rc);
        write_u32s(
            &format!("{dir}/{name}.meta"),
            &[rec_stride as u32, rows as u32, rc.len() as u32],
        );
        println!(
            "dumped {name}: width={width} rows={rows} rec_stride={rec_stride} rc_len={}",
            rc.len()
        );
    }

    #[test]
    #[ignore]
    fn dump_gpu_vectors() {
        let dir = std::env::var("DEVICE_PROGRAM_DUMP_DIR").unwrap_or("/tmp/dp_vectors".into());
        fs::create_dir_all(&dir).unwrap();

        // EcAddNe shape
        {
            let prime = secp256k1_coord_prime();
            let (range_checker, builder) = setup(&prime);
            let x1 = ExprBuilder::new_input(builder.clone());
            let y1 = ExprBuilder::new_input(builder.clone());
            let x2 = ExprBuilder::new_input(builder.clone());
            let y2 = ExprBuilder::new_input(builder.clone());
            let mut lambda = (y2 - y1.clone()) / (x2.clone() - x1.clone());
            let mut x3 = lambda.square() - x1.clone() - x2;
            x3.save_output();
            let mut y3 = lambda * (x1 - x3.clone()) - y1;
            y3.save_output();
            let b = (*builder).borrow().clone();
            let expr = crate::FieldExpr::new(b, range_checker.bus(), true);
            dump_case(&dir, "ecaddne", expr, vec![0, 2], vec![0], &[0], 32768);
        }
        // MulDiv with flags
        {
            let prime = secp256k1_coord_prime();
            let (range_checker, builder) = setup(&prime);
            let x = ExprBuilder::new_input(builder.clone());
            let y = ExprBuilder::new_input(builder.clone());
            let (z_idx, z) = (*builder).borrow_mut().new_var();
            let mut z = FieldVariable::from_var(builder.clone(), z);
            let is_mul_flag = (*builder).borrow_mut().new_flag();
            let is_div_flag = (*builder).borrow_mut().new_flag();
            let lvar = FieldVariable::select(is_mul_flag, &x, &z);
            let rvar = FieldVariable::select(is_mul_flag, &z, &x);
            let constraint = lvar * y.clone() - rvar;
            (*builder)
                .borrow_mut()
                .set_constraint(z_idx, constraint.expr);
            let compute = SymbolicExpr::Select(
                is_mul_flag,
                Box::new(x.expr.clone() * y.expr.clone()),
                Box::new(SymbolicExpr::Select(
                    is_div_flag,
                    Box::new(x.expr.clone() / y.expr.clone()),
                    Box::new(x.expr.clone()),
                )),
            );
            (*builder).borrow_mut().set_compute(z_idx, compute);
            z.save_output();
            let b = (*builder).borrow().clone();
            let expr = crate::FieldExpr::new(b, range_checker.bus(), true);
            dump_case(
                &dir,
                "muldiv",
                expr,
                vec![2, 3, 4],
                vec![0, 1],
                &[2, 3],
                16384,
            );
        }
        // Int ops (EcDouble flavored)
        {
            let prime = secp256k1_coord_prime();
            let (range_checker, builder) = setup(&prime);
            let mut x1 = ExprBuilder::new_input(builder.clone());
            let mut y1 = ExprBuilder::new_input(builder.clone());
            let mut nom = x1.square().int_mul(3);
            let mut denom = y1.int_mul(2);
            let mut lambda = nom.div(&mut denom);
            let mut x3 = lambda.square() - x1.int_mul(2);
            x3.save_output();
            let mut y3 = lambda * (x1.clone() - x3.clone()) - y1.clone();
            y3.save_output();
            let mut w = x1.int_add(-7) + y1.int_add(11);
            w.save_output();
            let b = (*builder).borrow().clone();
            let expr = crate::FieldExpr::new(b, range_checker.bus(), true);
            dump_case(&dir, "intops", expr, vec![0, 2], vec![0], &[0], 16384);
        }
    }
}
