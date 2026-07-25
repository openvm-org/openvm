//! CPU reference interpreter: defines the exact semantics the CUDA kernel must match.
//! Outputs canonical BabyBear u32 values and (value, bits) range-check pairs.
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

use super::{LimbOpcode, TracegenIr, ValueOpcode};

const F_P: u64 = 0x78000001; // BabyBear

fn f_of_i64(v: i64) -> u32 {
    (((v % F_P as i64) + F_P as i64) % F_P as i64) as u32
}

pub struct ReferenceInterpreter<'a> {
    pub prog: &'a TracegenIr,
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
                ValueOpcode::LoadInput => {
                    let mut canon = vec![0u32; k];
                    let base = op.a as usize * nl;
                    for (i, &byte) in input_limbs[base..base + nl].iter().enumerate() {
                        canon[i * lb / 32] |= (byte as u32) << ((i * lb) % 32);
                    }
                    slots[op.dst as usize] = self.mont_mul(&canon, &prog.r2_u32);
                }
                ValueOpcode::Constant => slots[op.dst as usize] = payload(op.a),
                ValueOpcode::Add => {
                    slots[op.dst as usize] =
                        self.add_mod(&slots[op.a as usize], &slots[op.b as usize])
                }
                ValueOpcode::Sub => {
                    slots[op.dst as usize] =
                        self.sub_mod(&slots[op.a as usize], &slots[op.b as usize])
                }
                ValueOpcode::Mul => {
                    slots[op.dst as usize] =
                        self.mont_mul(&slots[op.a as usize], &slots[op.b as usize])
                }
                ValueOpcode::Div => {
                    let inv = self.mont_inv(&slots[op.b as usize]);
                    slots[op.dst as usize] = self.mont_mul(&slots[op.a as usize], &inv);
                }
                ValueOpcode::IntAdd => {
                    slots[op.dst as usize] = self.add_mod(&slots[op.a as usize], &payload(op.b))
                }
                ValueOpcode::IntMul => {
                    slots[op.dst as usize] = self.mont_mul(&slots[op.a as usize], &payload(op.b))
                }
                ValueOpcode::Select => {
                    slots[op.dst as usize] = if flags[op.flag as usize] {
                        slots[op.a as usize].clone()
                    } else {
                        slots[op.b as usize].clone()
                    }
                }
                ValueOpcode::SaveVar => {
                    var_canon[op.a as usize] = self.mont_mul(&slots[op.b as usize], &one);
                    // Var slot keeps the Montgomery form for later computes.
                    slots[op.dst as usize] = slots[op.b as usize].clone();
                }
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
                    LimbOpcode::Input => {
                        let base = op.a_off as usize * nl;
                        for i in 0..dl {
                            scratch[d + i] = input_limbs[base + i] as i64;
                        }
                    }
                    LimbOpcode::Var => {
                        for i in 0..dl {
                            scratch[d + i] = var_limbs[op.a_off as usize][i] as i64;
                        }
                    }
                    LimbOpcode::Constant => {
                        for i in 0..dl {
                            scratch[d + i] = prog.const_limbs_payload[op.a_off as usize + i] as i64;
                        }
                    }
                    LimbOpcode::Add | LimbOpcode::Sub => {
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
                            scratch[d + i] = if op.opcode == LimbOpcode::Add {
                                a + b
                            } else {
                                a - b
                            };
                        }
                    }
                    LimbOpcode::Mul => {
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
                    LimbOpcode::IntAdd => {
                        for i in 0..dl {
                            scratch[d + i] = scratch[op.a_off as usize + i];
                        }
                        scratch[d] += op.imm as i64;
                    }
                    LimbOpcode::IntMul => {
                        for i in 0..dl {
                            scratch[d + i] = scratch[op.a_off as usize + i] * op.imm as i64;
                        }
                    }
                    LimbOpcode::Select => {
                        let (src, sl) = if flags[op.flag as usize] {
                            (op.a_off as usize, op.a_len as usize)
                        } else {
                            (op.b_off as usize, op.b_len as usize)
                        };
                        for i in 0..dl {
                            scratch[d + i] = if i < sl { scratch[src + i] } else { 0 };
                        }
                    }
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
