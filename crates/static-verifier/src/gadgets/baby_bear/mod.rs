use halo2_base::{
    AssignedValue, Context,
    QuantumCell::Constant,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
    utils::ScalarField,
};

use crate::circuit::Fr;

pub const BABY_BEAR_MODULUS_U64: u64 = 0x7800_0001;
pub const BABY_BEAR_BITS: usize = 31;
pub const BABY_BEAR_EXT_DEGREE: usize = 4;
pub const BABY_BEAR_EXT_W_U64: u64 = 11;

#[derive(Clone, Debug)]
pub struct BabyBearVar {
    pub cell: AssignedValue<Fr>,
}

impl BabyBearVar {
    pub fn as_u64(&self) -> u64 {
        self.cell.value().get_lower_64()
    }
}

#[derive(Clone, Debug)]
pub struct BabyBearExtVar {
    pub coeffs: [BabyBearVar; BABY_BEAR_EXT_DEGREE],
}

impl BabyBearExtVar {
    pub fn as_u64(&self) -> [u64; BABY_BEAR_EXT_DEGREE] {
        core::array::from_fn(|i| self.coeffs[i].as_u64())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BabyBearArithmeticGadgets;

impl BabyBearArithmeticGadgets {
    fn modulus_fe() -> Fr {
        Fr::from(BABY_BEAR_MODULUS_U64)
    }

    fn assert_canonical(value: u64) {
        assert!(
            value < BABY_BEAR_MODULUS_U64,
            "BabyBear witness out of range: {value}"
        );
    }

    pub fn load_witness(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        value: u64,
    ) -> BabyBearVar {
        Self::assert_canonical(value);
        let cell = ctx.load_witness(Fr::from(value));
        range.check_less_than_safe(ctx, cell, BABY_BEAR_MODULUS_U64);
        BabyBearVar { cell }
    }

    pub fn load_constant(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        value: u64,
    ) -> BabyBearVar {
        Self::assert_canonical(value);
        let cell = ctx.load_constant(Fr::from(value));
        range.check_less_than_safe(ctx, cell, BABY_BEAR_MODULUS_U64);
        BabyBearVar { cell }
    }

    pub fn zero(&self, ctx: &mut Context<Fr>, range: &RangeChip<Fr>) -> BabyBearVar {
        self.load_constant(ctx, range, 0)
    }

    pub fn one(&self, ctx: &mut Context<Fr>, range: &RangeChip<Fr>) -> BabyBearVar {
        self.load_constant(ctx, range, 1)
    }

    pub fn load_ext_witness(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        coeffs: [u64; BABY_BEAR_EXT_DEGREE],
    ) -> BabyBearExtVar {
        BabyBearExtVar {
            coeffs: coeffs.map(|coeff| self.load_witness(ctx, range, coeff)),
        }
    }

    pub fn load_ext_constant(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        coeffs: [u64; BABY_BEAR_EXT_DEGREE],
    ) -> BabyBearExtVar {
        BabyBearExtVar {
            coeffs: coeffs.map(|coeff| self.load_constant(ctx, range, coeff)),
        }
    }

    pub fn ext_zero(&self, ctx: &mut Context<Fr>, range: &RangeChip<Fr>) -> BabyBearExtVar {
        self.load_ext_constant(ctx, range, [0; BABY_BEAR_EXT_DEGREE])
    }

    pub fn ext_one(&self, ctx: &mut Context<Fr>, range: &RangeChip<Fr>) -> BabyBearExtVar {
        self.load_ext_constant(ctx, range, [1, 0, 0, 0])
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, lhs: &BabyBearVar, rhs: &BabyBearVar) {
        ctx.constrain_equal(&lhs.cell, &rhs.cell);
    }

    pub fn assert_ext_equal(
        &self,
        ctx: &mut Context<Fr>,
        lhs: &BabyBearExtVar,
        rhs: &BabyBearExtVar,
    ) {
        for i in 0..BABY_BEAR_EXT_DEGREE {
            self.assert_equal(ctx, &lhs.coeffs[i], &rhs.coeffs[i]);
        }
    }

    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearVar,
        b: &BabyBearVar,
    ) -> BabyBearVar {
        let sum = a.as_u64() + b.as_u64();
        let q_u64 = sum / BABY_BEAR_MODULUS_U64;
        let out_u64 = sum % BABY_BEAR_MODULUS_U64;

        debug_assert!(q_u64 <= 1);
        let out = self.load_witness(ctx, range, out_u64);
        let q = ctx.load_witness(Fr::from(q_u64));
        range.range_check(ctx, q, 1);

        let gate = range.gate();
        let lhs = gate.add(ctx, a.cell, b.cell);
        let rhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), out.cell);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn sub(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearVar,
        b: &BabyBearVar,
    ) -> BabyBearVar {
        let (q_u64, out_u64) = if a.as_u64() >= b.as_u64() {
            (0, a.as_u64() - b.as_u64())
        } else {
            (1, a.as_u64() + BABY_BEAR_MODULUS_U64 - b.as_u64())
        };

        let out = self.load_witness(ctx, range, out_u64);
        let q = ctx.load_witness(Fr::from(q_u64));
        range.range_check(ctx, q, 1);

        let gate = range.gate();
        let lhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), a.cell);
        let rhs = gate.add(ctx, b.cell, out.cell);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearVar,
        b: &BabyBearVar,
    ) -> BabyBearVar {
        let prod = (a.as_u64() as u128) * (b.as_u64() as u128);
        let modulus = BABY_BEAR_MODULUS_U64 as u128;
        let q_u64 = (prod / modulus) as u64;
        let out_u64 = (prod % modulus) as u64;

        let out = self.load_witness(ctx, range, out_u64);
        let q = ctx.load_witness(Fr::from(q_u64));
        range.check_less_than_safe(ctx, q, BABY_BEAR_MODULUS_U64);

        let gate = range.gate();
        let lhs = gate.mul(ctx, a.cell, b.cell);
        let rhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), out.cell);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn square(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearVar,
    ) -> BabyBearVar {
        self.mul(ctx, range, a, a)
    }

    pub fn mul_const(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearVar,
        constant: u64,
    ) -> BabyBearVar {
        let constant = constant % BABY_BEAR_MODULUS_U64;
        let prod = (a.as_u64() as u128) * (constant as u128);
        let modulus = BABY_BEAR_MODULUS_U64 as u128;
        let q_u64 = (prod / modulus) as u64;
        let out_u64 = (prod % modulus) as u64;

        let out = self.load_witness(ctx, range, out_u64);
        let q = ctx.load_witness(Fr::from(q_u64));
        range.check_less_than_safe(ctx, q, BABY_BEAR_MODULUS_U64);

        let gate = range.gate();
        let lhs = gate.mul(ctx, a.cell, Constant(Fr::from(constant)));
        let rhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), out.cell);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn neg(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearVar,
    ) -> BabyBearVar {
        let zero = self.zero(ctx, range);
        self.sub(ctx, range, &zero, a)
    }

    fn add2(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: BabyBearVar,
        b: BabyBearVar,
    ) -> BabyBearVar {
        self.add(ctx, range, &a, &b)
    }

    fn add3(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: BabyBearVar,
        b: BabyBearVar,
        c: BabyBearVar,
    ) -> BabyBearVar {
        let ab = self.add2(ctx, range, a, b);
        self.add2(ctx, range, ab, c)
    }

    fn add4(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: BabyBearVar,
        b: BabyBearVar,
        c: BabyBearVar,
        d: BabyBearVar,
    ) -> BabyBearVar {
        let ab = self.add2(ctx, range, a, b);
        let cd = self.add2(ctx, range, c, d);
        self.add2(ctx, range, ab, cd)
    }

    pub fn ext_add(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearExtVar,
        b: &BabyBearExtVar,
    ) -> BabyBearExtVar {
        let coeffs = core::array::from_fn(|i| self.add(ctx, range, &a.coeffs[i], &b.coeffs[i]));
        BabyBearExtVar { coeffs }
    }

    pub fn ext_sub(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearExtVar,
        b: &BabyBearExtVar,
    ) -> BabyBearExtVar {
        let coeffs = core::array::from_fn(|i| self.sub(ctx, range, &a.coeffs[i], &b.coeffs[i]));
        BabyBearExtVar { coeffs }
    }

    pub fn ext_mul(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearExtVar,
        b: &BabyBearExtVar,
    ) -> BabyBearExtVar {
        let a0 = a.coeffs[0].clone();
        let a1 = a.coeffs[1].clone();
        let a2 = a.coeffs[2].clone();
        let a3 = a.coeffs[3].clone();
        let b0 = b.coeffs[0].clone();
        let b1 = b.coeffs[1].clone();
        let b2 = b.coeffs[2].clone();
        let b3 = b.coeffs[3].clone();

        let t0 = self.mul(ctx, range, &a0, &b0);

        let m01 = self.mul(ctx, range, &a0, &b1);
        let m10 = self.mul(ctx, range, &a1, &b0);
        let t1 = self.add2(ctx, range, m01, m10);

        let m02 = self.mul(ctx, range, &a0, &b2);
        let m11 = self.mul(ctx, range, &a1, &b1);
        let m20 = self.mul(ctx, range, &a2, &b0);
        let t2 = self.add3(ctx, range, m02, m11, m20);

        let m03 = self.mul(ctx, range, &a0, &b3);
        let m12 = self.mul(ctx, range, &a1, &b2);
        let m21 = self.mul(ctx, range, &a2, &b1);
        let m30 = self.mul(ctx, range, &a3, &b0);
        let t3 = self.add4(ctx, range, m03, m12, m21, m30);

        let m13 = self.mul(ctx, range, &a1, &b3);
        let m22 = self.mul(ctx, range, &a2, &b2);
        let m31 = self.mul(ctx, range, &a3, &b1);
        let t4 = self.add3(ctx, range, m13, m22, m31);

        let m23 = self.mul(ctx, range, &a2, &b3);
        let m32 = self.mul(ctx, range, &a3, &b2);
        let t5 = self.add2(ctx, range, m23, m32);

        let t6 = self.mul(ctx, range, &a3, &b3);

        let wt4 = self.mul_const(ctx, range, &t4, BABY_BEAR_EXT_W_U64);
        let wt5 = self.mul_const(ctx, range, &t5, BABY_BEAR_EXT_W_U64);
        let wt6 = self.mul_const(ctx, range, &t6, BABY_BEAR_EXT_W_U64);

        BabyBearExtVar {
            coeffs: [
                self.add2(ctx, range, t0, wt4),
                self.add2(ctx, range, t1, wt5),
                self.add2(ctx, range, t2, wt6),
                t3,
            ],
        }
    }

    pub fn ext_square(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        a: &BabyBearExtVar,
    ) -> BabyBearExtVar {
        self.ext_mul(ctx, range, a, a)
    }

}

#[cfg(test)]
mod tests;
