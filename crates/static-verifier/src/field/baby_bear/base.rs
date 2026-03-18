use halo2_base::{
    gates::{range::RangeChip, GateChip, GateInstructions, RangeInstructions},
    utils::ScalarField,
    AssignedValue, Context,
    QuantumCell::Constant,
};
use openvm_stark_sdk::openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::{
    utils::{bits_for_u64, usize_to_u64},
    ChildF, Fr,
};

pub const BABY_BEAR_MODULUS_U64: u64 = 0x7800_0001;
pub const BABY_BEAR_BITS: usize = 31;
pub const BABY_BEAR_EXT_DEGREE: usize = 4;
pub const BABY_BEAR_EXT_W_U64: u64 = 11;

#[derive(Copy, Clone, Debug)]
pub struct BabyBearWire(pub AssignedValue<Fr>);

impl BabyBearWire {
    pub fn as_u64(&self) -> u64 {
        self.0.value().get_lower_64()
    }

    pub fn value(&self) -> ChildF {
        ChildF::from_u64(self.0.value().get_lower_64())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BabyBearChip<'a> {
    range: &'a RangeChip<Fr>,
}

impl<'a> BabyBearChip<'a> {
    pub fn new(range: &'a RangeChip<Fr>) -> Self {
        Self { range }
    }

    pub fn range(&self) -> &RangeChip<Fr> {
        self.range
    }

    pub fn gate(&self) -> &GateChip<Fr> {
        self.range.gate()
    }

    fn modulus_fe() -> Fr {
        Fr::from(BABY_BEAR_MODULUS_U64)
    }

    fn assert_canonical(value: u64) {
        assert!(
            value < BABY_BEAR_MODULUS_U64,
            "BabyBear witness out of range: {value}"
        );
    }

    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: ChildF) -> BabyBearWire {
        let value_u64 = value.as_canonical_u64();
        Self::assert_canonical(value_u64);
        let cell = ctx.load_witness(Fr::from(value_u64));
        self.range
            .check_less_than_safe(ctx, cell, BABY_BEAR_MODULUS_U64);
        BabyBearWire(cell)
    }

    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: ChildF) -> BabyBearWire {
        let value_u64 = value.as_canonical_u64();
        Self::assert_canonical(value_u64);
        let cell = ctx.load_constant(Fr::from(value_u64));
        BabyBearWire(cell)
    }

    pub fn zero(&self, ctx: &mut Context<Fr>) -> BabyBearWire {
        self.load_constant(ctx, ChildF::ZERO)
    }

    pub fn one(&self, ctx: &mut Context<Fr>) -> BabyBearWire {
        self.load_constant(ctx, ChildF::ONE)
    }

    pub fn add(&self, ctx: &mut Context<Fr>, a: &BabyBearWire, b: &BabyBearWire) -> BabyBearWire {
        let sum = a.as_u64() + b.as_u64();
        let q_u64 = sum / BABY_BEAR_MODULUS_U64;
        let out_u64 = sum % BABY_BEAR_MODULUS_U64;

        debug_assert!(q_u64 <= 1);
        let out = self.load_witness(ctx, ChildF::from_u64(out_u64));
        let q = ctx.load_witness(Fr::from(q_u64));
        self.range.gate().assert_bit(ctx, q);

        let gate = self.gate();
        let lhs = gate.add(ctx, a.0, b.0);
        let rhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), out.0);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn sub(&self, ctx: &mut Context<Fr>, a: &BabyBearWire, b: &BabyBearWire) -> BabyBearWire {
        let (q_u64, out_u64) = if a.as_u64() >= b.as_u64() {
            (0, a.as_u64() - b.as_u64())
        } else {
            (1, a.as_u64() + BABY_BEAR_MODULUS_U64 - b.as_u64())
        };

        let out = self.load_witness(ctx, ChildF::from_u64(out_u64));
        let q = ctx.load_witness(Fr::from(q_u64));
        self.range.gate().assert_bit(ctx, q);

        let gate = self.gate();
        let lhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), a.0);
        let rhs = gate.add(ctx, b.0, out.0);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn mul(&self, ctx: &mut Context<Fr>, a: &BabyBearWire, b: &BabyBearWire) -> BabyBearWire {
        let prod = (a.as_u64() as u128) * (b.as_u64() as u128);
        let modulus = BABY_BEAR_MODULUS_U64 as u128;
        let q_u64 = (prod / modulus) as u64;
        let out_u64 = (prod % modulus) as u64;

        let out = self.load_witness(ctx, ChildF::from_u64(out_u64));
        let q = ctx.load_witness(Fr::from(q_u64));
        self.range
            .check_less_than_safe(ctx, q, BABY_BEAR_MODULUS_U64);

        let gate = self.gate();
        let lhs = gate.mul(ctx, a.0, b.0);
        let rhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), out.0);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn mul_const(
        &self,
        ctx: &mut Context<Fr>,
        a: &BabyBearWire,
        constant: ChildF,
    ) -> BabyBearWire {
        let constant_u64 = constant.as_canonical_u64() % BABY_BEAR_MODULUS_U64;
        let prod = (a.as_u64() as u128) * (constant_u64 as u128);
        let modulus = BABY_BEAR_MODULUS_U64 as u128;
        let q_u64 = (prod / modulus) as u64;
        let out_u64 = (prod % modulus) as u64;

        let out = self.load_witness(ctx, ChildF::from_u64(out_u64));
        let q = ctx.load_witness(Fr::from(q_u64));
        self.range
            .check_less_than_safe(ctx, q, BABY_BEAR_MODULUS_U64);

        let gate = self.gate();
        let lhs = gate.mul(ctx, a.0, Constant(Fr::from(constant_u64)));
        let rhs = gate.mul_add(ctx, q, Constant(Self::modulus_fe()), out.0);
        ctx.constrain_equal(&lhs, &rhs);

        out
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: &BabyBearWire) -> BabyBearWire {
        let zero = self.zero(ctx);
        self.sub(ctx, &zero, a)
    }

    pub fn square(&self, ctx: &mut Context<Fr>, a: &BabyBearWire) -> BabyBearWire {
        self.mul(ctx, a, a)
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, lhs: &BabyBearWire, rhs: &BabyBearWire) {
        ctx.constrain_equal(&lhs.0, &rhs.0);
    }

    pub(super) fn add2(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearWire,
        b: BabyBearWire,
    ) -> BabyBearWire {
        self.add(ctx, &a, &b)
    }

    pub(super) fn add3(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearWire,
        b: BabyBearWire,
        c: BabyBearWire,
    ) -> BabyBearWire {
        let ab = self.add2(ctx, a, b);
        self.add2(ctx, ab, c)
    }

    pub(super) fn add4(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearWire,
        b: BabyBearWire,
        c: BabyBearWire,
        d: BabyBearWire,
    ) -> BabyBearWire {
        let ab = self.add2(ctx, a, b);
        let cd = self.add2(ctx, c, d);
        self.add2(ctx, ab, cd)
    }

    pub fn assign_and_range_u64(&self, ctx: &mut Context<Fr>, value: u64) -> AssignedValue<Fr> {
        let cell = ctx.load_witness(Fr::from(value));
        self.range.range_check(ctx, cell, bits_for_u64(value));
        cell
    }

    pub fn assign_and_range_usize(&self, ctx: &mut Context<Fr>, value: usize) -> AssignedValue<Fr> {
        self.assign_and_range_u64(ctx, usize_to_u64(value))
    }
}
