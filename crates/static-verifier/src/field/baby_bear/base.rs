use halo2_base::{
    gates::{range::RangeChip, GateChip, GateInstructions, RangeInstructions},
    AssignedValue, Context,
    QuantumCell::Constant,
};
use num_bigint::BigUint;
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

    fn modulus_biguint() -> BigUint {
        BigUint::from(BABY_BEAR_MODULUS_U64)
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
        let sum = self.gate().add(ctx, a.0, b.0);
        let (q, out) = self
            .range
            .div_mod(ctx, sum, Self::modulus_biguint(), BABY_BEAR_BITS + 1);
        self.range.gate().assert_bit(ctx, q);
        BabyBearWire(out)
    }

    pub fn sub(&self, ctx: &mut Context<Fr>, a: &BabyBearWire, b: &BabyBearWire) -> BabyBearWire {
        let lifted = self.gate().add(ctx, a.0, Constant(Self::modulus_fe()));
        let diff = self.gate().sub(ctx, lifted, b.0);
        let (q, out) = self
            .range
            .div_mod(ctx, diff, Self::modulus_biguint(), BABY_BEAR_BITS + 1);
        self.range.gate().assert_bit(ctx, q);
        BabyBearWire(out)
    }

    pub fn mul(&self, ctx: &mut Context<Fr>, a: &BabyBearWire, b: &BabyBearWire) -> BabyBearWire {
        let prod = self.gate().mul(ctx, a.0, b.0);
        let (q, out) = self
            .range
            .div_mod(ctx, prod, Self::modulus_biguint(), 2 * BABY_BEAR_BITS);
        self.range
            .check_less_than_safe(ctx, q, BABY_BEAR_MODULUS_U64);
        BabyBearWire(out)
    }

    pub fn mul_const(
        &self,
        ctx: &mut Context<Fr>,
        a: &BabyBearWire,
        constant: ChildF,
    ) -> BabyBearWire {
        let constant_u64 = constant.as_canonical_u64() % BABY_BEAR_MODULUS_U64;
        let prod = self.gate().mul(ctx, a.0, Constant(Fr::from(constant_u64)));
        let (q, out) = self
            .range
            .div_mod(ctx, prod, Self::modulus_biguint(), 2 * BABY_BEAR_BITS);
        self.range
            .check_less_than_safe(ctx, q, BABY_BEAR_MODULUS_U64);
        BabyBearWire(out)
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: &BabyBearWire) -> BabyBearWire {
        let zero = self.zero(ctx);
        self.sub(ctx, &zero, a)
    }

    pub fn square(&self, ctx: &mut Context<Fr>, a: &BabyBearWire) -> BabyBearWire {
        self.mul(ctx, a, a)
    }

    pub fn invert(&self, ctx: &mut Context<Fr>, value: &BabyBearWire) -> BabyBearWire {
        let is_zero = self.gate().is_zero(ctx, value.0);
        self.gate().assert_is_const(ctx, &is_zero, &Fr::from(0u64));

        let mut acc = self.one(ctx);
        let mut base = *value;
        let mut exp = BABY_BEAR_MODULUS_U64 - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(ctx, &acc, &base);
            }
            exp >>= 1;
            if exp > 0 {
                base = self.square(ctx, &base);
            }
        }
        acc
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
