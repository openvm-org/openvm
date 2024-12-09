use std::sync::Arc;

use ax_stark_backend::p3_field::{
    extension::{BinomialExtensionField, BinomiallyExtendable},
    AbstractExtensionField, AbstractField, Field, PrimeField32, PrimeField64,
};
use ax_stark_sdk::p3_baby_bear::BabyBear;
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use snark_verifier_sdk::snark_verifier::{
    halo2_base::{
        gates::{GateChip, GateInstructions, RangeChip, RangeInstructions},
        halo2_proofs::halo2curves::bn256::Fr,
        utils::{bigint_to_fe, biguint_to_fe, bit_length, fe_to_bigint, BigPrimeField},
        AssignedValue, Context, QuantumCell,
    },
    util::arithmetic::{Field as _, PrimeField as _},
};

pub(crate) const BABYBEAR_MAX_BITS: usize = 31;

#[derive(Copy, Clone, Debug)]
pub struct AssignedBabyBear {
    pub value: AssignedValue<Fr>,
    /// The value is guaranteed to be less than 2^max_bits.
    pub max_bits: usize,
}

impl AssignedBabyBear {
    pub fn to_baby_bear(&self) -> BabyBear {
        let mut b_int = fe_to_bigint(self.value.value()) % BabyBear::ORDER_U32;
        if b_int < BigInt::from(0) {
            b_int += BabyBear::ORDER_U32;
        }
        BabyBear::from_canonical_u32(b_int.try_into().unwrap())
    }
}

pub struct BabyBearChip {
    pub range: Arc<RangeChip<Fr>>,
}

impl BabyBearChip {
    pub fn new(range_chip: Arc<RangeChip<Fr>>) -> Self {
        BabyBearChip { range: range_chip }
    }

    pub fn gate(&self) -> &GateChip<Fr> {
        self.range.gate()
    }

    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBear) -> AssignedBabyBear {
        let value = ctx.load_witness(Fr::from(PrimeField64::as_canonical_u64(&value)));
        self.range.range_check(ctx, value, BABYBEAR_MAX_BITS);
        AssignedBabyBear {
            value,
            max_bits: BABYBEAR_MAX_BITS,
        }
    }

    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: BabyBear) -> AssignedBabyBear {
        let max_bits = bit_length(value.as_canonical_u64());
        let value = ctx.load_constant(Fr::from(PrimeField64::as_canonical_u64(&value)));
        AssignedBabyBear { value, max_bits }
    }

    pub fn reduce(&self, ctx: &mut Context<Fr>, a: AssignedBabyBear) -> AssignedBabyBear {
        let (_, r) = signed_div_mod(&self.range, ctx, a.value, BabyBear::ORDER_U32, a.max_bits);
        let r = AssignedBabyBear {
            value: r,
            max_bits: BABYBEAR_MAX_BITS,
        };
        debug_assert_eq!(a.to_baby_bear(), r.to_baby_bear());
        r
    }

    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBear,
        b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        let value = self.gate().add(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;

        let mut c = AssignedBabyBear { value, max_bits };
        if c.max_bits >= Fr::CAPACITY as usize - 1 {
            c = self.reduce(ctx, c);
        }
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() + b.to_baby_bear());
        c
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: AssignedBabyBear) -> AssignedBabyBear {
        let value = self.gate().neg(ctx, a.value);
        let b = AssignedBabyBear {
            value,
            max_bits: a.max_bits,
        };
        debug_assert_eq!(b.to_baby_bear(), -a.to_baby_bear());
        b
    }

    pub fn sub(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBear,
        b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        let value = self.gate().sub(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;

        let mut c = AssignedBabyBear { value, max_bits };
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() - b.to_baby_bear());
        if c.max_bits >= Fr::CAPACITY as usize - 1 {
            c = self.reduce(ctx, c);
        }
        c
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        mut a: AssignedBabyBear,
        mut b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        if a.max_bits < b.max_bits {
            std::mem::swap(&mut a, &mut b);
        }
        if a.max_bits + b.max_bits >= Fr::CAPACITY as usize - 1 {
            a = self.reduce(ctx, a);
            if a.max_bits + b.max_bits >= Fr::CAPACITY as usize - 1 {
                b = self.reduce(ctx, b);
            }
        }
        let value = self.gate().mul(ctx, a.value, b.value);
        let max_bits = a.max_bits + b.max_bits;

        let mut c = AssignedBabyBear { value, max_bits };
        if c.max_bits >= Fr::CAPACITY as usize - 1 {
            c = self.reduce(ctx, c);
        }
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() * b.to_baby_bear());
        c
    }

    pub fn div(
        &self,
        ctx: &mut Context<Fr>,
        mut a: AssignedBabyBear,
        mut b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        let b_val = b.to_baby_bear();
        let b_inv = b_val.try_inverse().unwrap();

        let mut c = self.load_witness(ctx, a.to_baby_bear() * b_inv);
        // constraint a = b * c (mod p)
        if a.max_bits >= Fr::CAPACITY as usize - 1 {
            a = self.reduce(ctx, a);
        }
        if b.max_bits + c.max_bits >= Fr::CAPACITY as usize - 1 {
            b = self.reduce(ctx, b);
        }
        if b.max_bits + c.max_bits >= Fr::CAPACITY as usize - 1 {
            c = self.reduce(ctx, c);
        }
        let diff = self.gate().sub_mul(ctx, a.value, b.value, c.value);
        let max_bits = a.max_bits.max(b.max_bits + c.max_bits) + 1;
        let (_, r) = signed_div_mod(&self.range, ctx, diff, BabyBear::ORDER_U32, max_bits);
        self.gate().assert_is_const(ctx, &r, &Fr::ZERO);
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() / b.to_baby_bear());
        c
    }

    pub fn select(
        &self,
        ctx: &mut Context<Fr>,
        cond: AssignedValue<Fr>,
        a: AssignedBabyBear,
        b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        let value = self.gate().select(ctx, a.value, b.value, cond);
        let max_bits = a.max_bits.max(b.max_bits);
        AssignedBabyBear { value, max_bits }
    }

    pub fn assert_zero(&self, ctx: &mut Context<Fr>, a: AssignedBabyBear) {
        debug_assert_eq!(a.to_baby_bear(), BabyBear::ZERO);
        assert!(a.max_bits < Fr::CAPACITY as usize);
        let (_, r) = signed_div_mod(&self.range, ctx, a.value, BabyBear::ORDER_U32, a.max_bits);
        self.gate().assert_is_const(ctx, &r, &Fr::ZERO);
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, a: AssignedBabyBear, b: AssignedBabyBear) {
        debug_assert_eq!(a.to_baby_bear(), b.to_baby_bear());
        let diff = self.sub(ctx, a, b);
        self.assert_zero(ctx, diff);
    }
}

/// Constrains and returns `(c, r)` such that `a = b * c + r`.
///
/// * a: [QuantumCell] value to divide
/// * b: [BigUint] value to divide by
/// * a_num_bits: number of bits needed to represent the absolute value of `a`
///
/// ## Assumptions
/// * `b != 0` and that `abs(a) < 2^a_max_bits`
/// * `a_max_bits < F::CAPACITY = F::NUM_BITS - 1`
///   * Unsafe behavior if `a_max_bits >= F::CAPACITY`
fn signed_div_mod<F>(
    range: &RangeChip<F>,
    ctx: &mut Context<F>,
    a: impl Into<QuantumCell<F>>,
    b: impl Into<BigUint>,
    a_num_bits: usize,
) -> (AssignedValue<F>, AssignedValue<F>)
where
    F: BigPrimeField,
{
    let a = a.into();
    let b = b.into();
    let a_val = fe_to_bigint(a.value());
    assert!(a_val.bits() <= a_num_bits as u64);
    let (div, rem) = a_val.div_mod_floor(&b.clone().into());
    let [div, rem] = [div, rem].map(|v| bigint_to_fe(&v));
    ctx.assign_region(
        [
            QuantumCell::Witness(rem),
            QuantumCell::Constant(biguint_to_fe(&b)),
            QuantumCell::Witness(div),
            a,
        ],
        [0],
    );
    let rem = ctx.get(-4);
    let div = ctx.get(-2);
    // Constrain that `abs(div) <= 2 ** a_num_bits / b`.
    let bound = (BigUint::from(1u32) << (a_num_bits as u32)) / &b;
    let shifted_div = range
        .gate()
        .add(ctx, div, QuantumCell::Constant(biguint_to_fe(&bound)));
    debug_assert!(*shifted_div.value() < biguint_to_fe(&(&bound * 2u32 + 1u32)));
    range.check_big_less_than_safe(ctx, shifted_div, bound * 2u32 + 1u32);
    // Constrain that remainder is less than divisor (i.e. `r < b`).
    debug_assert!(*rem.value() < biguint_to_fe(&b));
    range.check_big_less_than_safe(ctx, rem, b);
    (div, rem)
}

// irred poly is x^4 - 11
pub struct BabyBearExt4Chip {
    pub base: Arc<BabyBearChip>,
}

#[derive(Copy, Clone, Debug)]
pub struct AssignedBabyBearExt4(pub [AssignedBabyBear; 4]);
pub type BabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

impl AssignedBabyBearExt4 {
    pub fn to_extension_field(&self) -> BabyBearExt4 {
        let b_val = (0..4).map(|i| self.0[i].to_baby_bear()).collect_vec();
        BabyBearExt4::from_base_slice(&b_val)
    }
}

impl BabyBearExt4Chip {
    pub fn new(base_chip: Arc<BabyBearChip>) -> Self {
        BabyBearExt4Chip { base: base_chip }
    }
    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBearExt4) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            value
                .as_base_slice()
                .iter()
                .map(|x| self.base.load_witness(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }
    pub fn load_constant(
        &self,
        ctx: &mut Context<Fr>,
        value: BabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            value
                .as_base_slice()
                .iter()
                .map(|x| self.base.load_constant(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }
    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.add(ctx, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: AssignedBabyBearExt4) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            a.0.iter()
                .map(|x| self.base.neg(ctx, *x))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn sub(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.sub(ctx, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn scalar_mul(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBear,
    ) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            a.0.iter()
                .map(|x| self.base.mul(ctx, *x, b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn select(
        &self,
        ctx: &mut Context<Fr>,
        cond: AssignedValue<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            a.0.iter()
                .zip(b.0.iter())
                .map(|(a, b)| self.base.select(ctx, cond, *a, *b))
                .collect_vec()
                .try_into()
                .unwrap(),
        )
    }

    pub fn assert_zero(&self, ctx: &mut Context<Fr>, a: AssignedBabyBearExt4) {
        for x in a.0.iter() {
            self.base.assert_zero(ctx, *x);
        }
    }

    pub fn assert_equal(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBearExt4,
    ) {
        for (a, b) in a.0.iter().zip(b.0.iter()) {
            self.base.assert_equal(ctx, *a, *b);
        }
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        let mut coeffs = Vec::with_capacity(7);
        for i in 0..4 {
            for j in 0..4 {
                if i + j < coeffs.len() {
                    let tmp = self.base.mul(ctx, a.0[i], b.0[j]);
                    coeffs[i + j] = self.base.add(ctx, coeffs[i + j], tmp);
                } else {
                    coeffs.push(self.base.mul(ctx, a.0[i], b.0[j]));
                }
            }
        }
        let w = self
            .base
            .load_constant(ctx, <BabyBear as BinomiallyExtendable<4>>::W);
        for i in 4..7 {
            let tmp = self.base.mul(ctx, coeffs[i], w);
            coeffs[i - 4] = self.base.add(ctx, coeffs[i - 4], tmp);
        }
        coeffs.truncate(4);
        let c = AssignedBabyBearExt4(coeffs.try_into().unwrap());
        debug_assert_eq!(
            c.to_extension_field(),
            a.to_extension_field() * b.to_extension_field()
        );
        c
    }

    pub fn div(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
        b: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        let b_val = b.to_extension_field();
        let b_inv = b_val.try_inverse().unwrap();

        let c = self.load_witness(ctx, a.to_extension_field() * b_inv);
        // constraint a = b * c
        let prod = self.mul(ctx, b, c);
        self.assert_equal(ctx, a, prod);

        debug_assert_eq!(
            c.to_extension_field(),
            a.to_extension_field() / b.to_extension_field()
        );
        c
    }
}
