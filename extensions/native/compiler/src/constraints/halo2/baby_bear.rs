use std::sync::Arc;

use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use openvm_stark_backend::p3_field::{
    extension::{BinomialExtensionField, BinomiallyExtendable},
    Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField32, PrimeField64,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
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
// bits reserved so that if we do lazy range checking, we still have a valid result
// the first reserved bit is so that we can represent negative numbers
// the second is to accommodate lazy range checking
const RESERVED_HIGH_BITS: usize = 2;

#[derive(Copy, Clone, Debug)]
pub struct AssignedBabyBear {
    /// Logically `value` is a signed integer represented as `Bn254Fr`.
    /// Invariants:
    /// - `|value|` never overflows `Bn254Fr`
    /// - `|value| < 2^max_bits` and `max_bits <= Fr::CAPACITY - RESERVED_HIGH_BITS`
    ///
    /// Basically `value` could do arithmetic operations without extra constraints as long as the
    /// result doesn't overflow `Bn254Fr`. And it's easy to track `max_bits` of the result.
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
        debug_assert!(fe_to_bigint(a.value.value()).bits() as usize <= a.max_bits);
        let (_, r) = signed_div_mod(&self.range, ctx, a.value, a.max_bits);
        let r = AssignedBabyBear {
            value: r,
            max_bits: BABYBEAR_MAX_BITS,
        };
        debug_assert_eq!(a.to_baby_bear(), r.to_baby_bear());
        r
    }

    /// Reduce max_bits if possible. This function doesn't guarantee that the actual value is within
    /// BabyBear.
    pub fn reduce_max_bits(&self, ctx: &mut Context<Fr>, a: AssignedBabyBear) -> AssignedBabyBear {
        if a.max_bits > BABYBEAR_MAX_BITS {
            self.reduce(ctx, a)
        } else {
            a
        }
    }

    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        mut a: AssignedBabyBear,
        mut b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
            if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
                b = self.reduce(ctx, b);
            }
        }
        let value = self.gate().add(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let mut c = AssignedBabyBear { value, max_bits };
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() + b.to_baby_bear());
        if c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c);
        }
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
        mut a: AssignedBabyBear,
        mut b: AssignedBabyBear,
    ) -> AssignedBabyBear {
        if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
            if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
                b = self.reduce(ctx, b);
            }
        }
        let value = self.gate().sub(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let mut c = AssignedBabyBear { value, max_bits };
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() - b.to_baby_bear());
        if c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
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
        if a.max_bits + b.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
            if a.max_bits + b.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
                b = self.reduce(ctx, b);
            }
        }
        let value = self.gate().mul(ctx, a.value, b.value);
        let max_bits = a.max_bits + b.max_bits;

        let mut c = AssignedBabyBear { value, max_bits };
        if c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c);
        }
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() * b.to_baby_bear());
        c
    }

    pub fn mul_add(
        &self,
        ctx: &mut Context<Fr>,
        mut a: AssignedBabyBear,
        mut b: AssignedBabyBear,
        mut c: AssignedBabyBear,
    ) -> AssignedBabyBear {
        if a.max_bits < b.max_bits {
            std::mem::swap(&mut a, &mut b);
        }
        if a.max_bits + b.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
            if a.max_bits + b.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
                b = self.reduce(ctx, b);
            }
        }
        if c.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c)
        }
        let value = self.gate().mul_add(ctx, a.value, b.value, c.value);
        let max_bits = c.max_bits.max(a.max_bits + b.max_bits) + 1;

        let mut d = AssignedBabyBear { value, max_bits };
        if d.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            d = self.reduce(ctx, d);
        }
        debug_assert_eq!(
            d.to_baby_bear(),
            a.to_baby_bear() * b.to_baby_bear() + c.to_baby_bear()
        );
        d
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
        if a.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
        }
        if b.max_bits + c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            b = self.reduce(ctx, b);
        }
        if b.max_bits + c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c);
        }
        let diff = self.gate().sub_mul(ctx, a.value, b.value, c.value);
        let max_bits = a.max_bits.max(b.max_bits + c.max_bits) + 1;
        self.assert_zero(
            ctx,
            AssignedBabyBear {
                value: diff,
                max_bits,
            },
        );
        debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() / b.to_baby_bear());
        c
    }

    // This inner product function will be used exclusively for optimizing extension element
    // multiplication.
    fn special_inner_product(
        &self,
        ctx: &mut Context<Fr>,
        a: &mut [AssignedBabyBear],
        b: &mut [AssignedBabyBear],
        s: usize,
    ) -> AssignedBabyBear {
        assert!(a.len() == b.len());
        assert!(a.len() == 4);
        let mut max_bits = 0;
        let lb = s.saturating_sub(3);
        let ub = 4.min(s + 1);
        let range = lb..ub;
        let other_range = (s + 1 - ub)..(s + 1 - lb);
        let len = if s < 3 { s + 1 } else { 7 - s };
        for (i, (c, d)) in a[range.clone()]
            .iter_mut()
            .zip(b[other_range.clone()].iter_mut().rev())
            .enumerate()
        {
            if c.max_bits + d.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS - len + i {
                if c.max_bits >= d.max_bits {
                    *c = self.reduce(ctx, *c);
                    if c.max_bits + d.max_bits
                        > Fr::CAPACITY as usize - RESERVED_HIGH_BITS - len + i
                    {
                        *d = self.reduce(ctx, *d);
                    }
                } else {
                    *d = self.reduce(ctx, *d);
                    if c.max_bits + d.max_bits
                        > Fr::CAPACITY as usize - RESERVED_HIGH_BITS - len + i
                    {
                        *c = self.reduce(ctx, *c);
                    }
                }
            }
            if i == 0 {
                max_bits = c.max_bits + d.max_bits;
            } else {
                max_bits = max_bits.max(c.max_bits + d.max_bits) + 1
            }
        }
        let a_raw = a[range]
            .iter()
            .map(|a| QuantumCell::Existing(a.value))
            .collect_vec();
        let b_raw = b[other_range]
            .iter()
            .rev()
            .map(|b| QuantumCell::Existing(b.value))
            .collect_vec();
        let prod = self.gate().inner_product(ctx, a_raw, b_raw);
        AssignedBabyBear {
            value: prod,
            max_bits,
        }
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
        // The proof of correctness of this function is listed in `signed_div_mod`.
        debug_assert_eq!(a.to_baby_bear(), BabyBear::ZERO);
        assert!(a.max_bits <= Fr::CAPACITY as usize - RESERVED_HIGH_BITS);
        let a_num_bits = a.max_bits;
        let b: BigUint = BabyBear::ORDER_U32.into();
        let a_val = fe_to_bigint(a.value.value());
        assert!(a_val.bits() <= a_num_bits as u64);
        let (div, _) = a_val.div_mod_floor(&b.clone().into());
        let div = bigint_to_fe(&div);
        ctx.assign_region(
            [
                QuantumCell::Constant(Fr::ZERO),
                QuantumCell::Constant(biguint_to_fe(&b)),
                QuantumCell::Witness(div),
                a.value.into(),
            ],
            [0],
        );
        let div = ctx.get(-2);
        // Constrain that `abs(div) <= 2 ** (2 ** a_num_bits / b).bits()`.
        let bound = (BigUint::from(1u32) << (a_num_bits as u32)) / &b;
        let shifted_div =
            self.range
                .gate()
                .add(ctx, div, QuantumCell::Constant(biguint_to_fe(&bound)));
        debug_assert!(*shifted_div.value() < biguint_to_fe(&(&bound * 2u32 + 1u32)));
        self.range
            .range_check(ctx, shifted_div, (bound * 2u32 + 1u32).bits() as usize);
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, a: AssignedBabyBear, b: AssignedBabyBear) {
        debug_assert_eq!(a.to_baby_bear(), b.to_baby_bear());
        let diff = self.sub(ctx, a, b);
        self.assert_zero(ctx, diff);
    }
}

/// Constrains and returns `(c, r)` such that `a = BabyBear::ORDER_U32 * c + r`.
///
/// * a: [QuantumCell] value to divide
/// * a_num_bits: number of bits needed to represent the absolute value of `a`
///
/// ## Assumptions
/// * `a_max_bits < F::CAPACITY = F::NUM_BITS - RESERVED_HIGH_BITS`
///   * Unsafe behavior if `a_max_bits >= F::CAPACITY`
fn signed_div_mod<F>(
    range: &RangeChip<F>,
    ctx: &mut Context<F>,
    a: impl Into<QuantumCell<F>>,
    a_num_bits: usize,
) -> (AssignedValue<F>, AssignedValue<F>)
where
    F: BigPrimeField,
{
    // Proof of correctness:
    // Let `b` be the order of `BabyBear` and `p` be the order of `Fr`.
    // First we introduce witness `div` and `rem`.
    // We constraint:
    // (1) `div * b + rem ≡ a (mod p)`
    // (2) `0 <= rem < b`
    // Logically we want `div = a // b`. Because (2) and `a` could be negative, `div` could
    // be negative. Therefore, we have `|div| = |a // b| = |a| // b < 2^max_bits // b = bound` and
    // we can say `shifted_div = div + bound` is in `[0, 2 * bound)`.
    // In practice, it's expensive to assert `shifted_div` is less than `2 * bound` which is not a
    // power of 2s. Instead, we add a looser constraint:
    // (3) `shifted_div < 2^max_bits/2^(BABYBEAR_ORDER_BITS-1)=2^(max_bits-BABYBEAR_ORDER_BITS+1)`
    //
    // Let's check if |div * b + rem| can overflow:
    // - `div` has at most `max_bits-BABYBEAR_ORDER_BITS` bits
    // - `b` has `BABYBEAR_ORDER_BITS` bits.
    // - `rem` has at most `BABYBEAR_ORDER_BITS` bits.
    // When `max_bits > BABYBEAR_ORDER_BITS`, `|div * b + rem|` has at most `max_bits+1` bits.
    // Because of the invariant `max_bits <= Fr::CAPACITY - RESERVED_HIGH_BITS`, `|div * b + rem|`
    // cannot overflow.
    //
    // Let's check if the looser constraint will cause some problem:
    // Assume there are other `div'` and `rem'` satisfying:
    // `div * b + rem ≡ div' * b + rem' (mod p)`
    // Then we have:
    // `(div - div') * b ≡ rem' - rem (mod p)`
    // (3) => `|(div - div') * b| < 2^(max_bits+1) < p`
    // (2) => `|rem' - rem| < b`
    // There could be 3 cases:
    // a. `-b < (div - div') * b < b` or;
    // b. `0 < (div - div') * b + p < b` or;
    // c. `-b < (div - div') * b - p < 0`
    // Case (a) is impossible because `div != div'`.
    // Case (b) and (c) imply:
    // |div - div'|  > (p-b) // b > 2^(Fr::CAPACITY - (BABYBEAR_ORDER_BITS - 1) - 1) = 2^(Fr::CAPACITY - BABYBEAR_ORDER_BITS)
    // (3) also constrains that this is impossible.
    let a = a.into();
    let b = BigUint::from(BabyBear::ORDER_U32);
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
    // Constrain that `abs(div) <= 2 ** (2 ** a_num_bits / b).bits()`.
    let bound = (BigUint::from(1u32) << (a_num_bits as u32)) / &b;
    let shifted_div = range
        .gate()
        .add(ctx, div, QuantumCell::Constant(biguint_to_fe(&bound)));
    debug_assert!(*shifted_div.value() < biguint_to_fe(&(&bound * 2u32 + 1u32)));
    range.range_check(ctx, shifted_div, (bound * 2u32 + 1u32).bits() as usize);
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
        mut a: AssignedBabyBearExt4,
        mut b: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        let mut coeffs = Vec::with_capacity(7);
        for s in 0..7 {
            coeffs.push(self.base.special_inner_product(ctx, &mut a.0, &mut b.0, s));
        }
        let w = self
            .base
            .load_constant(ctx, <BabyBear as BinomiallyExtendable<4>>::W);
        for i in 4..7 {
            coeffs[i - 4] = self.base.mul_add(ctx, coeffs[i], w, coeffs[i - 4]);
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

    pub fn reduce_max_bits(
        &self,
        ctx: &mut Context<Fr>,
        a: AssignedBabyBearExt4,
    ) -> AssignedBabyBearExt4 {
        AssignedBabyBearExt4(
            a.0.into_iter()
                .map(|x| self.base.reduce_max_bits(ctx, x))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}
