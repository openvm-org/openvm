use std::{cell::RefCell, collections::HashMap, sync::Arc};

use halo2_base::{
    gates::{GateChip, GateInstructions, RangeChip, RangeInstructions},
    halo2_proofs::{
        arithmetic::Field as _,
        halo2curves::{bn256::Fr, ff::PrimeField as _},
    },
    utils::{bigint_to_fe, biguint_to_fe, bit_length, fe_to_bigint, BigPrimeField},
    AssignedValue, Context, QuantumCell,
};
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{Field, PrimeCharacteristicRing, PrimeField32, PrimeField64},
    p3_baby_bear::BabyBear,
};

use crate::utils::{guarded_debug_assert, guarded_debug_assert_eq};

pub(crate) const BABYBEAR_MAX_BITS: usize = 31;
// bits reserved so that if we do lazy range checking, we still have a valid result
// the first reserved bit is so that we can represent negative numbers
// the second is to accommodate lazy range checking
const RESERVED_HIGH_BITS: usize = 2;

#[derive(Copy, Clone, Debug)]
pub struct BabyBearWire {
    /// Logically `value` is a signed integer represented as `Bn254`.
    /// Invariants:
    /// - `|value|` never overflows `Bn254`
    /// - `|value| < 2^max_bits` and `max_bits <= Fr::CAPACITY - RESERVED_HIGH_BITS`
    ///
    /// Basically `value` could do arithmetic operations without extra constraints as long as the
    /// result doesn't overflow `Bn254`. And it's easy to track `max_bits` of the result.
    pub value: AssignedValue<Fr>,
    /// The value is guaranteed to be less than 2^max_bits.
    pub max_bits: usize,
}

impl BabyBearWire {
    pub fn to_baby_bear(&self) -> BabyBear {
        let mut b_int = fe_to_bigint(self.value.value()) % BabyBear::ORDER_U32;
        if b_int < BigInt::from(0) {
            b_int += BabyBear::ORDER_U32;
        }
        BabyBear::from_u32(b_int.try_into().unwrap())
    }

    pub fn as_u64(&self) -> u64 {
        PrimeField64::as_canonical_u64(&self.to_baby_bear())
    }
}

#[derive(Clone, Debug)]
pub struct BabyBearChip {
    pub range: Arc<RangeChip<Fr>>,
    /// Cache for loaded constants, keyed by canonical u64 value.
    const_cache: RefCell<HashMap<u64, BabyBearWire>>,
}

impl BabyBearChip {
    pub fn new(range_chip: Arc<RangeChip<Fr>>) -> Self {
        BabyBearChip {
            range: range_chip,
            const_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn gate(&self) -> &GateChip<Fr> {
        self.range.gate()
    }

    pub fn range(&self) -> &RangeChip<Fr> {
        &self.range
    }

    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBear) -> BabyBearWire {
        let value = ctx.load_witness(Fr::from(PrimeField64::as_canonical_u64(&value)));
        self.range.range_check(ctx, value, BABYBEAR_MAX_BITS);
        BabyBearWire {
            value,
            max_bits: BABYBEAR_MAX_BITS,
        }
    }

    pub fn load_constant(&self, ctx: &mut Context<Fr>, value: BabyBear) -> BabyBearWire {
        let key = value.as_canonical_u64();
        if let Some(&cached) = self.const_cache.borrow().get(&key) {
            return cached;
        }
        let max_bits = bit_length(key);
        let assigned = if value == BabyBear::ZERO {
            ctx.load_zero()
        } else {
            ctx.load_constant(Fr::from(key))
        };
        let wire = BabyBearWire {
            value: assigned,
            max_bits,
        };
        self.const_cache.borrow_mut().insert(key, wire);
        wire
    }

    pub fn reduce(&self, ctx: &mut Context<Fr>, a: BabyBearWire) -> BabyBearWire {
        guarded_debug_assert!(fe_to_bigint(a.value.value()).bits() as usize <= a.max_bits);
        let (_, r) = signed_div_mod(&self.range, ctx, a.value, a.max_bits);
        let r = BabyBearWire {
            value: r,
            max_bits: BABYBEAR_MAX_BITS,
        };
        guarded_debug_assert_eq!(a.to_baby_bear(), r.to_baby_bear());
        r
    }

    /// Reduce max_bits if possible. This function doesn't guarantee that the actual value is within
    /// BabyBear.
    pub fn reduce_max_bits(&self, ctx: &mut Context<Fr>, a: BabyBearWire) -> BabyBearWire {
        if a.max_bits > BABYBEAR_MAX_BITS {
            self.reduce(ctx, a)
        } else {
            a
        }
    }

    pub fn add(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearWire,
        mut b: BabyBearWire,
    ) -> BabyBearWire {
        if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
            if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
                b = self.reduce(ctx, b);
            }
        }
        let value = self.gate().add(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let mut c = BabyBearWire { value, max_bits };
        guarded_debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() + b.to_baby_bear());
        if c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c);
        }
        c
    }

    pub fn neg(&self, ctx: &mut Context<Fr>, a: BabyBearWire) -> BabyBearWire {
        let value = self.gate().neg(ctx, a.value);
        let b = BabyBearWire {
            value,
            max_bits: a.max_bits,
        };
        guarded_debug_assert_eq!(b.to_baby_bear(), -a.to_baby_bear());
        b
    }

    pub fn sub(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearWire,
        mut b: BabyBearWire,
    ) -> BabyBearWire {
        if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
            if a.max_bits.max(b.max_bits) + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
                b = self.reduce(ctx, b);
            }
        }
        let value = self.gate().sub(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let mut c = BabyBearWire { value, max_bits };
        guarded_debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() - b.to_baby_bear());
        if c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c);
        }
        c
    }

    pub fn mul(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearWire,
        mut b: BabyBearWire,
    ) -> BabyBearWire {
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

        let mut c = BabyBearWire { value, max_bits };
        if c.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            c = self.reduce(ctx, c);
        }
        guarded_debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() * b.to_baby_bear());
        c
    }

    pub fn mul_add(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearWire,
        mut b: BabyBearWire,
        mut c: BabyBearWire,
    ) -> BabyBearWire {
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

        let mut d = BabyBearWire { value, max_bits };
        if d.max_bits > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            d = self.reduce(ctx, d);
        }
        guarded_debug_assert_eq!(
            d.to_baby_bear(),
            a.to_baby_bear() * b.to_baby_bear() + c.to_baby_bear()
        );
        d
    }

    pub fn div(
        &self,
        ctx: &mut Context<Fr>,
        mut a: BabyBearWire,
        mut b: BabyBearWire,
    ) -> BabyBearWire {
        let b_val = b.to_baby_bear();
        let b_inv_val = b_val.try_inverse().unwrap();
        // Constrain b is non-zero by checking b * b_inv == 1
        let b_inv = self.load_witness(ctx, b_inv_val);
        let one = self.load_constant(ctx, BabyBear::ONE);
        let inv_prod = self.mul(ctx, b, b_inv);
        self.assert_equal(ctx, inv_prod, one);

        // Constrain a = b * c (mod p)
        let mut c = self.load_witness(ctx, a.to_baby_bear() * b_inv_val);
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
            BabyBearWire {
                value: diff,
                max_bits,
            },
        );
        guarded_debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() / b.to_baby_bear());
        c
    }

    // This inner product function will be used exclusively for optimizing extension element
    // multiplication.
    pub(super) fn special_inner_product(
        &self,
        ctx: &mut Context<Fr>,
        a: &mut [BabyBearWire],
        b: &mut [BabyBearWire],
        s: usize,
    ) -> BabyBearWire {
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
        BabyBearWire {
            value: prod,
            max_bits,
        }
    }

    pub fn select(
        &self,
        ctx: &mut Context<Fr>,
        cond: AssignedValue<Fr>,
        a: BabyBearWire,
        b: BabyBearWire,
    ) -> BabyBearWire {
        let value = self.gate().select(ctx, a.value, b.value, cond);
        let max_bits = a.max_bits.max(b.max_bits);
        BabyBearWire { value, max_bits }
    }

    pub fn assert_zero(&self, ctx: &mut Context<Fr>, a: BabyBearWire) {
        // The proof of correctness of this function is listed in `signed_div_mod`.
        guarded_debug_assert_eq!(a.to_baby_bear(), BabyBear::ZERO);
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
        guarded_debug_assert!(*shifted_div.value() < biguint_to_fe(&(&bound * 2u32 + 1u32)));
        self.range
            .range_check(ctx, shifted_div, (bound * 2u32 + 1u32).bits() as usize);
    }

    pub fn assert_equal(&self, ctx: &mut Context<Fr>, a: BabyBearWire, b: BabyBearWire) {
        guarded_debug_assert_eq!(a.to_baby_bear(), b.to_baby_bear());
        let diff = self.sub(ctx, a, b);
        self.assert_zero(ctx, diff);
    }

    pub fn zero(&self, ctx: &mut Context<Fr>) -> BabyBearWire {
        self.load_constant(ctx, BabyBear::ZERO)
    }

    pub fn one(&self, ctx: &mut Context<Fr>) -> BabyBearWire {
        self.load_constant(ctx, BabyBear::ONE)
    }

    pub fn mul_const(&self, ctx: &mut Context<Fr>, a: BabyBearWire, c: BabyBear) -> BabyBearWire {
        let c_wire = self.load_constant(ctx, c);
        self.mul(ctx, a, c_wire)
    }

    pub fn square(&self, ctx: &mut Context<Fr>, a: BabyBearWire) -> BabyBearWire {
        self.mul(ctx, a, a)
    }

    pub fn pow_power_of_two(
        &self,
        ctx: &mut Context<Fr>,
        a: BabyBearWire,
        n: usize,
    ) -> BabyBearWire {
        let mut result = a;
        for _ in 0..n {
            result = self.square(ctx, result);
        }
        result
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
    guarded_debug_assert!(*shifted_div.value() < biguint_to_fe(&(&bound * 2u32 + 1u32)));
    range.range_check(ctx, shifted_div, (bound * 2u32 + 1u32).bits() as usize);
    guarded_debug_assert!(*rem.value() < biguint_to_fe(&b));
    range.check_big_less_than_safe(ctx, rem, b);
    (div, rem)
}
