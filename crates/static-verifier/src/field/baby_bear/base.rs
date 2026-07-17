use std::{cell::RefCell, collections::HashMap, sync::Arc};

use halo2_base::{
    gates::{GateChip, GateInstructions, RangeChip, RangeInstructions},
    halo2_proofs::{
        arithmetic::Field as _,
        halo2curves::{bn256::Fr, ff::PrimeField as _},
    },
    safe_types::SafeBool,
    utils::{bigint_to_fe, biguint_to_fe, bit_length, fe_to_bigint, modulus, BigPrimeField},
    AssignedValue, Context, QuantumCell,
};
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{Field, PrimeCharacteristicRing, PrimeField32, PrimeField64},
    p3_baby_bear::BabyBear,
};

use super::BABY_BEAR_MODULUS_U64;
use crate::utils::{guarded_debug_assert, guarded_debug_assert_eq};

pub(crate) const BABYBEAR_MAX_BITS: usize = 31;
// bits reserved so that if we do lazy range checking, we still have a valid result
// the first reserved bit is so that we can represent negative numbers
// the second is to accommodate lazy range checking
pub(crate) const RESERVED_HIGH_BITS: usize = 2;

/// Generic over the cell representation `F`, which is `AssignedValue<Fr>` for the halo2 backend
/// and a node id for the graph IR backend.
#[derive(Copy, Clone, Debug)]
pub struct BabyBearWire<F = AssignedValue<Fr>> {
    /// Logically `value` is a signed integer represented as `Bn254`.
    /// Invariants:
    /// - `|value|` never overflows `Bn254`
    /// - `|value| < 2^max_bits` and `max_bits <= Fr::CAPACITY - RESERVED_HIGH_BITS`
    ///
    /// Basically `value` could do arithmetic operations without extra constraints as long as the
    /// result doesn't overflow `Bn254`. And it's easy to track `max_bits` of the result.
    pub value: F,
    /// The value is guaranteed to be less than 2^max_bits.
    pub max_bits: usize,
}

/// A BabyBear wire constrained to its canonical representative in `[0, p)`.
///
/// This type marks values that are safe to absorb into BabyBear-domain transcript
/// and hash inputs. Arithmetic may still use the underlying `BabyBearWire`; converting
/// via `BabyBearWire::from` only drops this type-level evidence and does not add or
/// remove constraints.
#[derive(Copy, Clone, Debug)]
pub struct ReducedBabyBearWire<F = AssignedValue<Fr>>(BabyBearWire<F>);

impl<F: Copy> ReducedBabyBearWire<F> {
    pub fn value(&self) -> F {
        self.0.value
    }
    
    /// Wraps a wire in canonicality evidence. Callers must guarantee the wire is
    /// constrained to `[0, p)`; this adds no constraints.
    #[cfg(feature = "halo2-gpu")]
    pub(crate) fn assume_reduced(wire: BabyBearWire<F>) -> Self {
        ReducedBabyBearWire(wire)
    }
}

impl<F> From<ReducedBabyBearWire<F>> for BabyBearWire<F> {
    /// Drops the canonicality evidence and returns the underlying arithmetic wire.
    fn from(wire: ReducedBabyBearWire<F>) -> Self {
        wire.0
    }
}

impl<F: Copy> From<&ReducedBabyBearWire<F>> for BabyBearWire<F> {
    fn from(wire: &ReducedBabyBearWire<F>) -> Self {
        (*wire).into()
    }
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

    /// Loads a BabyBear witness and constrains only that the assigned advice cell
    /// fits in 31 bits.
    ///
    /// The Rust input is canonicalized for the honest witness assignment, but the
    /// circuit does not prove the advice cell is `< p`. Use `load_reduced_witness`
    /// for values that will be absorbed into transcripts or hashes.
    pub fn load_witness(&self, ctx: &mut Context<Fr>, value: BabyBear) -> BabyBearWire {
        let value = ctx.load_witness(Fr::from(PrimeField64::as_canonical_u64(&value)));
        self.range.range_check(ctx, value, BABYBEAR_MAX_BITS);
        BabyBearWire {
            value,
            max_bits: BABYBEAR_MAX_BITS,
        }
    }

    /// Loads a witness and constrains it to the canonical BabyBear range `[0, p)`.
    pub fn load_reduced_witness(
        &self,
        ctx: &mut Context<Fr>,
        value: BabyBear,
    ) -> ReducedBabyBearWire {
        let value = ctx.load_witness(Fr::from(PrimeField64::as_canonical_u64(&value)));
        self.range
            .check_less_than_safe(ctx, value, BABY_BEAR_MODULUS_U64);
        ReducedBabyBearWire(BabyBearWire {
            value,
            max_bits: BABYBEAR_MAX_BITS,
        })
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

    /// Loads a canonical BabyBear constant and returns it with reduced type evidence.
    pub fn load_reduced_constant(
        &self,
        ctx: &mut Context<Fr>,
        value: BabyBear,
    ) -> ReducedBabyBearWire {
        // Constants are canonical by construction.
        ReducedBabyBearWire(self.load_constant(ctx, value))
    }

    pub fn reduce(&self, ctx: &mut Context<Fr>, a: BabyBearWire) -> BabyBearWire {
        assert!(a.max_bits <= Fr::CAPACITY as usize - RESERVED_HIGH_BITS);
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
        if a.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
        }
        if b.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            b = self.reduce(ctx, b);
        }
        let value = self.gate().add(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let c = BabyBearWire { value, max_bits };
        guarded_debug_assert_eq!(c.to_baby_bear(), a.to_baby_bear() + b.to_baby_bear());
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
        #[cfg(debug_assertions)]
        let expected = a.to_baby_bear() - b.to_baby_bear();
        if a.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
        }
        if b.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            b = self.reduce(ctx, b);
        }
        let value = self.gate().sub(ctx, a.value, b.value);
        let max_bits = a.max_bits.max(b.max_bits) + 1;
        let c = BabyBearWire { value, max_bits };
        guarded_debug_assert_eq!(c.to_baby_bear(), expected);
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

        let c = BabyBearWire { value, max_bits };
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

        let d = BabyBearWire { value, max_bits };
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
        if a.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            a = self.reduce(ctx, a);
        }
        if b.max_bits + c.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
            b = self.reduce(ctx, b);
        }
        if b.max_bits + c.max_bits + 1 > Fr::CAPACITY as usize - RESERVED_HIGH_BITS {
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
        cond: SafeBool<Fr>,
        a: BabyBearWire,
        b: BabyBearWire,
    ) -> BabyBearWire {
        let value = self.gate().select(ctx, a.value, b.value, *cond.as_ref());
        let max_bits = a.max_bits.max(b.max_bits);
        BabyBearWire { value, max_bits }
    }

    pub fn assert_zero(&self, ctx: &mut Context<Fr>, a: BabyBearWire) {
        guarded_debug_assert_eq!(a.to_baby_bear(), BabyBear::ZERO);
        assert!(a.max_bits <= Fr::CAPACITY as usize - RESERVED_HIGH_BITS);
        let a_num_bits = a.max_bits;
        let b: BigUint = BabyBear::ORDER_U32.into();
        let a_val = fe_to_bigint(a.value.value());
        assert!(a_val.bits() <= a_num_bits as u64);
        // The honest input is congruent to zero modulo the BabyBear prime, so
        // Euclidean division by `b` has exact remainder zero.
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
        // Constrain the exact quotient to the range implied by `|a| < 2^a_num_bits`.
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

/// Constrains and returns `(div, rem)` encoding integers `D` and `R` such that
/// `A = BabyBear::ORDER_U32 * D + R` with `0 <= R < BabyBear::ORDER_U32`, where
/// `A = fe_to_bigint(a)` is the canonical signed representative of `a` in
/// `(-p/2, p/2]` and `p = F::MODULUS`.
///
/// The returned `div` cell encodes the possibly negative integer `D` modulo
/// `p`; the returned `rem` cell encodes the canonical nonnegative integer `R`.
///
/// # Arguments
///
/// * `a` - the [`QuantumCell`] value to divide.
/// * `a_num_bits` - a bound such that `|A| < 2^a_num_bits`.
///
/// # Preconditions
///
/// This function does not itself range-check `a`. The caller must ensure,
/// either by prior constraints or by construction, that the canonical signed
/// representative `A = fe_to_bigint(a)` satisfies `|A| < 2^a_num_bits`.
fn signed_div_mod<F>(
    range: &RangeChip<F>,
    ctx: &mut Context<F>,
    a: impl Into<QuantumCell<F>>,
    a_num_bits: usize,
) -> (AssignedValue<F>, AssignedValue<F>)
where
    F: BigPrimeField,
{
    assert!(a_num_bits <= F::CAPACITY as usize - RESERVED_HIGH_BITS);
    // Proof sketch:
    //
    // Let `b = BabyBear::ORDER_U32`, `p = F::MODULUS`, and let
    // `A = fe_to_bigint(a)` be the canonical signed representative of `a`.
    // Assume `|A| < 2^a_num_bits`.
    //
    // The intended witnesses are the Euclidean quotient and remainder:
    //
    //   div = floor(A / b),   rem = A mod b,   0 <= rem < b.
    //
    // We enforce:
    //
    //   (1) rem + b * div = a      over F
    //   (2) 0 <= rem < b
    //   (3) 0 <= div + bound < 2^k
    //
    // where
    //
    //   bound = ceil((2^a_num_bits - 1) / b)
    //   k     = bits(2 * bound + 1).
    //
    // Completeness is immediate: since `|A| <= 2^a_num_bits - 1`, the honest
    // quotient satisfies `|div| <= bound`, so `div + bound` lies in
    // `[0, 2 * bound]` and passes the `k`-bit range check.
    //
    // For soundness, note that the `k`-bit check is slightly looser than
    // `div + bound <= 2 * bound`. For any satisfying assignment, let `S` be the
    // canonical integer value of `div + bound`, so `0 <= S < 2^k`, and set
    // `D = S - bound`. Since `2^k <= 4 * bound + 2`, we have
    //
    //   -bound <= D <= 3 * bound + 1.
    //
    // Thus any two satisfying assignments `(D, R)` and `(D', R')` obey
    //
    //   |D - D'| <= 4 * bound + 1,
    //   |R - R'| < b.
    //
    // Both assignments satisfy:
    //
    //   R  + b * D  = a mod p
    //   R' + b * D' = a mod p
    //
    // Subtracting the second congruence from the first gives
    //
    //   (D - D') * b - (R' - R) = 0 mod p.
    //
    // Equivalently, this integer is some multiple of `p`:
    //
    //   (D - D') * b - (R' - R) = m * p.
    //
    // Its magnitude is bounded by
    //
    //   |(D - D') * b - (R' - R)| < (4 * bound + 2) * b.
    //
    // The runtime assertion
    //
    //   assert!((4 * bound + 2) * b <= p)
    //
    // ensures this magnitude is strictly less than `p`. The only multiple of `p`
    // with magnitude less than `p` is zero, so `m = 0`. Therefore
    //
    //   (D - D') * b = R' - R.
    //
    // The left side is divisible by `b`, while the right side has magnitude `< b`.
    // Hence both sides are zero, so `D = D'` and `R = R'`.
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
    // `bound = ceil((2^a_num_bits - 1) / b)`; the bit-length range check below admits
    // `div in [-bound, 3*bound+1]`, and the assertion enforces the no-wrap headroom
    // `(4 * bound + 2) * b <= p`. See the proof above for both.
    let bound = ((BigUint::from(1u32) << a_num_bits) - 1u32).div_ceil(&b);
    assert!((&bound * 4u32 + 2u32) * &b <= modulus::<F>());
    let shifted_div = range
        .gate()
        .add(ctx, div, QuantumCell::Constant(biguint_to_fe(&bound)));
    guarded_debug_assert!(*shifted_div.value() < biguint_to_fe(&(&bound * 2u32 + 1u32)));
    range.range_check(ctx, shifted_div, (bound * 2u32 + 1u32).bits() as usize);
    guarded_debug_assert!(*rem.value() < biguint_to_fe(&b));
    range.check_big_less_than_safe(ctx, rem, b);
    (div, rem)
}
