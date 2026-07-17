//! Standalone tape-replay implementations of [`Halo2Opcode`]s.
//!
//! Every op is executed by [`run_op`] against a [`ReplayTape`]:
//! [`CalculateOffsetsTape`] records values plus output offsets and yields an
//! [`OpcodeMeta`] (tape lengths + output offset table + constant-skip indices);
//! it carries a constant cache to simulate the chips' caching behavior, so it is
//! only used at IR-build time and in tests. [`WitnessTape`] streams the witness
//! values into caller-provided buffers at runtime; it holds **no** cache — the
//! build-time [`OpcodeMeta::constant_skip_inds`] tells it exactly which
//! `load_constant` calls materialize a cell, so replay is stateless and
//! parallelizable across nodes.
//! Gate/range/BabyBear primitives are provided methods on [`ReplayTape`], generic
//! over the tape's cell type; ops that may reduce take [`BbWire`]s (value + bit
//! bound) and pure gate ops take plain `Fr` values.

use std::cmp::Ordering;

use halo2_base::{
    halo2_proofs::{
        arithmetic::Field as _,
        halo2curves::{bn256::Fr, ff::PrimeField as _},
    },
    utils::{
        bigint_to_fe, biguint_to_fe, bit_length, decompose_fe_to_u64_limbs, fe_to_bigint,
        fe_to_biguint, modulus,
    },
};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use openvm_stark_sdk::{
    openvm_stark_backend::p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
        PrimeField32, PrimeField64,
    },
    p3_baby_bear::BabyBear,
};

use crate::{
    field::baby_bear::{
        BabyBearExt4, BABYBEAR_MAX_BITS, BABY_BEAR_MODULUS_U64, RESERVED_HIGH_BITS,
    },
    halo2_ir_builder::Halo2Opcode,
    hash::{poseidon2::Poseidon2Params, POSEIDON2_COMPRESS_PARAMS, POSEIDON2_PARAMS},
    transcript::NUM_SAMPLES_PER_WORD,
};

/// Reduce when a bound would exceed this, mirroring `BabyBearChip`.
const REDUCE_THRESHOLD: u16 = (Fr::CAPACITY as usize - RESERVED_HIGH_BITS) as u16;

/// Sentinel offset for cells that exist outside the op's tape window (external
/// operands and constants that hit the constant cache and assigned no advice cell).
pub(crate) const UNMATERIALIZED: usize = usize::MAX;

fn pow_of_two_fr(n: usize) -> Fr {
    biguint_to_fe(&(BigUint::from(1u32) << n))
}

fn to_baby_bear(v: &Fr) -> BabyBear {
    let mut b_int = fe_to_bigint(v) % BabyBear::ORDER_U32;
    if b_int < BigInt::from(0) {
        b_int += BabyBear::ORDER_U32;
    }
    BabyBear::from_u32(b_int.try_into().unwrap())
}

fn fr_from_bb(v: BabyBear) -> Fr {
    Fr::from(v.as_canonical_u64())
}

/// A tape cell: at minimum it knows the advice value it carries.
pub(crate) trait TapeCell: Copy {
    fn value(&self) -> Fr;
}

impl TapeCell for Fr {
    #[inline]
    fn value(&self) -> Fr {
        *self
    }
}

/// A value together with the relative context-tape offset it was written at
/// ([`UNMATERIALIZED`] for cells outside the op's window).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct OffsetCell {
    pub value: Fr,
    pub offset: usize,
}

impl TapeCell for OffsetCell {
    #[inline]
    fn value(&self) -> Fr {
        self.value
    }
}

/// Constant cache mirroring `BabyBearChip`'s dedup: the first load of a
/// constant pushes a cell, repeats reuse it. Only ZERO/ONE/W occur, so a fixed
/// array with linear scan suffices.
struct ConstCache<C> {
    entries: [Option<(Fr, C)>; 4],
}

impl<C> Default for ConstCache<C> {
    fn default() -> Self {
        ConstCache {
            entries: core::array::from_fn(|_| None),
        }
    }
}

impl<C: Copy> ConstCache<C> {
    fn get(&self, value: Fr) -> Option<C> {
        self.entries
            .iter()
            .flatten()
            .find(|(v, _)| *v == value)
            .map(|(_, c)| *c)
    }

    fn insert(&mut self, value: Fr, cell: C) {
        let slot = self
            .entries
            .iter_mut()
            .find(|e| e.is_none())
            .expect("constant cache full");
        *slot = Some((value, cell));
    }
}

/// Replay target for opcode execution. Required methods define how cells land on
/// the tapes; the provided methods implement the exact halo2 cell layouts.
pub(crate) trait ReplayTape {
    type TapeCell: TapeCell;

    fn push(&mut self, value: Fr) -> Self::TapeCell;
    fn lookup(&mut self, value: Fr);
    fn lookup_bits(&self) -> usize;
    /// A cell whose value flows in from outside the op's tape window.
    fn external(&self, value: Fr) -> Self::TapeCell;
    /// A constant loaded through the chips' constant caches: pushes a cell only
    /// when the constant is not already materialized. [`CalculateOffsetsTape`]
    /// decides via its cache; [`WitnessTape`] via precomputed skip indices.
    fn load_constant(&mut self, value: Fr) -> Self::TapeCell;
    /// Records a logical output of the op, in output-index order.
    fn output(&mut self, _cell: Self::TapeCell) {}

    // --- gate ops (exact `GateInstructions` layouts); pure Fr in, cell out ---

    /// `| a | b | 1 | a + b |`, out at +3.
    fn gate_add(&mut self, a: Fr, b: Fr) -> Self::TapeCell {
        self.push(a);
        self.push(b);
        self.push(Fr::ONE);
        self.push(a + b)
    }

    /// `| a - b | b | 1 | a |`, out at +0.
    fn gate_sub(&mut self, a: Fr, b: Fr) -> Self::TapeCell {
        let out = self.push(a - b);
        self.push(b);
        self.push(Fr::ONE);
        self.push(a);
        out
    }

    /// `| a - b * c | b | c | a |`, out at +0.
    fn gate_sub_mul(&mut self, a: Fr, b: Fr, c: Fr) -> Self::TapeCell {
        let out = self.push(a - b * c);
        self.push(b);
        self.push(c);
        self.push(a);
        out
    }

    /// `| a | -a | 1 | 0 |`, out at +1.
    fn gate_neg(&mut self, a: Fr) -> Self::TapeCell {
        self.push(a);
        let out = self.push(-a);
        self.push(Fr::ONE);
        self.push(Fr::ZERO);
        out
    }

    /// `| 0 | a | b | a * b |`, out at +3.
    fn gate_mul(&mut self, a: Fr, b: Fr) -> Self::TapeCell {
        self.push(Fr::ZERO);
        self.push(a);
        self.push(b);
        self.push(a * b)
    }

    /// `| c | a | b | a * b + c |`, out at +3.
    fn gate_mul_add(&mut self, a: Fr, b: Fr, c: Fr) -> Self::TapeCell {
        self.push(c);
        self.push(a);
        self.push(b);
        self.push(a * b + c)
    }

    /// `| 0 | x | x | x |`.
    fn gate_assert_bit(&mut self, x: Fr) {
        self.push(Fr::ZERO);
        self.push(x);
        self.push(x);
        self.push(x);
    }

    fn gate_not(&mut self, a: Fr) -> Self::TapeCell {
        self.gate_sub(Fr::ONE, a)
    }

    /// `| a - b | 1 | b | a | b | sel | a - b | out |`, out at +7.
    fn gate_select(&mut self, a: Fr, b: Fr, sel: Fr) -> Self::TapeCell {
        let diff = a - b;
        self.push(diff);
        self.push(Fr::ONE);
        self.push(b);
        self.push(a);
        self.push(b);
        self.push(sel);
        self.push(diff);
        self.push(sel * diff + b)
    }

    /// `| is_zero | a | inv | 1 | 0 | a | is_zero | 0 |`, out at +6.
    fn gate_is_zero(&mut self, a: Fr) -> Self::TapeCell {
        let is_zero = if a == Fr::ZERO { Fr::ONE } else { Fr::ZERO };
        let inv = a.invert().unwrap_or(Fr::ONE);
        self.push(is_zero);
        self.push(a);
        self.push(inv);
        self.push(Fr::ONE);
        self.push(Fr::ZERO);
        self.push(a);
        let out = self.push(is_zero);
        self.push(Fr::ZERO);
        out
    }

    fn gate_is_equal(&mut self, a: Fr, b: Fr) -> Self::TapeCell {
        self.gate_sub(a, b);
        self.gate_is_zero(a - b)
    }

    /// `| v_0 |` then `| v_i | 1 | run |` per further element; out is the last cell.
    fn gate_sum(&mut self, values: impl IntoIterator<Item = Fr>) -> Self::TapeCell {
        let mut iter = values.into_iter();
        let mut sum = iter.next().unwrap();
        let mut last = self.push(sum);
        for v in iter {
            self.push(v);
            self.push(Fr::ONE);
            sum += v;
            last = self.push(sum);
        }
        last
    }

    /// `inner_product_simple`: with `starts_with_one` the cells are `| a_0 |` then
    /// `| a_i | b_i | run |` per further pair; otherwise `| 0 |` then
    /// `| a_i | b_i | run |` per pair. Out is the last cell assigned.
    fn gate_inner_product(
        &mut self,
        pairs: impl IntoIterator<Item = (Fr, Fr)>,
        starts_with_one: bool,
    ) -> Self::TapeCell {
        let mut iter = pairs.into_iter();
        let mut sum;
        let mut last;
        if starts_with_one {
            let (a0, _) = iter.next().unwrap();
            sum = a0;
            last = self.push(a0);
        } else {
            sum = Fr::ZERO;
            last = self.push(Fr::ZERO);
        }
        for (av, bv) in iter {
            self.push(av);
            self.push(bv);
            sum += av * bv;
            last = self.push(sum);
        }
        last
    }

    /// `num_to_bits`: little-endian bit decomposition + per-bit `assert_bit`. Each
    /// bit cell is emitted as an op output, in bit order.
    fn gate_num_to_bits(&mut self, a: Fr, range_bits: usize) {
        let bits = decompose_fe_to_u64_limbs(&a, range_bits, 1);
        // Inner product against powers of two; `pow_of_two[0] == 1`, so it starts
        // with one and bit 0 is the first cell.
        let mut sum = Fr::from(bits[0]);
        let first = self.push(sum);
        self.output(first);
        for (i, &b) in bits.iter().enumerate().skip(1) {
            let bv = Fr::from(b);
            let cell = self.push(bv);
            self.output(cell);
            self.push(pow_of_two_fr(i));
            sum += bv * pow_of_two_fr(i);
            self.push(sum);
        }
        // constrain_equal: no advice cells
        for &b in &bits {
            self.gate_assert_bit(Fr::from(b));
        }
    }

    // --- range ops (exact `RangeInstructions` layouts) ---

    /// `_range_check`; returns the last (highest) limb value.
    fn range_check(&mut self, a: Fr, range_bits: usize) -> Fr {
        if range_bits == 0 {
            // assert_is_const assigns no advice cells
            return a;
        }
        let lookup_bits = self.lookup_bits();
        let num_limbs = range_bits.div_ceil(lookup_bits);
        let rem_bits = range_bits % lookup_bits;
        let last_limb = if num_limbs == 1 {
            self.lookup(a);
            a
        } else {
            let limbs = decompose_fe_to_u64_limbs(&a, num_limbs, lookup_bits);
            // `limb_bases[0] == 1`, so the inner product starts with one.
            self.gate_inner_product(
                limbs
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (Fr::from(l), pow_of_two_fr(i * lookup_bits))),
                true,
            );
            // constrain_equal: no advice cells; then all limbs are sent to lookup in
            // natural order.
            for &l in &limbs {
                self.lookup(Fr::from(l));
            }
            Fr::from(limbs[num_limbs - 1])
        };
        match rem_bits.cmp(&1) {
            Ordering::Equal => {
                self.gate_assert_bit(last_limb);
            }
            Ordering::Greater => {
                let shift = pow_of_two_fr(lookup_bits - rem_bits);
                self.gate_mul(last_limb, shift);
                self.lookup(last_limb * shift);
            }
            Ordering::Less => {}
        }
        last_limb
    }

    /// `| a + 2^n - b | b | 1 | a + 2^n | -2^n | 1 | a |` + range check of the diff.
    fn check_less_than(&mut self, a: Fr, b: Fr, num_bits: usize) {
        let pow = pow_of_two_fr(num_bits);
        let shift_a = pow + a;
        let diff = shift_a - b;
        self.push(diff);
        self.push(b);
        self.push(Fr::ONE);
        self.push(shift_a);
        self.push(-pow);
        self.push(Fr::ONE);
        self.push(a);
        self.range_check(diff, num_bits);
    }

    fn check_less_than_safe(&mut self, a: Fr, b: u64) {
        let range_bits = bit_length(b).div_ceil(self.lookup_bits()) * self.lookup_bits();
        self.range_check(a, range_bits);
        self.check_less_than(a, Fr::from(b), range_bits);
    }

    fn check_big_less_than_safe(&mut self, a: Fr, b: &BigUint) {
        let range_bits = (b.bits() as usize).div_ceil(self.lookup_bits()) * self.lookup_bits();
        self.range_check(a, range_bits);
        self.check_less_than(a, biguint_to_fe(b), range_bits);
    }

    /// Same 7-cell shape with `2^padded`, then `is_zero` of the top limb; returns
    /// the comparison bit cell.
    fn is_less_than(&mut self, a: Fr, b: Fr, num_bits: usize) -> Self::TapeCell {
        let lookup_bits = self.lookup_bits();
        let padded_bits = num_bits.div_ceil(lookup_bits) * lookup_bits;
        let pow = pow_of_two_fr(padded_bits);
        let shift_a = pow + a;
        let shifted = shift_a - b;
        self.push(shifted);
        self.push(b);
        self.push(Fr::ONE);
        self.push(shift_a);
        self.push(-pow);
        self.push(Fr::ONE);
        self.push(a);
        let last_limb = self.range_check(shifted, padded_bits + lookup_bits);
        self.gate_is_zero(last_limb)
    }

    fn is_big_less_than_safe(&mut self, a: Fr, b: &BigUint) -> Self::TapeCell {
        let range_bits = (b.bits() as usize).div_ceil(self.lookup_bits()) * self.lookup_bits();
        self.range_check(a, range_bits);
        self.is_less_than(a, biguint_to_fe(b), range_bits)
    }

    /// `| rem | b | div | a |` + quotient and remainder bound checks; returns the
    /// remainder cell.
    fn div_mod(&mut self, a: Fr, b: &BigUint, a_num_bits: usize) -> Self::TapeCell {
        let a_val = fe_to_biguint(&a);
        let (div_val, rem_val) = a_val.div_mod_floor(b);
        let rem = self.push(biguint_to_fe(&rem_val));
        self.push(biguint_to_fe(b));
        let div = self.push(biguint_to_fe(&div_val));
        self.push(a);
        self.check_big_less_than_safe(
            div.value(),
            &((BigUint::from(1u32) << a_num_bits) / b + 1u32),
        );
        self.check_big_less_than_safe(rem.value(), b);
        rem
    }

    // --- BabyBear engine (exact `BabyBearChip` behavior) ---

    fn bb_external(&self, value: Fr, max_bits: u16) -> BbWire<Self::TapeCell> {
        BbWire {
            cell: self.external(value),
            max_bits,
        }
    }

    /// `signed_div_mod`: `| rem | p | div | a |` + shifted-quotient range check +
    /// `check_big_less_than_safe(rem, p)`; returns the remainder cell.
    fn bb_signed_div_mod(&mut self, a: Fr, a_num_bits: u16) -> Self::TapeCell {
        let b = BigUint::from(BABY_BEAR_MODULUS_U64);
        let a_val = fe_to_bigint(&a);
        let (div_val, rem_val) = a_val.div_mod_floor(&b.clone().into());
        let rem = self.push(bigint_to_fe(&rem_val));
        self.push(biguint_to_fe(&b));
        let div = self.push(bigint_to_fe(&div_val));
        self.push(a);
        let bound = ((BigUint::from(1u32) << a_num_bits) - 1u32).div_ceil(&b);
        let shifted = self.gate_add(div.value(), biguint_to_fe(&bound));
        self.range_check(shifted.value(), (bound * 2u32 + 1u32).bits() as usize);
        self.check_big_less_than_safe(rem.value(), &b);
        rem
    }

    fn bb_reduce(&mut self, a: BbWire<Self::TapeCell>) -> BbWire<Self::TapeCell> {
        let rem = self.bb_signed_div_mod(a.cell.value(), a.max_bits);
        BbWire {
            cell: rem,
            max_bits: BABYBEAR_MAX_BITS as u16,
        }
    }

    /// `assert_zero`: `| 0 | p | div | a |` + shifted-quotient range check (plain
    /// division bound, no remainder check).
    fn bb_assert_zero(&mut self, a: BbWire<Self::TapeCell>) {
        let b = BigUint::from(BABY_BEAR_MODULUS_U64);
        let a_val = fe_to_bigint(&a.cell.value());
        let (div_val, _) = a_val.div_mod_floor(&b.clone().into());
        self.push(Fr::ZERO);
        self.push(biguint_to_fe(&b));
        let div = self.push(bigint_to_fe(&div_val));
        self.push(a.cell.value());
        let bound = (BigUint::from(1u32) << a.max_bits) / &b;
        let shifted = self.gate_add(div.value(), biguint_to_fe(&bound));
        self.range_check(shifted.value(), (bound * 2u32 + 1u32).bits() as usize);
    }

    fn bb_load_witness(&mut self, value: BabyBear) -> BbWire<Self::TapeCell> {
        let cell = self.push(fr_from_bb(value));
        self.range_check(cell.value(), BABYBEAR_MAX_BITS);
        BbWire {
            cell,
            max_bits: BABYBEAR_MAX_BITS as u16,
        }
    }

    /// `load_constant`: assigns one cell on the first load of a constant; repeats
    /// hit the [`ConstCache`] and contribute no cells.
    fn bb_load_constant(&mut self, value: BabyBear) -> BbWire<Self::TapeCell> {
        let key = value.as_canonical_u64();
        let cell = self.load_constant(Fr::from(key));
        BbWire {
            cell,
            max_bits: bit_length(key) as u16,
        }
    }

    /// Full `BabyBearChip::mul` with swap + interleaved reduces.
    fn bb_mul(
        &mut self,
        mut a: BbWire<Self::TapeCell>,
        mut b: BbWire<Self::TapeCell>,
    ) -> BbWire<Self::TapeCell> {
        if a.max_bits < b.max_bits {
            std::mem::swap(&mut a, &mut b);
        }
        if a.max_bits + b.max_bits > REDUCE_THRESHOLD {
            a = self.bb_reduce(a);
            if a.max_bits + b.max_bits > REDUCE_THRESHOLD {
                b = self.bb_reduce(b);
            }
        }
        let cell = self.gate_mul(a.cell.value(), b.cell.value());
        BbWire {
            cell,
            max_bits: a.max_bits + b.max_bits,
        }
    }

    /// Full `BabyBearChip::sub` with interleaved reduces.
    fn bb_sub(
        &mut self,
        mut a: BbWire<Self::TapeCell>,
        mut b: BbWire<Self::TapeCell>,
    ) -> BbWire<Self::TapeCell> {
        if a.max_bits + 1 > REDUCE_THRESHOLD {
            a = self.bb_reduce(a);
        }
        if b.max_bits + 1 > REDUCE_THRESHOLD {
            b = self.bb_reduce(b);
        }
        let cell = self.gate_sub(a.cell.value(), b.cell.value());
        BbWire {
            cell,
            max_bits: a.max_bits.max(b.max_bits) + 1,
        }
    }

    /// Full `BabyBearChip::mul_add` with swap + interleaved reduces.
    fn bb_mul_add(
        &mut self,
        mut a: BbWire<Self::TapeCell>,
        mut b: BbWire<Self::TapeCell>,
        mut c: BbWire<Self::TapeCell>,
    ) -> BbWire<Self::TapeCell> {
        if a.max_bits < b.max_bits {
            std::mem::swap(&mut a, &mut b);
        }
        if a.max_bits + b.max_bits + 1 > REDUCE_THRESHOLD {
            a = self.bb_reduce(a);
            if a.max_bits + b.max_bits + 1 > REDUCE_THRESHOLD {
                b = self.bb_reduce(b);
            }
        }
        if c.max_bits + 1 > REDUCE_THRESHOLD {
            c = self.bb_reduce(c);
        }
        let cell = self.gate_mul_add(a.cell.value(), b.cell.value(), c.cell.value());
        BbWire {
            cell,
            max_bits: c.max_bits.max(a.max_bits + b.max_bits) + 1,
        }
    }

    fn bb_assert_equal(&mut self, a: BbWire<Self::TapeCell>, b: BbWire<Self::TapeCell>) {
        let diff = self.bb_sub(a, b);
        self.bb_assert_zero(diff);
    }

    /// `BabyBearChip::special_inner_product`: reduce decisions mutate the operand
    /// arrays persistently, mirroring the chip.
    fn bb_special_inner_product(
        &mut self,
        a: &mut [BbWire<Self::TapeCell>; 4],
        b: &mut [BbWire<Self::TapeCell>; 4],
        s: usize,
    ) -> BbWire<Self::TapeCell> {
        let lb = s.saturating_sub(3);
        let ub = 4.min(s + 1);
        let len = if s < 3 { s + 1 } else { 7 - s };
        let mut max_bits = 0;
        for i in 0..(ub - lb) {
            let ai = lb + i;
            let bi = s - lb - i;
            let limit = REDUCE_THRESHOLD - len as u16 + i as u16;
            if a[ai].max_bits + b[bi].max_bits > limit {
                if a[ai].max_bits >= b[bi].max_bits {
                    a[ai] = self.bb_reduce(a[ai]);
                    if a[ai].max_bits + b[bi].max_bits > limit {
                        b[bi] = self.bb_reduce(b[bi]);
                    }
                } else {
                    b[bi] = self.bb_reduce(b[bi]);
                    if a[ai].max_bits + b[bi].max_bits > limit {
                        a[ai] = self.bb_reduce(a[ai]);
                    }
                }
            }
            max_bits = if i == 0 {
                a[ai].max_bits + b[bi].max_bits
            } else {
                max_bits.max(a[ai].max_bits + b[bi].max_bits) + 1
            };
        }
        // All operands are Existing cells, so the inner product does NOT start with
        // one: `| 0 |` then `| a_i | b_i | run |` per pair.
        let out = self.gate_inner_product(
            (0..(ub - lb)).map(|i| (a[lb + i].cell.value(), b[s - lb - i].cell.value())),
            false,
        );
        BbWire {
            cell: out,
            max_bits,
        }
    }

    /// `BabyBearExt4Chip::mul`. Mutates the operand arrays (reduce decisions persist
    /// across the seven `special_inner_product` calls). Returns the four product
    /// coefficients and the W constant cell (unmaterialized when it hit the cache).
    fn ext_mul(
        &mut self,
        a: &mut [BbWire<Self::TapeCell>; 4],
        b: &mut [BbWire<Self::TapeCell>; 4],
    ) -> ([BbWire<Self::TapeCell>; 4], Self::TapeCell) {
        let mut coeffs: [BbWire<Self::TapeCell>; 7] =
            core::array::from_fn(|_| self.bb_external(Fr::ZERO, 0));
        for (s, coeff) in coeffs.iter_mut().enumerate() {
            *coeff = self.bb_special_inner_product(a, b, s);
        }
        let w = self.bb_load_constant(<BabyBear as BinomiallyExtendable<4>>::W);
        for i in 4..7 {
            coeffs[i - 4] = self.bb_mul_add(coeffs[i], w, coeffs[i - 4]);
        }
        ([coeffs[0], coeffs[1], coeffs[2], coeffs[3]], w.cell)
    }
}

/// Mirror of `BabyBearWire` over tape cells.
#[derive(Copy, Clone, Debug)]
pub(crate) struct BbWire<C> {
    cell: C,
    max_bits: u16,
}

fn ext_value<C: TapeCell>(wires: &[BbWire<C>; 4]) -> BabyBearExt4 {
    BabyBearExt4::from_basis_coefficients_fn(|i| to_baby_bear(&wires[i].cell.value()))
}

/// Records the full tapes plus per-output offsets; use this tape to derive
/// [`OpcodeMeta`] (offsets, ctx len, lookups len, constant-skip indices).
///
/// The constant cache simulates the chips' caching behavior: `warm` constants
/// (already materialized by an earlier node) are pre-seeded as
/// [`UNMATERIALIZED`] cells, so loading them assigns no advice cell. Build-time
/// metadata derivation and tests only; the runtime path is [`WitnessTape`].
pub(crate) struct CalculateOffsetsTape {
    pub advice: Vec<Fr>,
    pub lookups: Vec<Fr>,
    pub outputs: Vec<OffsetCell>,
    /// Indices of the `load_constant` calls that missed the cache (i.e. wrote a
    /// cell), in call order.
    pub skip_inds: Vec<u32>,
    lookup_bits: usize,
    const_cache: ConstCache<OffsetCell>,
    const_calls: u32,
}

impl CalculateOffsetsTape {
    pub(crate) fn new(lookup_bits: usize, warm: &[Fr]) -> Self {
        let mut const_cache = ConstCache::default();
        for &value in warm {
            const_cache.insert(
                value,
                OffsetCell {
                    value,
                    offset: UNMATERIALIZED,
                },
            );
        }
        CalculateOffsetsTape {
            advice: Vec::new(),
            lookups: Vec::new(),
            outputs: Vec::new(),
            skip_inds: Vec::new(),
            lookup_bits,
            const_cache,
            const_calls: 0,
        }
    }
}

impl ReplayTape for CalculateOffsetsTape {
    type TapeCell = OffsetCell;

    fn push(&mut self, value: Fr) -> OffsetCell {
        let offset = self.advice.len();
        self.advice.push(value);
        OffsetCell { value, offset }
    }

    fn lookup(&mut self, value: Fr) {
        self.lookups.push(value);
    }

    fn lookup_bits(&self) -> usize {
        self.lookup_bits
    }

    fn external(&self, value: Fr) -> OffsetCell {
        OffsetCell {
            value,
            offset: UNMATERIALIZED,
        }
    }

    fn load_constant(&mut self, value: Fr) -> OffsetCell {
        let idx = self.const_calls;
        self.const_calls += 1;
        if let Some(cell) = self.const_cache.get(value) {
            return cell;
        }
        self.skip_inds.push(idx);
        let cell = self.push(value);
        self.const_cache.insert(value, cell);
        cell
    }

    fn output(&mut self, cell: OffsetCell) {
        self.outputs.push(cell);
    }
}

/// Streams witness values into caller-provided buffers via raw pointer bumps.
///
/// Holds no constant cache: `write_const_inds` (the node's
/// [`OpcodeMeta::constant_skip_inds`]) lists exactly which `load_constant` calls
/// write a cell, so replay carries no cross-node state and nodes can run in
/// parallel. An empty slice means no `load_constant` call writes.
///
/// Safety: the buffers passed to [`WitnessTape::new`] must be at least
/// [`OpcodeMeta::ctx_len`] / [`OpcodeMeta::lookups_len`] long for the op replayed.
pub(crate) struct WitnessTape {
    advice: *mut Fr,
    lookups: *mut Fr,
    write_const_inds: *const u32,
    write_const_end: *const u32,
    num_const_idx: u32,
    lookup_bits: usize,
}

impl WitnessTape {
    pub(crate) fn new(
        ctx: &mut [Fr],
        lookups: &mut [Fr],
        lookup_bits: usize,
        write_const_inds: &[u32],
    ) -> Self {
        let range = write_const_inds.as_ptr_range();
        WitnessTape {
            advice: ctx.as_mut_ptr(),
            lookups: lookups.as_mut_ptr(),
            write_const_inds: range.start,
            write_const_end: range.end,
            num_const_idx: 0,
            lookup_bits,
        }
    }
}

#[allow(unsafe_code)]
impl ReplayTape for WitnessTape {
    type TapeCell = Fr;

    #[inline]
    fn push(&mut self, value: Fr) -> Fr {
        unsafe {
            *self.advice = value;
            self.advice = self.advice.add(1);
        }
        value
    }

    #[inline]
    fn lookup(&mut self, value: Fr) {
        unsafe {
            *self.lookups = value;
            self.lookups = self.lookups.add(1);
        }
    }

    #[inline]
    fn lookup_bits(&self) -> usize {
        self.lookup_bits
    }

    #[inline]
    fn external(&self, value: Fr) -> Fr {
        value
    }

    #[inline]
    fn load_constant(&mut self, value: Fr) -> Fr {
        unsafe {
            if self.write_const_inds != self.write_const_end
                && *self.write_const_inds == self.num_const_idx
            {
                self.write_const_inds = self.write_const_inds.add(1);
                self.push(value);
            }
        }
        self.num_const_idx += 1;
        value
    }
}

// --- per-opcode run logic ---

fn bb_div_run<T: ReplayTape>(t: &mut T, a_val: Fr, b_val: Fr, a_bits: u16, b_bits: u16) {
    let mut a = t.bb_external(a_val, a_bits);
    let b = t.bb_external(b_val, b_bits);
    let b_bb = to_baby_bear(&b_val);
    let b_inv_val = b_bb.try_inverse().unwrap();
    let b_inv = t.bb_load_witness(b_inv_val);
    let one = t.bb_load_constant(BabyBear::ONE);
    // `b` is passed to `mul` by value, so reduces inside do not affect the outer `b`.
    let inv_prod = t.bb_mul(b, b_inv);
    t.bb_assert_equal(inv_prod, one);
    let mut c = t.bb_load_witness(to_baby_bear(&a_val) * b_inv_val);
    if a.max_bits + 1 > REDUCE_THRESHOLD {
        a = t.bb_reduce(a);
    }
    let mut b = b;
    if b.max_bits + c.max_bits + 1 > REDUCE_THRESHOLD {
        b = t.bb_reduce(b);
    }
    if b.max_bits + c.max_bits + 1 > REDUCE_THRESHOLD {
        c = t.bb_reduce(c);
    }
    let diff = t.gate_sub_mul(a.cell.value(), b.cell.value(), c.cell.value());
    let max_bits = a.max_bits.max(b.max_bits + c.max_bits) + 1;
    t.bb_assert_zero(BbWire {
        cell: diff,
        max_bits,
    });
    t.output(c.cell);
    t.output(one.cell);
}

fn ext_mul_run<T: ReplayTape>(t: &mut T, args: &[Fr], bits: &[u16]) {
    let mut a: [BbWire<T::TapeCell>; 4] = core::array::from_fn(|i| t.bb_external(args[i], bits[i]));
    let mut b: [BbWire<T::TapeCell>; 4] =
        core::array::from_fn(|i| t.bb_external(args[4 + i], bits[4 + i]));
    let (coeffs, w) = t.ext_mul(&mut a, &mut b);
    for coeff in &coeffs {
        t.output(coeff.cell);
    }
    t.output(w);
}

fn ext_div_run<T: ReplayTape>(t: &mut T, args: &[Fr], bits: &[u16]) {
    let a: [BbWire<T::TapeCell>; 4] = core::array::from_fn(|i| t.bb_external(args[i], bits[i]));
    let b: [BbWire<T::TapeCell>; 4] =
        core::array::from_fn(|i| t.bb_external(args[4 + i], bits[4 + i]));
    let b_ext = ext_value(&b);
    let b_inv_val = b_ext.try_inverse().unwrap();
    let b_inv_coeffs = b_inv_val.as_basis_coefficients_slice();
    let b_inv: [BbWire<T::TapeCell>; 4] =
        core::array::from_fn(|i| t.bb_load_witness(b_inv_coeffs[i]));
    // ext load_constant(ONE) = coeffs [1, 0, 0, 0]: ONE and the first ZERO may
    // materialize; the remaining zeros always hit the constant cache.
    let one_c = t.bb_load_constant(BabyBear::ONE);
    let zero_c = t.bb_load_constant(BabyBear::ZERO);
    let _ = t.bb_load_constant(BabyBear::ZERO);
    let _ = t.bb_load_constant(BabyBear::ZERO);
    let one_ext = [one_c, zero_c, zero_c, zero_c];
    // `mul` takes its ext operands by value, so pass copies; W materializes here
    // (or hits a pre-seeded cache).
    let (inv_prod, w) = t.ext_mul(&mut { b }, &mut { b_inv });
    for i in 0..4 {
        t.bb_assert_equal(inv_prod[i], one_ext[i]);
    }
    let a_ext = ext_value(&a);
    let c_val = a_ext * b_inv_val;
    let c_coeffs = c_val.as_basis_coefficients_slice();
    let c: [BbWire<T::TapeCell>; 4] = core::array::from_fn(|i| t.bb_load_witness(c_coeffs[i]));
    // Second mul uses the outer `b` at its original bit bounds; W is now cached.
    let (prod, _) = t.ext_mul(&mut { b }, &mut { c });
    for i in 0..4 {
        t.bb_assert_equal(a[i], prod[i]);
    }
    for coeff in &c {
        t.output(coeff.cell);
    }
    t.output(one_c.cell);
    t.output(zero_c.cell);
    t.output(w);
}

fn poseidon_x_power5<T: ReplayTape>(t: &mut T, x: T::TapeCell) -> T::TapeCell {
    let x2 = t.gate_mul(x.value(), x.value());
    let x4 = t.gate_mul(x2.value(), x2.value());
    t.gate_mul(x.value(), x4.value())
}

fn poseidon_sbox<T: ReplayTape, const N: usize>(t: &mut T, state: &mut [T::TapeCell; N]) {
    for x in state.iter_mut() {
        *x = poseidon_x_power5(t, *x);
    }
}

fn poseidon_add_rc<T: ReplayTape, const N: usize>(
    t: &mut T,
    state: &mut [T::TapeCell; N],
    rc: &[Fr; N],
) {
    for (x, rc) in state.iter_mut().zip(rc.iter()) {
        *x = t.gate_add(x.value(), *rc);
    }
}

fn poseidon_matmul_external<T: ReplayTape, const N: usize>(
    t: &mut T,
    state: &mut [T::TapeCell; N],
) {
    let sum = t.gate_sum(state.map(|c| c.value()));
    for (i, x) in state.iter_mut().enumerate() {
        let new_x = x.value() + sum.value();
        if i % 2 == 0 {
            // `| new_x | x | -1 | sum |`, out at +0
            let out = t.push(new_x);
            t.push(x.value());
            t.push(-Fr::ONE);
            t.push(sum.value());
            *x = out;
        } else {
            // `| x | 1 | new_x |`, out at +2
            t.push(x.value());
            t.push(Fr::ONE);
            *x = t.push(new_x);
        }
    }
}

fn poseidon_matmul_internal<T: ReplayTape, const N: usize>(
    t: &mut T,
    state: &mut [T::TapeCell; N],
    diag: &[Fr; N],
) {
    let sum = t.gate_sum(state.map(|c| c.value()));
    for i in 0..N {
        let new_s = state[i].value() * diag[i] + sum.value();
        if i % 2 == 0 {
            // `| new_s | s_i | -diag_i | sum |`, out at +0
            let out = t.push(new_s);
            t.push(state[i].value());
            t.push(-diag[i]);
            t.push(sum.value());
            state[i] = out;
        } else {
            // `| s_i | diag_i | new_s |`, out at +2
            t.push(state[i].value());
            t.push(diag[i]);
            state[i] = t.push(new_s);
        }
    }
}

fn poseidon_permute_run<T: ReplayTape, const N: usize>(
    t: &mut T,
    args: &[Fr],
    params: &Poseidon2Params<Fr, N>,
) {
    let mut state: [T::TapeCell; N] = core::array::from_fn(|i| t.external(args[i]));
    let rounds_f_beginning = params.rounds_f / 2;
    poseidon_matmul_external(t, &mut state);
    for r in 0..rounds_f_beginning {
        poseidon_add_rc(t, &mut state, &params.external_rc[r]);
        poseidon_sbox(t, &mut state);
        poseidon_matmul_external(t, &mut state);
    }
    for r in 0..params.rounds_p {
        state[0] = t.gate_add(state[0].value(), params.internal_rc[r]);
        state[0] = poseidon_x_power5(t, state[0]);
        poseidon_matmul_internal(t, &mut state, &params.mat_internal_diag_m_1);
    }
    for r in rounds_f_beginning..params.rounds_f {
        poseidon_add_rc(t, &mut state, &params.external_rc[r]);
        poseidon_sbox(t, &mut state);
        poseidon_matmul_external(t, &mut state);
    }
    for cell in state {
        t.output(cell);
    }
}

/// Mirrors `decompose_bn254_to_base_baby_bear_digits`: 6 hint cells (5 digits + top
/// quotient) followed by per-digit canonicity checks, recomposition, and boundary
/// checks. Outputs `0..NUM_SAMPLES_PER_WORD` are the digit cells.
fn decompose_run<T: ReplayTape>(t: &mut T, packed: Fr) {
    let p = BigUint::from(BABY_BEAR_MODULUS_U64);
    let one = BigUint::from(1u64);
    let q = modulus::<Fr>();
    let q_minus_one = &q - &one;
    let pow_k = p.pow(NUM_SAMPLES_PER_WORD as u32);
    let top_quotient_max = &q_minus_one / &pow_k;
    let lower_max_plus_one = &q_minus_one - &top_quotient_max * &pow_k + &one;

    // Hint witnesses: digits then top quotient.
    let mut value = fe_to_biguint(&packed);
    let digit_values: [Fr; NUM_SAMPLES_PER_WORD] = core::array::from_fn(|_| {
        let digit = &value % &p;
        value /= &p;
        biguint_to_fe(&digit)
    });
    let top_quotient_value: Fr = biguint_to_fe(&value);
    for &digit in &digit_values {
        let cell = t.push(digit);
        t.output(cell);
    }
    let top_quotient = t.push(top_quotient_value);

    for &digit in &digit_values {
        t.check_less_than_safe(digit, BABY_BEAR_MODULUS_U64);
    }
    t.is_big_less_than_safe(top_quotient.value(), &(&top_quotient_max + &one));
    // assert_is_const: no advice cells

    // lower = sum(digit_i * p^i); `p^0 == 1`, so the inner product starts with one.
    let mut power = one.clone();
    let lower = t.gate_inner_product(
        digit_values.iter().map(|&digit| {
            let coeff = biguint_to_fe(&power);
            power *= &p;
            (digit, coeff)
        }),
        true,
    );

    // packed == top_quotient * p^k + lower
    t.gate_mul_add(top_quotient.value(), biguint_to_fe(&pow_k), lower.value());
    // constrain_equal: no advice cells

    let at_top_boundary = t.gate_is_equal(top_quotient.value(), biguint_to_fe(&top_quotient_max));
    let lower_range_bits = (pow_k.bits() as usize).div_ceil(t.lookup_bits()) * t.lookup_bits();
    t.range_check(lower.value(), lower_range_bits);
    let lower_is_valid = t.is_less_than(
        lower.value(),
        biguint_to_fe(&lower_max_plus_one),
        lower_range_bits,
    );
    let lower_is_invalid = t.gate_not(lower_is_valid.value());
    t.gate_mul(at_top_boundary.value(), lower_is_invalid.value());
    // assert_is_const: no advice cells
}

/// Executes one opcode against a replay tape. `args` are the operand values in
/// order; `bits` are the operand bit bounds (used by the BabyBear ops).
pub(crate) fn run_op<T: ReplayTape>(t: &mut T, opcode: &Halo2Opcode, args: &[Fr], bits: &[u16]) {
    match *opcode {
        Halo2Opcode::Const => {
            let out = t.push(args[0]);
            t.output(out);
        }
        Halo2Opcode::Select => {
            let out = t.gate_select(args[0], args[1], args[2]);
            t.output(out);
        }
        Halo2Opcode::Num2Bits(n) => t.gate_num_to_bits(args[0], n as usize),
        Halo2Opcode::BBReduce => {
            let a = t.bb_external(args[0], bits[0]);
            let out = t.bb_reduce(a);
            t.output(out.cell);
        }
        // The IR builder pre-reduces operands, so these are the pure gates (and
        // `BBMul`/`BBMulAdd` operands arrive post-swap).
        Halo2Opcode::BBAdd => {
            let out = t.gate_add(args[0], args[1]);
            t.output(out);
        }
        Halo2Opcode::BBNeg => {
            let out = t.gate_neg(args[0]);
            t.output(out);
        }
        Halo2Opcode::BBSub => {
            let out = t.gate_sub(args[0], args[1]);
            t.output(out);
        }
        Halo2Opcode::BBMul => {
            let out = t.gate_mul(args[0], args[1]);
            t.output(out);
        }
        Halo2Opcode::BBMulAdd => {
            let out = t.gate_mul_add(args[0], args[1], args[2]);
            t.output(out);
        }
        Halo2Opcode::BBDiv => bb_div_run(t, args[0], args[1], bits[0], bits[1]),
        Halo2Opcode::BBAssertZero => {
            let a = t.bb_external(args[0], bits[0]);
            t.bb_assert_zero(a);
        }
        Halo2Opcode::ExtMul => ext_mul_run(t, args, bits),
        Halo2Opcode::ExtDiv => ext_div_run(t, args, bits),
        Halo2Opcode::PoseidonPermute2T2 => {
            poseidon_permute_run(t, args, &POSEIDON2_COMPRESS_PARAMS)
        }
        Halo2Opcode::PoseidonPermute2T3 => poseidon_permute_run(t, args, &POSEIDON2_PARAMS),
        Halo2Opcode::LoadWitness => {
            let out = t.push(args.first().copied().unwrap_or(Fr::ONE));
            t.output(out);
        }
        Halo2Opcode::LoadBBReducedWitness => {
            let out = t.push(args.first().copied().unwrap_or(Fr::ONE));
            t.check_less_than_safe(out.value(), BABY_BEAR_MODULUS_U64);
            t.output(out);
        }
        Halo2Opcode::InnerProduct(_) => {
            // Operands are interleaved `[v_0, c_0, v_1, c_1, ...]`; the gate starts
            // with one exactly when the first coefficient is the constant ONE.
            let starts_with_one = args[1] == Fr::ONE;
            let out = t.gate_inner_product(
                args.chunks_exact(2).map(|pair| (pair[0], pair[1])),
                starts_with_one,
            );
            t.output(out);
        }
        Halo2Opcode::DecomposeBn254ToBabyBear => decompose_run(t, args[0]),
        Halo2Opcode::RangeDiv(n) => {
            let rem = t.div_mod(
                args[0],
                &(BigUint::from(1u32) << (n as usize)),
                BABYBEAR_MAX_BITS,
            );
            t.output(rem);
        }
    }
}

/// Shape of one opcode's tape footprint for a given constant-cache state.
pub(crate) struct OpcodeMeta {
    /// Relative context-tape offset of each logical output, in output-index order
    /// ([`UNMATERIALIZED`] for constants that hit the cache).
    pub output_offsets: Vec<usize>,
    /// Number of context-tape (advice) slots the op appends.
    pub ctx_len: usize,
    /// Number of range-tape (lookup) slots the op appends.
    pub lookups_len: usize,
    /// Indices of the `load_constant` calls that write a cell (cache misses),
    /// in call order; feed to [`WitnessTape::new`] at replay time.
    pub constant_skip_inds: Vec<u32>,
}

/// Derives [`OpcodeMeta`] by replaying the op on a [`CalculateOffsetsTape`] whose
/// constant cache is pre-seeded with `warm` (the constants already materialized
/// by earlier nodes). The tape shape depends only on the operand bit bounds,
/// constant argument values, `lookup_bits`, and the warm set — never on runtime
/// cell values.
pub(crate) fn derive_opcode_metadata(
    opcode: &Halo2Opcode,
    args: &[Fr],
    bits: &[u16],
    lookup_bits: usize,
    warm: &[Fr],
) -> OpcodeMeta {
    let mut tape = CalculateOffsetsTape::new(lookup_bits, warm);
    run_op(&mut tape, opcode, args, bits);
    OpcodeMeta {
        output_offsets: tape.outputs.iter().map(|c| c.offset).collect(),
        ctx_len: tape.advice.len(),
        lookups_len: tape.lookups.len(),
        constant_skip_inds: tape.skip_inds,
    }
}

/// Replays the op at runtime, writing the advice stream into `ctx` and the lookup
/// stream into `lookups`. Both buffers must be at least as long as the
/// corresponding [`OpcodeMeta`] lengths, and `write_const_inds` must be the
/// node's [`OpcodeMeta::constant_skip_inds`].
pub(crate) fn interpret_op(
    opcode: &Halo2Opcode,
    args: &[Fr],
    bits: &[u16],
    ctx: &mut [Fr],
    lookups: &mut [Fr],
    lookup_bits: usize,
    write_const_inds: &[u32],
) {
    let mut tape = WitnessTape::new(ctx, lookups, lookup_bits, write_const_inds);
    run_op(&mut tape, opcode, args, bits);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use halo2_base::{
        gates::{
            circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
            GateInstructions, RangeChip, RangeInstructions,
        },
        halo2_proofs::arithmetic::Field as _,
        AssignedValue, Context, QuantumCell,
    };

    use super::*;
    use crate::{
        field::baby_bear::{BabyBearChip, BabyBearExt4Chip, BabyBearExt4Wire, BabyBearWire},
        hash::poseidon2::Poseidon2State,
        transcript::decompose_bn254_to_base_baby_bear_digits,
    };

    const LOOKUP_BITS: usize = 11;

    /// Flattens the range tape: every value sent to `add_cell_to_lookup`, in order.
    /// All tests use a single context, so the lookup manager holds a single tag.
    fn lookup_tape(range: &RangeChip<Fr>) -> Vec<Fr> {
        let map = range.lookup_manager()[0].cells_to_lookup.lock().unwrap();
        assert!(map.len() <= 1, "expected a single context tag");
        map.values()
            .flat_map(|cells| cells.iter().map(|c| c[0].value.evaluate()))
            .collect()
    }

    /// `2^bits - 1 - salt`, an "adversarially large" value for a given bit bound.
    fn big_val(bits: usize, salt: u64) -> Fr {
        biguint_to_fe(&((BigUint::from(1u32) << bits) - 1u32 - salt))
    }

    /// A raw (unconstrained) wire with an externally asserted bit bound, standing in
    /// for the output of an earlier node.
    fn raw_wire(ctx: &mut Context<Fr>, value: Fr, max_bits: usize) -> BabyBearWire {
        BabyBearWire {
            value: ctx.load_witness(value),
            max_bits,
        }
    }

    fn bb_ext(vals: [u32; 4]) -> BabyBearExt4 {
        BabyBearExt4::from_basis_coefficients_fn(|i| BabyBear::from_u32(vals[i]))
    }

    /// Runs the real halo2 op and the tape replay and compares them bit for bit.
    ///
    /// `run_real` loads its own inputs (and pre-warms constant caches for every
    /// value in `warm`), records the tape start positions, runs the chip op, and
    /// returns `(ctx_start, range_start, outputs)` with the outputs in logical
    /// output-index order. The replay runs on a [`CalculateOffsetsTape`] seeded
    /// with `warm`; [`derive_opcode_metadata`] and [`interpret_op`] (driven by the
    /// derived constant-skip indices) are also checked against the real tapes.
    fn check_opcode(
        lookup_bits: usize,
        opcode: Halo2Opcode,
        args: &[Fr],
        bits: &[u16],
        warm: &[Fr],
        run_real: impl FnOnce(
            &mut Context<Fr>,
            &Arc<RangeChip<Fr>>,
        ) -> (usize, usize, Vec<AssignedValue<Fr>>),
    ) {
        let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
            .use_k(lookup_bits + 1)
            .use_lookup_bits(lookup_bits);
        let range = Arc::new(builder.range_chip());
        let ctx = builder.main(0);

        let (ctx_start, range_start, outputs) = run_real(ctx, &range);
        let name = opcode.name();
        let real_ctx: Vec<Fr> = ctx.advice_cells()[ctx_start..]
            .iter()
            .map(|a| a.evaluate())
            .collect();
        let real_range: Vec<Fr> = lookup_tape(&range)[range_start..].to_vec();

        let mut tape = CalculateOffsetsTape::new(lookup_bits, warm);
        run_op(&mut tape, &opcode, args, bits);
        assert_eq!(tape.advice, real_ctx, "{name}: context tape");
        assert_eq!(tape.lookups, real_range, "{name}: range tape");

        assert_eq!(
            outputs.len(),
            tape.outputs.len(),
            "{name}: number of outputs"
        );
        for (i, val) in outputs.iter().enumerate() {
            let out = tape.outputs[i];
            assert_eq!(val.value.evaluate(), out.value, "{name}: output {i} value");
            let real_offset = val.cell.unwrap().offset;
            if out.offset == UNMATERIALIZED {
                assert!(
                    real_offset < ctx_start,
                    "{name}: output {i} should be a pre-warmed cell"
                );
            } else {
                assert_eq!(
                    real_offset,
                    ctx_start + out.offset,
                    "{name}: output {i} offset"
                );
            }
        }

        let meta = derive_opcode_metadata(&opcode, args, bits, lookup_bits, warm);
        assert_eq!(meta.ctx_len, real_ctx.len(), "{name}: meta ctx_len");
        assert_eq!(
            meta.lookups_len,
            real_range.len(),
            "{name}: meta lookups_len"
        );
        let offsets: Vec<usize> = tape.outputs.iter().map(|c| c.offset).collect();
        assert_eq!(meta.output_offsets, offsets, "{name}: meta output_offsets");
        assert_eq!(
            meta.constant_skip_inds, tape.skip_inds,
            "{name}: meta constant_skip_inds"
        );

        let mut ctx_buf = vec![Fr::ZERO; meta.ctx_len];
        let mut range_buf = vec![Fr::ZERO; meta.lookups_len];
        interpret_op(
            &opcode,
            args,
            bits,
            &mut ctx_buf,
            &mut range_buf,
            lookup_bits,
            &meta.constant_skip_inds,
        );
        assert_eq!(ctx_buf, real_ctx, "{name}: interpret_op context tape");
        assert_eq!(range_buf, real_range, "{name}: interpret_op range tape");
    }

    #[test]
    fn const_matches_backend() {
        for v in [Fr::from(42u64), Fr::ZERO] {
            check_opcode(LOOKUP_BITS, Halo2Opcode::Const, &[v], &[], &[], |ctx, _| {
                let start = ctx.advice_cells().len();
                let out = if v == Fr::ZERO {
                    ctx.load_zero()
                } else {
                    ctx.load_constant(v)
                };
                (start, 0, vec![out])
            });
        }
    }

    #[test]
    fn select_matches_backend() {
        for sel in [Fr::ZERO, Fr::ONE] {
            let (a, b) = (Fr::from(1234u64), Fr::from(5678u64));
            check_opcode(
                LOOKUP_BITS,
                Halo2Opcode::Select,
                &[a, b, sel],
                &[],
                &[],
                |ctx, range| {
                    let av = ctx.load_witness(a);
                    let bv = ctx.load_witness(b);
                    let sv = ctx.load_witness(sel);
                    let start = ctx.advice_cells().len();
                    let out = range.gate().select(ctx, av, bv, sv);
                    (start, 0, vec![out])
                },
            );
        }
    }

    #[test]
    fn num2bits_matches_backend() {
        for (n, v) in [(1u16, Fr::ONE), (16, Fr::from(0xABCDu64))] {
            check_opcode(
                LOOKUP_BITS,
                Halo2Opcode::Num2Bits(n),
                &[v],
                &[],
                &[],
                |ctx, range| {
                    let a = ctx.load_witness(v);
                    let start = ctx.advice_cells().len();
                    let outputs = range.gate().num_to_bits(ctx, a, n as usize);
                    (start, 0, outputs)
                },
            );
        }
    }

    fn run_bb_reduce(lookup_bits: usize, bits: u16, v: Fr) {
        check_opcode(
            lookup_bits,
            Halo2Opcode::BBReduce,
            &[v],
            &[bits],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let wire = raw_wire(ctx, v, bits as usize);
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let out = chip.reduce(ctx, wire);
                (start, range_start, vec![out.value])
            },
        );
    }

    #[test]
    fn bb_reduce_matches_backend() {
        run_bb_reduce(LOOKUP_BITS, 32, big_val(32, 7));
        run_bb_reduce(LOOKUP_BITS, 60, big_val(60, 99));
        run_bb_reduce(LOOKUP_BITS, 200, big_val(200, 5));
        // negative value: |v| = 123456789 < 2^30
        run_bb_reduce(LOOKUP_BITS, 30, -Fr::from(123456789u64));
        // different lookup_bits changes limb decompositions
        run_bb_reduce(17, 60, big_val(60, 99));
    }

    #[test]
    fn bb_add_matches_backend() {
        let (va, vb) = (big_val(100, 3), big_val(60, 11));
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::BBAdd,
            &[va, vb],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let a = raw_wire(ctx, va, 100);
                let b = raw_wire(ctx, vb, 60);
                let start = ctx.advice_cells().len();
                let out = chip.add(ctx, a, b);
                (start, 0, vec![out.value])
            },
        );
    }

    #[test]
    fn bb_neg_matches_backend() {
        let va = big_val(100, 3);
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::BBNeg,
            &[va],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let a = raw_wire(ctx, va, 100);
                let start = ctx.advice_cells().len();
                let out = chip.neg(ctx, a);
                (start, 0, vec![out.value])
            },
        );
    }

    #[test]
    fn bb_sub_matches_backend() {
        let (va, vb) = (big_val(100, 3), big_val(60, 11));
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::BBSub,
            &[va, vb],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let a = raw_wire(ctx, va, 100);
                let b = raw_wire(ctx, vb, 60);
                let start = ctx.advice_cells().len();
                let out = chip.sub(ctx, a, b);
                (start, 0, vec![out.value])
            },
        );
    }

    #[test]
    fn bb_mul_matches_backend() {
        // a_bits >= b_bits: the IR builder emits operands post-swap.
        let (va, vb) = (big_val(100, 3), big_val(60, 11));
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::BBMul,
            &[va, vb],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let a = raw_wire(ctx, va, 100);
                let b = raw_wire(ctx, vb, 60);
                let start = ctx.advice_cells().len();
                let out = chip.mul(ctx, a, b);
                (start, 0, vec![out.value])
            },
        );
    }

    #[test]
    fn bb_mul_add_matches_backend() {
        let (va, vb, vc) = (big_val(100, 3), big_val(60, 11), big_val(90, 27));
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::BBMulAdd,
            &[va, vb, vc],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let a = raw_wire(ctx, va, 100);
                let b = raw_wire(ctx, vb, 60);
                let c = raw_wire(ctx, vc, 90);
                let start = ctx.advice_cells().len();
                let out = chip.mul_add(ctx, a, b, c);
                (start, 0, vec![out.value])
            },
        );
    }

    fn run_bb_div(a_bits: u16, b_bits: u16, va: Fr, vb: Fr, prewarm_one: bool) {
        let warm: &[Fr] = if prewarm_one { &[Fr::ONE] } else { &[] };
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::BBDiv,
            &[va, vb],
            &[a_bits, b_bits],
            warm,
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                if prewarm_one {
                    chip.load_constant(ctx, BabyBear::ONE);
                }
                let a = raw_wire(ctx, va, a_bits as usize);
                let b = raw_wire(ctx, vb, b_bits as usize);
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let out = chip.div(ctx, a, b);
                // Cache hit: recovers the ONE cell (assigned inside `div` when it
                // was not pre-warmed).
                let one = chip.load_constant(ctx, BabyBear::ONE);
                (start, range_start, vec![out.value, one.value])
            },
        );
    }

    #[test]
    fn bb_div_matches_backend() {
        run_bb_div(31, 31, Fr::from(123456u64), Fr::from(654321u64), false);
        run_bb_div(31, 31, Fr::from(123456u64), Fr::from(654321u64), true);
        // large operands exercise the internal reduces
        run_bb_div(240, 230, big_val(240, 17), big_val(230, 23), false);
    }

    #[test]
    fn bb_assert_zero_matches_backend() {
        let p = Fr::from(BABY_BEAR_MODULUS_U64);
        for (v, bits) in [(p * Fr::from(3u64), 34u16), (-(p * Fr::from(5u64)), 35)] {
            check_opcode(
                LOOKUP_BITS,
                Halo2Opcode::BBAssertZero,
                &[v],
                &[bits],
                &[],
                |ctx, range| {
                    let chip = BabyBearChip::new(range.clone());
                    let a = raw_wire(ctx, v, bits as usize);
                    let start = ctx.advice_cells().len();
                    let range_start = lookup_tape(range).len();
                    chip.assert_zero(ctx, a);
                    (start, range_start, vec![])
                },
            );
        }
    }

    const W_BB: BabyBear = <BabyBear as BinomiallyExtendable<4>>::W;

    fn run_ext_mul(a_bits: [u16; 4], a_vals: [Fr; 4], prewarm_w: bool) {
        let b_vals = [5u64, 6, 7, 8].map(Fr::from);
        let warm = if prewarm_w {
            vec![fr_from_bb(W_BB)]
        } else {
            vec![]
        };
        let bits: [u16; 8] = core::array::from_fn(|i| if i < 4 { a_bits[i] } else { 31 });
        let args: Vec<Fr> = a_vals.iter().chain(b_vals.iter()).copied().collect();
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::ExtMul,
            &args,
            &bits,
            &warm,
            |ctx, range| {
                let chip = BabyBearExt4Chip::new(BabyBearChip::new(range.clone()));
                if prewarm_w {
                    chip.base.load_constant(ctx, W_BB);
                }
                let a = BabyBearExt4Wire(core::array::from_fn(|i| {
                    raw_wire(ctx, a_vals[i], a_bits[i] as usize)
                }));
                let b = BabyBearExt4Wire(core::array::from_fn(|i| raw_wire(ctx, b_vals[i], 31)));
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let out = chip.mul(ctx, a, b);
                let mut outputs: Vec<AssignedValue<Fr>> = out.0.iter().map(|w| w.value).collect();
                // Cache hit: recovers the W cell.
                outputs.push(chip.base.load_constant(ctx, W_BB).value);
                (start, range_start, outputs)
            },
        );
    }

    #[test]
    fn ext_mul_matches_backend() {
        let small = [1u64, 2, 3, 4].map(Fr::from);
        run_ext_mul([31; 4], small, false);
        run_ext_mul([31; 4], small, true);
        // a wide first coefficient exercises the special_inner_product reduces
        run_ext_mul(
            [230, 31, 31, 31],
            [
                big_val(230, 9),
                Fr::from(2u64),
                Fr::from(3u64),
                Fr::from(4u64),
            ],
            false,
        );
    }

    fn run_ext_div(prewarm: bool) {
        let a_vals = [1u32, 2, 3, 4];
        let b_vals = [5u32, 6, 7, 8];
        let a_ext = bb_ext(a_vals);
        let b_ext = bb_ext(b_vals);
        let warm = if prewarm {
            vec![Fr::ONE, Fr::ZERO, fr_from_bb(W_BB)]
        } else {
            vec![]
        };
        let args: Vec<Fr> = a_vals
            .iter()
            .chain(b_vals.iter())
            .map(|&v| Fr::from(v as u64))
            .collect();
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::ExtDiv,
            &args,
            &[31; 8],
            &warm,
            |ctx, range| {
                let chip = BabyBearExt4Chip::new(BabyBearChip::new(range.clone()));
                if prewarm {
                    chip.base.load_constant(ctx, BabyBear::ONE);
                    chip.base.load_constant(ctx, BabyBear::ZERO);
                    chip.base.load_constant(ctx, W_BB);
                }
                let a = chip.load_witness(ctx, a_ext);
                let b = chip.load_witness(ctx, b_ext);
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let out = chip.div(ctx, a, b);
                let mut outputs: Vec<AssignedValue<Fr>> = out.0.iter().map(|w| w.value).collect();
                // Cache hits recover the constant cells.
                outputs.push(chip.base.load_constant(ctx, BabyBear::ONE).value);
                outputs.push(chip.base.load_constant(ctx, BabyBear::ZERO).value);
                outputs.push(chip.base.load_constant(ctx, W_BB).value);
                (start, range_start, outputs)
            },
        );
    }

    #[test]
    fn ext_div_matches_backend() {
        run_ext_div(false);
        run_ext_div(true);
    }

    #[test]
    fn poseidon_t3_matches_backend() {
        let vals = [11u64, 22, 33].map(Fr::from);
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::PoseidonPermute2T3,
            &vals,
            &[],
            &[],
            |ctx, range| {
                let s: [AssignedValue<Fr>; 3] = core::array::from_fn(|i| ctx.load_witness(vals[i]));
                let start = ctx.advice_cells().len();
                let mut state = Poseidon2State::new(s);
                state.permutation(ctx, range.gate(), &POSEIDON2_PARAMS);
                (start, 0, state.s.to_vec())
            },
        );
    }

    #[test]
    fn poseidon_t2_matches_backend() {
        let vals = [44u64, 55].map(Fr::from);
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::PoseidonPermute2T2,
            &vals,
            &[],
            &[],
            |ctx, range| {
                let s: [AssignedValue<Fr>; 2] = core::array::from_fn(|i| ctx.load_witness(vals[i]));
                let start = ctx.advice_cells().len();
                let mut state = Poseidon2State::new(s);
                state.permutation(ctx, range.gate(), &POSEIDON2_COMPRESS_PARAMS);
                (start, 0, state.s.to_vec())
            },
        );
    }

    #[test]
    fn load_witness_matches_backend() {
        let v = Fr::from(777u64);
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::LoadWitness,
            &[v],
            &[],
            &[],
            |ctx, _| {
                let start = ctx.advice_cells().len();
                let out = ctx.load_witness(v);
                (start, 0, vec![out])
            },
        );
    }

    #[test]
    fn load_bb_reduced_witness_matches_backend() {
        let v = BabyBear::from_u32(1234567);
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::LoadBBReducedWitness,
            &[fr_from_bb(v)],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let out = chip.load_reduced_witness(ctx, v);
                (start, range_start, vec![out.value()])
            },
        );
    }

    fn run_inner_product(n: usize, starts_with_one: bool) {
        let vals: Vec<Fr> = (0..n).map(|i| Fr::from(100 + i as u64)).collect();
        let coeffs: Vec<Fr> = (0..n)
            .map(|i| {
                if i == 0 && starts_with_one {
                    Fr::ONE
                } else {
                    Fr::from(7 + 3 * i as u64)
                }
            })
            .collect();
        let args: Vec<Fr> = vals
            .iter()
            .zip(&coeffs)
            .flat_map(|(&v, &c)| [v, c])
            .collect();
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::InnerProduct(n as u16),
            &args,
            &[],
            &[],
            |ctx, range| {
                let loaded: Vec<AssignedValue<Fr>> =
                    vals.iter().map(|&v| ctx.load_witness(v)).collect();
                let start = ctx.advice_cells().len();
                let out = range.gate().inner_product(
                    ctx,
                    loaded.iter().map(|v| QuantumCell::Existing(*v)),
                    coeffs.iter().map(|&c| QuantumCell::Constant(c)),
                );
                (start, 0, vec![out])
            },
        );
    }

    #[test]
    fn inner_product_matches_backend() {
        run_inner_product(1, true);
        run_inner_product(1, false);
        run_inner_product(3, true);
        run_inner_product(3, false);
    }

    fn run_decompose(v: Fr) {
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::DecomposeBn254ToBabyBear,
            &[v],
            &[],
            &[],
            |ctx, range| {
                let chip = BabyBearChip::new(range.clone());
                let packed = ctx.load_witness(v);
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let wires = decompose_bn254_to_base_baby_bear_digits(ctx, &chip, packed);
                let outputs = wires.iter().map(|w| w.value).collect();
                (start, range_start, outputs)
            },
        );
    }

    #[test]
    fn decompose_matches_backend() {
        run_decompose(big_val(250, 12345));
        // near the top boundary of the Bn254 field
        run_decompose(-Fr::from(2u64));
    }

    #[test]
    fn range_div_matches_backend() {
        let v = Fr::from(1234567u64);
        let bits = 16u16;
        check_opcode(
            LOOKUP_BITS,
            Halo2Opcode::RangeDiv(bits),
            &[v],
            &[],
            &[],
            |ctx, range| {
                let a = ctx.load_witness(v);
                let start = ctx.advice_cells().len();
                let range_start = lookup_tape(range).len();
                let (_, rem) = range.div_mod(
                    ctx,
                    a,
                    BigUint::from(1u64) << (bits as usize),
                    BABYBEAR_MAX_BITS,
                );
                (start, range_start, vec![rem])
            },
        );
    }
}
