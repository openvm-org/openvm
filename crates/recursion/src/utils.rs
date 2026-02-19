use std::ops::Index;

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::FiatShamirTranscript;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_perm, BabyBearPoseidon2Config, CHUNK, D_EF, F,
};
use p3_air::AirBuilder;
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing};
use p3_symmetric::Permutation;

/// Returns the number of transcript slots consumed by a proof-of-work check.
///
/// When `pow_bits > 0`, PoW uses 2 transcript slots (1 observe + 1 sample).
/// When `pow_bits == 0`, the stark-backend's `check_witness`/`grind` skip
/// observe/sample entirely, consuming 0 slots.
#[inline]
pub const fn pow_tidx_count(pow_bits: usize) -> usize {
    if pow_bits > 0 {
        2
    } else {
        0
    }
}

/// Runs the PoW observe/sample in a preflight transcript, returning the sample
/// (or `F::ZERO` when `pow_bits == 0`).
pub fn pow_observe_sample(
    ts: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    pow_bits: usize,
    witness: F,
) -> F {
    if pow_bits > 0 {
        ts.observe(witness);
        ts.sample()
    } else {
        F::ZERO
    }
}

pub fn base_to_ext<FA>(x: impl Into<FA>) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    [x.into(), FA::ZERO, FA::ZERO, FA::ZERO]
}

pub fn ext_field_one_minus<FA>(x: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    [FA::ONE - x0, -x1, -x2, -x3]
}

pub fn ext_field_add<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let [y0, y1, y2, y3] = y.map(Into::into);
    [x0 + y0, x1 + y1, x2 + y2, x3 + y3]
}

pub fn ext_field_subtract<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let [y0, y1, y2, y3] = y.map(Into::into);
    [x0 - y0, x1 - y1, x2 - y2, x3 - y3]
}

pub fn ext_field_multiply<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let [y0, y1, y2, y3] = y.map(Into::into);

    let w = FA::from_prime_subfield(FA::PrimeSubfield::W);

    let z0_beta_terms = x1.clone() * y3.clone() + x2.clone() * y2.clone() + x3.clone() * y1.clone();
    let z1_beta_terms = x2.clone() * y3.clone() + x3.clone() * y2.clone();
    let z2_beta_terms = x3.clone() * y3.clone();

    [
        x0.clone() * y0.clone() + z0_beta_terms * w.clone(),
        x0.clone() * y1.clone() + x1.clone() * y0.clone() + z1_beta_terms * w.clone(),
        x0.clone() * y2.clone()
            + x1.clone() * y1.clone()
            + x2.clone() * y0.clone()
            + z2_beta_terms * w,
        x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
    ]
}

pub fn ext_field_add_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    [x0 + y.into(), x1, x2, x3]
}

pub fn ext_field_subtract_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    [x0 - y.into(), x1, x2, x3]
}

pub fn scalar_subtract_ext_field<FA>(x: impl Into<FA>, y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [y0, y1, y2, y3] = y.map(Into::into);
    [x.into() - y0, -y1, -y2, -y3]
}

pub fn ext_field_multiply_scalar<FA>(x: [impl Into<FA>; D_EF], y: impl Into<FA>) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let [x0, x1, x2, x3] = x.map(Into::into);
    let y = y.into();
    [x0 * y.clone(), x1 * y.clone(), x2 * y.clone(), x3 * y]
}

pub fn eq_1<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    let x = x.map(Into::into);
    let y = y.map(Into::into);

    let xy = ext_field_multiply::<FA>(x.clone(), y.clone());
    let two_xy = ext_field_multiply_scalar::<FA>(xy, FA::TWO);
    let x_plus_y = ext_field_add::<FA>(x, y);
    let one_minus_x_plus_y = scalar_subtract_ext_field::<FA>(FA::ONE, x_plus_y);
    ext_field_add(one_minus_x_plus_y, two_xy)
}

/// Per-coordinate Möbius-adjusted equality kernel for eval-to-coeff RS encoding:
/// ```text
/// mobius_eq_1(u, x) = (1 - 2*u)*(1 - x) + u*x
///                   = 1 - x - 2*u + 3*u*x
/// ```
pub fn mobius_eq_1<FA>(u: [impl Into<FA>; D_EF], x: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    let omega = u.map(Into::into);
    let x = x.map(Into::into);
    // mobius_eq_1(u, x) = 1 - x - 2*u + 3*u*x
    let omega_x = ext_field_multiply::<FA>(omega.clone(), x.clone());
    let three_omega_x = ext_field_multiply_scalar::<FA>(omega_x, FA::from_u8(3));
    let two_omega = ext_field_multiply_scalar::<FA>(omega.clone(), FA::TWO);
    let x_plus_2omega = ext_field_add::<FA>(x, two_omega);
    let one_minus_rest = scalar_subtract_ext_field::<FA>(FA::ONE, x_plus_2omega);
    ext_field_add(one_minus_rest, three_omega_x)
}

pub fn assert_zeros<AB, const N: usize>(builder: &mut AB, array: [impl Into<AB::Expr>; N])
where
    AB: AirBuilder,
{
    for elem in array.into_iter() {
        builder.assert_zero(elem);
    }
}

pub fn assert_one_ext<AB>(builder: &mut AB, array: [impl Into<AB::Expr>; D_EF])
where
    AB: AirBuilder,
{
    for (i, elem) in array.into_iter().enumerate() {
        if i == 0 {
            builder.assert_one(elem);
        } else {
            builder.assert_zero(elem);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MultiProofVecVec<T> {
    data: Vec<T>,
    bounds: Vec<usize>,
}

impl<T> MultiProofVecVec<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: Vec::new(),
            bounds: vec![0],
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bounds: vec![0],
        }
    }

    pub(crate) fn push(&mut self, x: T) {
        self.data.push(x);
    }

    pub(crate) fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Clone,
    {
        self.data.extend_from_slice(slice);
    }

    pub(crate) fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
        self.data.extend(iter);
    }

    pub(crate) fn end_proof(&mut self) {
        self.bounds.push(self.data.len());
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn num_proofs(&self) -> usize {
        self.bounds.len() - 1
    }

    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }
}

impl<T> Index<usize> for MultiProofVecVec<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.num_proofs());
        &self.data[self.bounds[index]..self.bounds[index + 1]]
    }
}

#[derive(Debug, Clone)]
pub struct MultiVecWithBounds<T, const DIM_MINUS_ONE: usize> {
    pub data: Vec<T>,
    pub bounds: [Vec<usize>; DIM_MINUS_ONE],
}

impl<T, const DIM_MINUS_ONE: usize> MultiVecWithBounds<T, DIM_MINUS_ONE> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bounds: core::array::from_fn(|_| vec![0]),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bounds: core::array::from_fn(|_| vec![0]),
        }
    }

    pub fn push(&mut self, x: T) {
        self.data.push(x);
    }

    pub fn close_level(&mut self, level: usize) {
        debug_assert!(level < DIM_MINUS_ONE);
        for i in level..DIM_MINUS_ONE - 1 {
            self.bounds[i].push(self.bounds[i + 1].len());
        }
        self.bounds[DIM_MINUS_ONE - 1].push(self.data.len());
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
        self.data.extend(iter);
    }
}

impl<T, const DIM_MINUS_ONE: usize> Index<[usize; DIM_MINUS_ONE]>
    for MultiVecWithBounds<T, DIM_MINUS_ONE>
{
    type Output = [T];

    fn index(&self, index: [usize; DIM_MINUS_ONE]) -> &Self::Output {
        let mut idx = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..DIM_MINUS_ONE {
            idx += index[i];
            if i < DIM_MINUS_ONE - 1 {
                idx = self.bounds[i][idx];
            }
        }
        &self.data[self.bounds[DIM_MINUS_ONE - 1][idx]..self.bounds[DIM_MINUS_ONE - 1][idx + 1]]
    }
}

pub fn interpolate_quadratic<FA>(
    pre_claim: [impl Into<FA>; D_EF],
    ev1: [impl Into<FA>; D_EF],
    ev2: [impl Into<FA>; D_EF],
    alpha: [impl Into<FA>; D_EF],
) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    let pre_claim = pre_claim.map(Into::into);
    let ev1 = ev1.map(Into::into);
    let ev2 = ev2.map(Into::into);
    let alpha = alpha.map(Into::into);

    let ev0 = ext_field_subtract::<FA>(pre_claim.clone(), ev1.clone());
    let s1 = ext_field_subtract::<FA>(ev1.clone(), ev0.clone());
    let s2 = ext_field_subtract::<FA>(ev2.clone(), ev1.clone());
    let p = ext_field_multiply::<FA>(
        ext_field_subtract::<FA>(s2, s1.clone()),
        base_to_ext::<FA>(FA::from_prime_subfield(FA::PrimeSubfield::TWO.inverse())),
    );
    let q = ext_field_subtract::<FA>(s1, p.clone());
    ext_field_add::<FA>(
        ev0,
        ext_field_multiply::<FA>(
            alpha.clone(),
            ext_field_add::<FA>(q, ext_field_multiply::<FA>(p, alpha)),
        ),
    )
}

pub fn poseidon2_hash_slice(vals: &[F]) -> ([F; CHUNK], Vec<[F; POSEIDON2_WIDTH]>) {
    let num_chunks = vals.len().div_ceil(CHUNK);
    let mut pre_states = Vec::with_capacity(num_chunks);
    let perm = poseidon2_perm();
    let mut state = [F::ZERO; POSEIDON2_WIDTH];
    let mut i = 0;
    for &val in vals {
        state[i] = val;
        i += 1;
        if i == CHUNK {
            pre_states.push(state);
            perm.permute_mut(&mut state);
            i = 0;
        }
    }
    if i != 0 {
        pre_states.push(state);
        perm.permute_mut(&mut state);
    }
    (state[..CHUNK].try_into().unwrap(), pre_states)
}

#[inline]
pub fn poseidon2_hash_slice_with_states(
    vals: &[F],
) -> (
    [F; CHUNK],
    Vec<[F; POSEIDON2_WIDTH]>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    let num_chunks = vals.len().div_ceil(CHUNK);
    let mut pre_states = Vec::with_capacity(num_chunks);
    let mut post_states = Vec::with_capacity(num_chunks);
    let perm = poseidon2_perm();
    let mut state = [F::ZERO; POSEIDON2_WIDTH];
    let mut i = 0;
    for &val in vals {
        state[i] = val;
        i += 1;
        if i == CHUNK {
            pre_states.push(state);
            perm.permute_mut(&mut state);
            post_states.push(state);
            i = 0;
        }
    }
    if i != 0 {
        pre_states.push(state);
        perm.permute_mut(&mut state);
        post_states.push(state);
    }
    (state[..CHUNK].try_into().unwrap(), pre_states, post_states)
}
