use std::{
    cmp::{max, min},
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use num_bigint_dig::BigUint;
use p3_air::AirBuilder;
use p3_util::log2_ceil_usize;

pub mod check_carry_mod_to_zero;
pub mod check_carry_to_zero;
pub mod modular_arithmetic;
pub mod utils;

#[cfg(test)]
pub mod tests;

pub trait LimbConfig {
    fn limb_bits() -> usize;
}

#[derive(Debug, Clone)]
pub struct DefaultLimbConfig;
impl LimbConfig for DefaultLimbConfig {
    fn limb_bits() -> usize {
        8
    }
}

#[derive(Debug, Clone)]
pub struct OverflowInt<T> {
    // The limbs, e.g. [a_0, a_1, a_2, ...] , represents a_0 + a_1 x + a_2 x^2
    // T can be AB::Expr, for example when the OverflowInt represents x * y
    // a0 = x0 * y0
    // a1 = x0 * y1 + x1 * y0 ...
    pub limbs: Vec<T>,

    // Track the max abs of limbs, so we can do arithmetic on them.
    pub limb_max_abs: usize,

    // All limbs should be within [-2^max_overflow_bits, 2^max_overflow_bits)
    pub max_overflow_bits: usize,
}

// It's also a big unit similar to OverflowInt, but its limbs don't overflow.
#[derive(Debug, Clone)]
pub struct CanonicalUint<T, C: LimbConfig> {
    pub limbs: Vec<T>,
    _marker: PhantomData<C>,
}

impl<T, C: LimbConfig> CanonicalUint<T, C> {
    pub fn from_big_uint(x: &BigUint, min_limbs: Option<usize>) -> CanonicalUint<isize, C> {
        let mut x_bits = utils::big_uint_to_bits(x);
        let mut x_limbs: Vec<isize> = vec![];
        let mut limbs_len = 0;
        while !x_bits.is_empty() {
            let limb = utils::take_limb(&mut x_bits, C::limb_bits());
            x_limbs.push(limb as isize);
            limbs_len += 1;
        }
        if let Some(min_limbs) = min_limbs {
            if limbs_len < min_limbs {
                x_limbs.extend(vec![0; min_limbs - limbs_len]);
            }
        }
        CanonicalUint {
            limbs: x_limbs,
            _marker: PhantomData::<C>,
        }
    }

    pub fn from_vec(x: Vec<T>) -> CanonicalUint<T, C> {
        CanonicalUint {
            limbs: x,
            _marker: PhantomData::<C>,
        }
    }
}

impl<T, C: LimbConfig, S> From<CanonicalUint<T, C>> for OverflowInt<S>
where
    T: Into<S>,
{
    fn from(x: CanonicalUint<T, C>) -> OverflowInt<S> {
        OverflowInt {
            limbs: x.limbs.into_iter().map(|x| x.into()).collect(),
            max_overflow_bits: C::limb_bits(),
            limb_max_abs: (1 << C::limb_bits()) - 1,
        }
    }
}

impl<T> OverflowInt<T> {
    pub fn from_var_vec<AB: AirBuilder, V: Into<AB::Expr>>(
        x: Vec<V>,
        limb_bits: usize,
    ) -> OverflowInt<AB::Expr> {
        let limbs = x.into_iter().map(|x| x.into()).collect();
        OverflowInt {
            limbs,
            max_overflow_bits: limb_bits,
            limb_max_abs: (1 << limb_bits) - 1,
        }
    }
}

impl OverflowInt<isize> {
    pub fn calculate_carries(&self, limb_bits: usize) -> Vec<isize> {
        let mut carries = Vec::with_capacity(self.limbs.len());

        let mut carry = 0;
        for i in 0..self.limbs.len() {
            carry = (carry + self.limbs[i]) >> limb_bits;
            carries.push(carry);
        }
        carries
    }
}

// TODO: this doesn't work for references automatically?
impl<T> Add for OverflowInt<T>
where
    T: Add<Output = T> + Clone + Default,
{
    type Output = OverflowInt<T>;

    fn add(self, other: OverflowInt<T>) -> OverflowInt<T> {
        let len = max(self.limbs.len(), other.limbs.len());
        let mut limbs = Vec::with_capacity(len);
        let zero = T::default();
        for i in 0..len {
            let a = self.limbs.get(i).unwrap_or(&zero);
            let b = other.limbs.get(i).unwrap_or(&zero);
            limbs.push(a.clone() + b.clone());
        }
        let new_max = self.limb_max_abs + other.limb_max_abs;
        let max_bits = log2_ceil_usize(new_max);
        OverflowInt {
            limbs,
            max_overflow_bits: max_bits,
            limb_max_abs: new_max,
        }
    }
}

impl<T> Sub for OverflowInt<T>
where
    T: Sub<Output = T> + Clone + Default,
{
    type Output = OverflowInt<T>;

    fn sub(self, other: OverflowInt<T>) -> OverflowInt<T> {
        let len = max(self.limbs.len(), other.limbs.len());
        let mut limbs = Vec::with_capacity(len);
        for i in 0..len {
            let zero = T::default();
            let a = self.limbs.get(i).unwrap_or(&zero);
            let b = other.limbs.get(i).unwrap_or(&zero);
            limbs.push(a.clone() - b.clone());
        }
        let new_max = self.limb_max_abs + other.limb_max_abs;
        let max_bits = log2_ceil_usize(new_max);
        OverflowInt {
            limbs,
            max_overflow_bits: max_bits,
            limb_max_abs: new_max,
        }
    }
}

impl<T> Mul for OverflowInt<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + Default,
{
    type Output = OverflowInt<T>;

    fn mul(self, other: OverflowInt<T>) -> OverflowInt<T> {
        let len = self.limbs.len() + other.limbs.len() - 1;
        let mut limbs = vec![T::default(); len];
        for i in 0..self.limbs.len() {
            for j in 0..other.limbs.len() {
                // += doesn't work for T.
                limbs[i + j] =
                    limbs[i + j].clone() + self.limbs[i].clone() * other.limbs[j].clone();
            }
        }
        let new_max =
            self.limb_max_abs * other.limb_max_abs * min(self.limbs.len(), other.limbs.len());
        let max_bits = log2_ceil_usize(new_max);
        OverflowInt {
            limbs,
            max_overflow_bits: max_bits,
            limb_max_abs: new_max,
        }
    }
}
