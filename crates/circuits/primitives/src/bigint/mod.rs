use std::{
    cmp::{max, min},
    ops::{Add, AddAssign, Mul, MulAssign, Sub},
};

use num_bigint::BigUint;
use openvm_stark_backend::p3_util::log2_ceil_usize;

pub mod check_carry_mod_to_zero;
pub mod check_carry_to_zero;
pub mod utils;

#[derive(Debug, Clone)]
pub struct OverflowInt<T> {
    // The limbs, e.g. [a_0, a_1, a_2, ...] , represents a_0 + a_1 x + a_2 x^2
    // T can be AB::Expr, for example when the OverflowInt represents x * y
    // a0 = x0 * y0
    // a1 = x0 * y1 + x1 * y0 ...
    limbs: Vec<T>,

    // Track the max abs of limbs, so we can do arithmetic on them.
    limb_max_abs: usize,

    // All limbs should be within [-2^max_overflow_bits, 2^max_overflow_bits)
    max_overflow_bits: usize,
}

impl<T> OverflowInt<T> {
    // Note: sign or unsigned are not about the type T.
    // It's how we will range check the limbs. If the limbs are non-negative, use this one.
    pub fn from_canonical_unsigned_limbs(x: Vec<T>, limb_bits: usize) -> OverflowInt<T> {
        OverflowInt {
            limbs: x,
            max_overflow_bits: limb_bits,
            limb_max_abs: (1 << limb_bits) - 1,
        }
    }

    // Limbs can be negative. So the max_overflow_bits and limb_max_abs are different from the range
    // check result.
    pub fn from_canonical_signed_limbs(x: Vec<T>, limb_bits: usize) -> OverflowInt<T> {
        OverflowInt {
            limbs: x,
            max_overflow_bits: limb_bits + 1,
            limb_max_abs: (1 << limb_bits),
        }
    }

    // Used only when limbs are hand calculated.
    pub fn from_computed_limbs(
        x: Vec<T>,
        limb_max_abs: usize,
        max_overflow_bits: usize,
    ) -> OverflowInt<T> {
        OverflowInt {
            limbs: x,
            max_overflow_bits,
            limb_max_abs,
        }
    }

    pub fn max_overflow_bits(&self) -> usize {
        self.max_overflow_bits
    }

    pub fn limb_max_abs(&self) -> usize {
        self.limb_max_abs
    }

    pub fn num_limbs(&self) -> usize {
        self.limbs.len()
    }

    pub fn limb(&self, i: usize) -> &T {
        self.limbs.get(i).unwrap()
    }

    pub fn limbs(&self) -> &[T] {
        &self.limbs
    }
}

impl<T> OverflowInt<T>
where
    T: Clone + AddAssign + MulAssign,
{
    pub fn int_add(&self, s: isize, convert: fn(isize) -> T) -> OverflowInt<T> {
        let mut limbs = self.limbs.clone();
        limbs[0] += convert(s);
        let limb_max_abs = self.limb_max_abs + s.unsigned_abs();
        OverflowInt {
            limbs,
            limb_max_abs,
            max_overflow_bits: log2_ceil_usize(limb_max_abs),
        }
    }

    pub fn int_mul(&self, s: isize, convert: fn(isize) -> T) -> OverflowInt<T> {
        let mut limbs = self.limbs.clone();
        for limb in limbs.iter_mut() {
            *limb *= convert(s);
        }
        let limb_max_abs = self.limb_max_abs * s.unsigned_abs();
        OverflowInt {
            limbs,
            limb_max_abs,
            max_overflow_bits: log2_ceil_usize(limb_max_abs),
        }
    }
}

impl OverflowInt<isize> {
    pub fn from_biguint(
        x: &BigUint,
        limb_bits: usize,
        min_limbs: Option<usize>,
    ) -> OverflowInt<isize> {
        let limbs = match min_limbs {
            Some(min_limbs) => utils::big_uint_to_num_limbs(x, limb_bits, min_limbs),
            None => utils::big_uint_to_limbs(x, limb_bits),
        };
        let limbs: Vec<isize> = limbs.iter().map(|x| *x as isize).collect();
        OverflowInt::from_canonical_unsigned_limbs(limbs, limb_bits)
    }

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
