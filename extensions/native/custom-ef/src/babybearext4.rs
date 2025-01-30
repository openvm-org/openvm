use core::{
    array,
    fmt::{self, Display, Formatter},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use itertools::Itertools;
use num_bigint::BigUint;
use openvm_stark_backend::{
    p3_field::{
        extension::{BinomiallyExtendable, HasFrobenius, HasTwoAdicBinomialExtension},
        field_to_array, ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra, Packable,
        TwoAdicField,
    },
    p3_util::convert_vec,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{distributions::Standard, prelude::Distribution};
use serde::{Deserialize, Serialize};

use crate::packed::BinomialExtensionField;

// use crate::packed::BabyBearExt4 as PackedBabyBearExt4;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // to make the zero_vec implementation safe
pub struct BabyBearExt4 {
    #[serde(
        with = "openvm_stark_backend::p3_util::array_serialization",
        bound(
            serialize = "[BabyBear; 4]: Serialize",
            deserialize = "[BabyBear; 4]: Deserialize<'de>"
        )
    )]
    pub(crate) value: [BabyBear; 4],
}

impl Default for BabyBearExt4 {
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| BabyBear::ZERO),
        }
    }
}

impl From<BabyBear> for BabyBearExt4 {
    fn from(x: BabyBear) -> Self {
        Self {
            value: field_to_array::<BabyBear, 4>(x),
        }
    }
}

impl Packable for BabyBearExt4 {}

impl ExtensionField<BabyBear> for BabyBearExt4 {
    type ExtensionPacking = BinomialExtensionField<<BabyBear as Field>::Packing, 4>;
}

impl HasFrobenius<BabyBear> for BabyBearExt4 {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(n^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= 4 {
            // x |-> x^(n^D) is the identity, so x^(n^count) ==
            // x^(n^(count % D))
            return self.repeated_frobenius(count % 4);
        }
        let arr: &[BabyBear] = self.as_base_slice();

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((n-1)/D)
        let mut z0 = <BabyBear as BinomiallyExtendable<4>>::DTH_ROOT;
        for _ in 1..count {
            z0 *= <BabyBear as BinomiallyExtendable<4>>::DTH_ROOT;
        }

        let mut res = [BabyBear::ZERO; 4];
        for (i, z) in z0.powers().take(4).enumerate() {
            res[i] = arr[i] * z;
        }

        Self::from_base_slice(&res)
    }

    /// Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn frobenius_inv(&self) -> Self {
        // Writing 'a' for self, we need to compute a^(r-1):
        // r = n^D-1/n-1 = n^(D-1)+n^(D-2)+...+n
        let mut f = Self::ONE;
        for _ in 1..4 {
            f = (f * *self).frobenius();
        }

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product.
        let a = self.value;
        let b = f.value;
        let mut g = BabyBear::ZERO;
        for i in 1..4 {
            g += a[i] * b[4 - i];
        }
        g *= <BabyBear as BinomiallyExtendable<4>>::W;
        g += a[0] * b[0];
        debug_assert_eq!(Self::from(g), *self * f);

        f * g.inverse()
    }
}

impl FieldAlgebra for BabyBearExt4 {
    type F = BabyBearExt4;

    const ZERO: Self = Self {
        value: [BabyBear::ZERO; 4],
    };

    const ONE: Self = Self {
        value: field_to_array::<BabyBear, 4>(BabyBear::ONE),
    };

    const TWO: Self = Self {
        value: field_to_array::<BabyBear, 4>(BabyBear::TWO),
    };

    const NEG_ONE: Self = Self {
        value: field_to_array::<BabyBear, 4>(BabyBear::NEG_ONE),
    };

    #[inline]
    fn from_f(f: Self::F) -> Self {
        Self {
            value: f.value.map(BabyBear::from_f),
        }
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        BabyBear::from_bool(b).into()
    }

    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        BabyBear::from_canonical_u8(n).into()
    }

    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        BabyBear::from_canonical_u16(n).into()
    }

    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        BabyBear::from_canonical_u32(n).into()
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        BabyBear::from_canonical_u64(n).into()
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        BabyBear::from_canonical_usize(n).into()
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        BabyBear::from_wrapped_u32(n).into()
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        BabyBear::from_wrapped_u64(n).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        <Self as Mul<Self>>::mul(*self, *self)
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { convert_vec(BabyBear::zero_vec(len * 4)) }
    }
}

impl Field for BabyBearExt4 {
    type Packing = Self;

    const GENERATOR: Self = Self {
        value: BabyBear::EXT_GENERATOR,
    };

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        Some(self.frobenius_inv())
    }

    fn halve(&self) -> Self {
        Self {
            value: self.value.map(|x| x.halve()),
        }
    }

    fn order() -> BigUint {
        BabyBear::order().pow(4)
    }
}

impl Display for BabyBearExt4 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            let str = self
                .value
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, x)| match (i, x.is_one()) {
                    (0, _) => format!("{x}"),
                    (1, true) => "X".to_string(),
                    (1, false) => format!("{x} X"),
                    (_, true) => format!("X^{i}"),
                    (_, false) => format!("{x} X^{i}"),
                })
                .join(" + ");
            write!(f, "{}", str)
        }
    }
}

impl Neg for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(BabyBear::neg),
        }
    }
}

impl Add for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self { value: res }
    }
}

impl Add<BabyBear> for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: BabyBear) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl AddAssign for BabyBearExt4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl AddAssign<BabyBear> for BabyBearExt4 {
    #[inline]
    fn add_assign(&mut self, rhs: BabyBear) {
        self.value[0] += rhs;
    }
}

impl Sum for BabyBearExt4 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Sub for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r -= rhs_val;
        }
        Self { value: res }
    }
}

impl Sub<BabyBear> for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: BabyBear) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl SubAssign for BabyBearExt4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<BabyBear> for BabyBearExt4 {
    #[inline]
    fn sub_assign(&mut self, rhs: BabyBear) {
        *self = *self - rhs;
    }
}

impl Mul for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        let mut buf = Vec::with_capacity(7);
        let w = <<BabyBear as FieldAlgebra>::F as BinomiallyExtendable<4>>::W;
        let w_af = BabyBear::from_f(w);

        {
            #[allow(clippy::needless_range_loop)]
            for i in 0..4 {
                for j in 0..4 {
                    if i + j >= buf.len() {
                        buf.push(a[i] * b[j]);
                    } else {
                        buf[i + j] += a[i] * b[j];
                    }
                }
            }
            for i in 0..3 {
                res.value[i] = buf[i] + w_af * buf[i + 4];
            }
            res.value[3] = buf[3];
        }
        res
    }
}

impl Mul<BabyBear> for BabyBearExt4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        Self {
            value: self.value.map(|x| x * rhs),
        }
    }
}

impl Product for BabyBearExt4 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Div for BabyBearExt4 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for BabyBearExt4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl MulAssign for BabyBearExt4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<BabyBear> for BabyBearExt4 {
    #[inline]
    fn mul_assign(&mut self, rhs: BabyBear) {
        *self = *self * rhs;
    }
}

impl FieldExtensionAlgebra<BabyBear> for BabyBearExt4 {
    const D: usize = 4;

    #[inline]
    fn from_base(b: BabyBear) -> Self {
        Self {
            value: field_to_array(b),
        }
    }

    #[inline]
    fn from_base_slice(bs: &[BabyBear]) -> Self {
        Self::from_base_fn(|i| bs[i])
    }

    #[inline]
    fn from_base_fn<F: FnMut(usize) -> BabyBear>(f: F) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    #[inline]
    fn from_base_iter<I: Iterator<Item = BabyBear>>(iter: I) -> Self {
        let mut res = Self::default();
        for (i, b) in iter.enumerate() {
            res.value[i] = b;
        }
        res
    }

    #[inline(always)]
    fn as_base_slice(&self) -> &[BabyBear] {
        &self.value
    }
}

impl Distribution<BabyBearExt4> for Standard
where
    Standard: Distribution<BabyBear>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BabyBearExt4 {
        let mut res = [BabyBear::ZERO; 4];
        for r in res.iter_mut() {
            *r = Standard.sample(rng);
        }
        BabyBearExt4::from_base_slice(&res)
    }
}

impl TwoAdicField for BabyBearExt4 {
    const TWO_ADICITY: usize = <BabyBear as HasTwoAdicBinomialExtension<4>>::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self {
            value: BabyBear::ext_two_adic_generator(bits),
        }
    }
}
