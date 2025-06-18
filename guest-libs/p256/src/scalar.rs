use alloc::vec::Vec;
use core::{cmp::Ordering, ops::ShrAssign};

use elliptic_curve::{
    bigint::{ArrayEncoding, Encoding, U256},
    ops::{Invert, Reduce},
    rand_core::RngCore,
    scalar::{FromUintUnchecked, IsHigh},
    subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption},
    zeroize::DefaultIsZeroes,
    Field, PrimeField, ScalarPrimitive,
};
use hex_literal::hex;
use openvm_algebra_guest::IntMod;

use crate::{internal::P256Scalar, point::FieldBytes, NistP256, ORDER_HEX};

impl P256Scalar {
    /// Returns the SEC1 encoding of this scalar.
    pub fn to_bytes(&self) -> FieldBytes {
        self.to_be_bytes().into()
    }
}
// --- Implement elliptic_curve traits on P256Scalar ---

impl Copy for P256Scalar {}

impl From<u64> for P256Scalar {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl Default for P256Scalar {
    fn default() -> Self {
        <Self as IntMod>::ZERO
    }
}

// Requires canonical form
impl ConstantTimeEq for P256Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.as_le_bytes().ct_eq(other.as_le_bytes())
    }
}

impl ConditionallySelectable for P256Scalar {
    fn conditional_select(a: &P256Scalar, b: &P256Scalar, choice: Choice) -> P256Scalar {
        P256Scalar::from_le_bytes_unchecked(
            &a.as_le_bytes()
                .iter()
                .zip(b.as_le_bytes().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

impl Field for P256Scalar {
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    fn random(mut _rng: impl RngCore) -> Self {
        unimplemented!()
    }

    #[must_use]
    fn square(&self) -> Self {
        self * self
    }

    #[must_use]
    fn double(&self) -> Self {
        self + self
    }

    fn invert(&self) -> CtOption<Self> {
        // needs to be in canonical form for ct_eq
        self.assert_reduced();
        let is_zero = self.ct_eq(&<Self as IntMod>::ZERO);
        CtOption::new(
            <P256Scalar as openvm_algebra_guest::Field>::invert(self),
            !is_zero,
        )
    }

    #[allow(clippy::many_single_char_names)]
    fn sqrt(&self) -> CtOption<Self> {
        match <Self as openvm_algebra_guest::Sqrt>::sqrt(self) {
            Some(sqrt) => CtOption::new(sqrt, 1.into()),
            None => CtOption::new(<Self as Field>::ZERO, 0.into()),
        }
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }
}

const fn seven_le() -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[0] = 7;
    buf
}

impl PrimeField for P256Scalar {
    type Repr = FieldBytes;

    const MODULUS: &'static str = ORDER_HEX;
    const NUM_BITS: u32 = 256;
    const CAPACITY: u32 = 255;
    const TWO_INV: Self = Self::from_const_bytes(hex!(
        "a992317e61e5dc7942cf8bd3567d73deffffffffffffff7f00000080ffffff7f"
    ));
    const MULTIPLICATIVE_GENERATOR: Self = Self::from_const_bytes(seven_le());
    const S: u32 = 4;
    const ROOT_OF_UNITY: Self = Self::from_const_bytes(hex!(
        "02661eb4fbd79205af8d3704d0ca4615fc3d2a84ce7a80ba9209772a067fc9ff"
    ));
    const ROOT_OF_UNITY_INV: Self = Self::from_const_bytes(hex!(
        "6437c757067f9c3737414c797c11ace3ae1c135804fa45c62a6fd462556aa6a0"
    ));
    const DELTA: Self = Self::from_const_bytes(hex!(
        "817d05a5391e0000000000000000000000000000000000000000000000000000"
    ));

    /// Attempts to parse the given byte array as an SEC1-encoded scalar.
    ///
    /// Returns None if the byte array does not contain a big-endian integer in the range
    /// [0, p).
    fn from_repr(bytes: FieldBytes) -> CtOption<Self> {
        let ret = Self::from_be_bytes_unchecked(bytes.as_slice());
        CtOption::new(ret, (ret.is_reduced() as u8).into())
    }

    // Endianness should match from_repr
    fn to_repr(&self) -> FieldBytes {
        *FieldBytes::from_slice(&self.to_be_bytes())
    }

    fn is_odd(&self) -> Choice {
        (self.as_le_bytes()[0] & 1).into()
    }
}

impl ShrAssign<usize> for P256Scalar {
    fn shr_assign(&mut self, _rhs: usize) {
        // I don't think this is used anywhere
        unimplemented!()
    }
}

impl Reduce<U256> for P256Scalar {
    type Bytes = FieldBytes;

    fn reduce(w: U256) -> Self {
        <Self as openvm_algebra_guest::Reduce>::reduce_le_bytes(&w.to_le_bytes())
    }

    #[inline]
    fn reduce_bytes(bytes: &FieldBytes) -> Self {
        Self::reduce(U256::from_be_byte_array(*bytes))
    }
}

impl PartialOrd for P256Scalar {
    // requires self and other to be in canonical form
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.assert_reduced();
        other.assert_reduced();
        Some(
            self.to_be_bytes()
                .iter()
                .zip(other.to_be_bytes().iter())
                .map(|(a, b)| a.cmp(b))
                .find(|ord| *ord != Ordering::Equal)
                .unwrap_or(Ordering::Equal),
        )
    }
}

impl IsHigh for P256Scalar {
    fn is_high(&self) -> Choice {
        // self > n/2
        // iff self + self overflows
        // iff self + self < self
        ((self + self < *self) as u8).into()
    }
}

impl Invert for P256Scalar {
    type Output = CtOption<Self>;

    fn invert(&self) -> CtOption<Self> {
        <Self as Field>::invert(self)
    }
}

impl FromUintUnchecked for P256Scalar {
    type Uint = U256;

    fn from_uint_unchecked(uint: Self::Uint) -> Self {
        Self::from_le_bytes_unchecked(&uint.to_le_bytes())
    }
}

impl From<ScalarPrimitive<NistP256>> for P256Scalar {
    fn from(scalar: ScalarPrimitive<NistP256>) -> Self {
        Self::from_le_bytes_unchecked(&scalar.as_uint().to_le_bytes())
    }
}

impl From<P256Scalar> for ScalarPrimitive<NistP256> {
    fn from(scalar: P256Scalar) -> ScalarPrimitive<NistP256> {
        ScalarPrimitive::from_slice(&scalar.to_be_bytes()).unwrap()
    }
}

impl DefaultIsZeroes for P256Scalar {}

impl AsRef<P256Scalar> for P256Scalar {
    fn as_ref(&self) -> &P256Scalar {
        self
    }
}

impl From<P256Scalar> for U256 {
    fn from(scalar: P256Scalar) -> Self {
        U256::from_be_slice(&scalar.to_be_bytes())
    }
}

impl From<P256Scalar> for FieldBytes {
    fn from(scalar: P256Scalar) -> Self {
        *FieldBytes::from_slice(&scalar.to_be_bytes())
    }
}
