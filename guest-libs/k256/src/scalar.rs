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

use crate::{
    internal::{seven_le, Secp256k1Scalar},
    point::FieldBytes,
    Secp256k1, ORDER_HEX,
};

impl Secp256k1Scalar {
    /// Returns the SEC1 encoding of this scalar.
    pub fn to_bytes(&self) -> FieldBytes {
        self.to_be_bytes().into()
    }
}
// --- Implement elliptic_curve traits on Secp256k1Scalar ---

impl Copy for Secp256k1Scalar {}

impl From<u64> for Secp256k1Scalar {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl Default for Secp256k1Scalar {
    fn default() -> Self {
        <Self as IntMod>::ZERO
    }
}

// Requires canonical form
impl ConstantTimeEq for Secp256k1Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.as_le_bytes().ct_eq(other.as_le_bytes())
    }
}

impl ConditionallySelectable for Secp256k1Scalar {
    fn conditional_select(
        a: &Secp256k1Scalar,
        b: &Secp256k1Scalar,
        choice: Choice,
    ) -> Secp256k1Scalar {
        Secp256k1Scalar::from_le_bytes_unchecked(
            &a.as_le_bytes()
                .iter()
                .zip(b.as_le_bytes().iter())
                .map(|(a, b)| u8::conditional_select(a, b, choice))
                .collect::<Vec<_>>(),
        )
    }
}

impl Field for Secp256k1Scalar {
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
            <Secp256k1Scalar as openvm_algebra_guest::Field>::invert(self),
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

impl PrimeField for Secp256k1Scalar {
    type Repr = FieldBytes;

    const MODULUS: &'static str = ORDER_HEX;
    const NUM_BITS: u32 = 256;
    const CAPACITY: u32 = 255;
    const TWO_INV: Self = Self::from_const_bytes(hex!(
        "a1201b68462fe9df1d50a457736e575dffffffffffffffffffffffffffffff7f"
    ));
    const MULTIPLICATIVE_GENERATOR: Self = Self::from_const_bytes(seven_le());
    const S: u32 = 6;
    const ROOT_OF_UNITY: Self = Self::from_const_bytes(hex!(
        "f252b002544b2f9945607580b6eabd98a883c4fba37998df8619a9e760c01d0c"
    ));
    const ROOT_OF_UNITY_INV: Self = Self::from_const_bytes(hex!(
        "1c0d4f88a030fbb6c313a40a9175a27772bb8c5bc7b0c7ef96702df181e13afd"
    ));
    const DELTA: Self = Self::from_const_bytes(hex!(
        "0176bbc0c81794191e34e180e7783bd6c86145fe21bc0c000000000000000000"
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

impl ShrAssign<usize> for Secp256k1Scalar {
    fn shr_assign(&mut self, _rhs: usize) {
        // I don't think this is used anywhere
        unimplemented!()
    }
}

impl Reduce<U256> for Secp256k1Scalar {
    type Bytes = FieldBytes;

    fn reduce(w: U256) -> Self {
        <Self as openvm_algebra_guest::Reduce>::reduce_le_bytes(&w.to_le_bytes())
    }

    #[inline]
    fn reduce_bytes(bytes: &FieldBytes) -> Self {
        Self::reduce(U256::from_be_byte_array(*bytes))
    }
}

impl PartialOrd for Secp256k1Scalar {
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

impl IsHigh for Secp256k1Scalar {
    fn is_high(&self) -> Choice {
        // self > n/2
        // iff self + self overflows
        // iff self + self < self
        ((self + self < *self) as u8).into()
    }
}

impl Invert for Secp256k1Scalar {
    type Output = CtOption<Self>;

    fn invert(&self) -> CtOption<Self> {
        <Self as Field>::invert(self)
    }
}

impl FromUintUnchecked for Secp256k1Scalar {
    type Uint = U256;

    fn from_uint_unchecked(uint: Self::Uint) -> Self {
        Self::from_le_bytes_unchecked(&uint.to_le_bytes())
    }
}

impl From<ScalarPrimitive<Secp256k1>> for Secp256k1Scalar {
    fn from(scalar: ScalarPrimitive<Secp256k1>) -> Self {
        Self::from_le_bytes_unchecked(&scalar.as_uint().to_le_bytes())
    }
}

impl From<Secp256k1Scalar> for ScalarPrimitive<Secp256k1> {
    fn from(scalar: Secp256k1Scalar) -> ScalarPrimitive<Secp256k1> {
        ScalarPrimitive::from_slice(&scalar.to_be_bytes()).unwrap()
    }
}

impl DefaultIsZeroes for Secp256k1Scalar {}

impl AsRef<Secp256k1Scalar> for Secp256k1Scalar {
    fn as_ref(&self) -> &Secp256k1Scalar {
        self
    }
}

impl From<Secp256k1Scalar> for U256 {
    fn from(scalar: Secp256k1Scalar) -> Self {
        U256::from_be_slice(&scalar.to_be_bytes())
    }
}

impl From<Secp256k1Scalar> for FieldBytes {
    fn from(scalar: Secp256k1Scalar) -> Self {
        *FieldBytes::from_slice(&scalar.to_be_bytes())
    }
}
