#[cfg(target_os = "zkvm")]
use core::mem::MaybeUninit;
use core::{
    cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd},
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul,
        MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};

#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigUint;

#[derive(Copy, Clone, Debug)]
#[repr(align(32), C)]
struct U256 {
    limbs: [u8; 32],
}

impl U256 {
    #[cfg(not(target_os = "zkvm"))]
    fn as_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(&self.limbs)
    }

    #[cfg(not(target_os = "zkvm"))]
    fn from_biguint(value: &BigUint) -> Self {
        Self {
            limbs: value.to_bytes_le().try_into().unwrap(),
        }
    }

    #[cfg(target_os = "zkvm")]
    fn alloc() -> Self {
        let uninit = MaybeUninit::<Self>::uninit();
        let init = unsafe { uninit.assume_init() };
        init
    }
}

/// Addition

impl<'a> AddAssign<&'a U256> for U256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() + rhs.as_biguint()));
        }
    }
}

impl AddAssign<U256> for U256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: U256) {
        *self += &rhs;
    }
}

impl<'a> Add<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn add(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() + rhs.as_biguint()));
    }
}

impl<'a> Add<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            self += rhs;
            self
        }
    }
}

impl Add<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            self += &rhs;
            self
        }
    }
}

/// Subtraction

impl<'a> SubAssign<&'a U256> for U256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() - rhs.as_biguint()));
        }
    }
}

impl SubAssign<U256> for U256 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: U256) {
        *self -= &rhs;
    }
}

impl<'a> Sub<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn sub(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() - rhs.as_biguint()));
    }
}

impl<'a> Sub<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

/// Multiplication

impl<'a> MulAssign<&'a U256> for U256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() * rhs.as_biguint()));
        }
    }
}

impl MulAssign<U256> for U256 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: U256) {
        *self *= &rhs;
    }
}

impl<'a> Mul<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn mul(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() * rhs.as_biguint()));
    }
}

impl<'a> Mul<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= &rhs;
        self
    }
}

/// Bitwise XOR

impl<'a> BitXorAssign<&'a U256> for U256 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() ^ rhs.as_biguint()));
        }
    }
}

impl BitXorAssign<U256> for U256 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: U256) {
        *self ^= &rhs;
    }
}

impl<'a> BitXor<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn bitxor(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() ^ rhs.as_biguint()));
    }
}

impl<'a> BitXor<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(mut self, rhs: &Self) -> Self::Output {
        self ^= rhs;
        self
    }
}

impl BitXor<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(mut self, rhs: Self) -> Self::Output {
        self ^= &rhs;
        self
    }
}

/// Bitwise AND

impl<'a> BitAndAssign<&'a U256> for U256 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() & rhs.as_biguint()));
        }
    }
}

impl BitAndAssign<U256> for U256 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: U256) {
        *self &= &rhs;
    }
}

impl<'a> BitAnd<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn bitand(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() & rhs.as_biguint()));
    }
}

impl<'a> BitAnd<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitand(mut self, rhs: &Self) -> Self::Output {
        self &= rhs;
        self
    }
}

impl BitAnd<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitand(mut self, rhs: Self) -> Self::Output {
        self &= &rhs;
        self
    }
}

/// Bitwise OR

impl<'a> BitOrAssign<&'a U256> for U256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() | rhs.as_biguint()));
        }
    }
}

impl BitOrAssign<U256> for U256 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: U256) {
        *self |= &rhs;
    }
}

impl<'a> BitOr<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn bitor(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() | rhs.as_biguint()));
    }
}

impl<'a> BitOr<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: &Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl BitOr<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= &rhs;
        self
    }
}

/// Left shift

impl<'a> ShlAssign<&'a U256> for U256 {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() << rhs.limbs[0] as usize));
        }
    }
}

impl ShlAssign<U256> for U256 {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self <<= &rhs;
        }
    }
}

impl<'a> Shl<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn shl(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() << rhs.limbs[0] as usize));
    }
}

impl<'a> Shl<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shl(mut self, rhs: &Self) -> Self::Output {
        self <<= rhs;
        self
    }
}

impl Shl<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shl(mut self, rhs: Self) -> Self::Output {
        self <<= &rhs;
        self
    }
}

/// Right shift

impl<'a> ShrAssign<&'a U256> for U256 {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: &'a U256) {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        {
            *self = Self::from_biguint(&(self.as_biguint() >> rhs.limbs[0] as usize));
        }
    }
}

impl ShrAssign<U256> for U256 {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: U256) {
        *self >>= &rhs;
    }
}

impl<'a> Shr<&'a U256> for &U256 {
    type Output = U256;
    #[inline(always)]
    fn shr(self, rhs: &'a U256) -> U256 {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return U256::from_biguint(&(self.as_biguint() >> rhs.limbs[0] as usize));
    }
}

impl<'a> Shr<&'a U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shr(mut self, rhs: &Self) -> Self::Output {
        self >>= rhs;
        self
    }
}

impl Shr<U256> for U256 {
    type Output = Self;
    #[inline(always)]
    fn shr(mut self, rhs: Self) -> Self::Output {
        self >>= &rhs;
        self
    }
}

impl PartialEq for U256 {
    fn eq(&self, other: &Self) -> bool {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return self.as_biguint() == other.as_biguint();
    }
}

impl Eq for U256 {}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return self.as_biguint().partial_cmp(&other.as_biguint());
    }
}

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> Ordering {
        #[cfg(target_os = "zkvm")]
        todo!();
        #[cfg(not(target_os = "zkvm"))]
        return self.as_biguint().cmp(&other.as_biguint());
    }
}
