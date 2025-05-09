use crate::Uint;
use core::cmp::Ordering;

impl<const BITS: usize, const LIMBS: usize> PartialOrd for Uint<BITS, LIMBS> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const BITS: usize, const LIMBS: usize> Ord for Uint<BITS, LIMBS> {
    #[cfg(not(target_os = "zkvm"))]
    #[inline]
    fn cmp(&self, rhs: &Self) -> Ordering {
        crate::algorithms::cmp(self.as_limbs(), rhs.as_limbs())
    }

    #[cfg(target_os = "zkvm")]
    #[inline]
    fn cmp(&self, rhs: &Self) -> Ordering {
        use crate::support::zkvm::zkvm_u256_cmp_impl;
        if BITS == 256 {
            return unsafe {
                zkvm_u256_cmp_impl(
                    self.limbs.as_ptr() as *const u8,
                    rhs.limbs.as_ptr() as *const u8,
                )
            };
        }
        self.cmp(rhs)
    }
}

impl<const BITS: usize, const LIMBS: usize> Uint<BITS, LIMBS> {
    /// Returns true if the value is zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use crate::Uint;

    #[test]
    fn test_is_zero() {
        assert!(Uint::<0, 0>::ZERO.is_zero());
        assert!(Uint::<1, 1>::ZERO.is_zero());
        assert!(Uint::<7, 1>::ZERO.is_zero());
        assert!(Uint::<64, 1>::ZERO.is_zero());

        assert!(!Uint::<1, 1>::from_limbs([1]).is_zero());
        assert!(!Uint::<7, 1>::from_limbs([1]).is_zero());
        assert!(!Uint::<64, 1>::from_limbs([1]).is_zero());
    }
}
