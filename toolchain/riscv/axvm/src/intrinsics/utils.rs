#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::{BigInt, BigUint};

#[inline]
#[cfg(not(target_os = "zkvm"))]
#[allow(dead_code)]
/// Convert a `BigUint` to a `[u8; NUM_LIMBS]`.
pub fn biguint_to_limbs<const NUM_LIMBS: usize>(x: &BigUint) -> [u8; NUM_LIMBS] {
    let mut sm = x.to_bytes_le();
    sm.resize(NUM_LIMBS, 0);
    sm.try_into().unwrap()
}

#[inline]
#[cfg(not(target_os = "zkvm"))]
pub(super) fn bigint_to_limbs<const NUM_LIMBS: usize>(x: &BigInt) -> [u8; NUM_LIMBS] {
    let mut sm = x.to_bytes_le().1;
    sm.resize(NUM_LIMBS, 0);
    sm.try_into().unwrap()
}

/// A macro that implements all the following for the given struct and operation:
/// a op= b, a op= &b, a op b, a op &b, &a op b, &a op &b
#[macro_export]
macro_rules! impl_bin_op {
    ($struct_name:ty, $trait_name:ident,
        $trait_assign_name:ident, $trait_fn:ident,
        $trait_assign_fn:ident, $opcode:expr,
        $func3:expr, $func7:expr, $op_sym:tt,
        $rust_expr:expr) => {
        impl<'a> $trait_assign_name<&'a $struct_name> for $struct_name {
            #[inline(always)]
            fn $trait_assign_fn(&mut self, rhs: &'a $struct_name) {
                #[cfg(target_os = "zkvm")]
                custom_insn_r!(
                    $opcode,
                    $func3,
                    $func7,
                    self as *mut Self,
                    self as *const Self,
                    rhs as *const Self
                );
                #[cfg(not(target_os = "zkvm"))]
                {
                    *self = $rust_expr(self, rhs);
                }
            }
        }

        impl $trait_assign_name<$struct_name> for $struct_name {
            #[inline(always)]
            fn $trait_assign_fn(&mut self, rhs: $struct_name) {
                *self $op_sym &rhs;
            }
        }

        impl<'a> $trait_name<&'a $struct_name> for &$struct_name {
            type Output = $struct_name;
            #[inline(always)]
            fn $trait_fn(self, rhs: &'a $struct_name) -> Self::Output {
                #[cfg(target_os = "zkvm")]
                {
                    let mut uninit: MaybeUninit<$struct_name> = MaybeUninit::uninit();
                    custom_insn_r!(
                        $opcode,
                        $func3,
                        $func7,
                        uninit.as_mut_ptr(),
                        self as *const $struct_name,
                        rhs as *const $struct_name
                    );
                    unsafe { uninit.assume_init() }
                }
                #[cfg(not(target_os = "zkvm"))]
                return $rust_expr(self, rhs);
            }
        }

        impl<'a> $trait_name<&'a $struct_name> for $struct_name {
            type Output = $struct_name;
            #[inline(always)]
            fn $trait_fn(mut self, rhs: &'a $struct_name) -> Self::Output {
                self $op_sym rhs;
                self
            }
        }

        impl $trait_name<$struct_name> for $struct_name {
            type Output = $struct_name;
            #[inline(always)]
            fn $trait_fn(mut self, rhs: $struct_name) -> Self::Output {
                self $op_sym &rhs;
                self
            }
        }

        impl $trait_name<$struct_name> for &$struct_name {
            type Output = $struct_name;
            #[inline(always)]
            fn $trait_fn(self, mut rhs: $struct_name) -> Self::Output {
                rhs $op_sym self;
                rhs
            }
        }
    };
}
