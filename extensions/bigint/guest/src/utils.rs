#[cfg(not(target_os = "zkvm"))]
use num_bigint_dig::BigInt;

#[inline]
#[cfg(not(target_os = "zkvm"))]
#[allow(dead_code)]
/// Convert a `BigInt` to a `[u8; NUM_LIMBS]` in two's complement little-endian format.
pub(super) fn bigint_to_limbs<const NUM_LIMBS: usize>(x: &BigInt) -> [u8; NUM_LIMBS] {
    let mut sm = x.to_signed_bytes_le();
    let mut ext = 0;
    if let Some(last) = sm.last() {
        if (*last as i8) < 0 {
            ext = u8::MAX;
        }
    }
    sm.resize(NUM_LIMBS, ext);
    sm.try_into().unwrap()
}

/// A macro that implements all the following for the given struct and operation:
/// a op= b, a op= &b, a op b, a op &b, &a op b, &a op &b
/// Description of the parameters (see [u256.rs] for an example):
/// - $struct_name: The struct to implement the operation for.
/// - $trait_name: The trait name of the operation to implement.
/// - $trait_assign_name: The trait name of the assignment operation to implement.
/// - $trait_fn: The trait function name to implement.
/// - $trait_assign_fn: The assignment trait function name to implement.
/// - $opcode: The custom opcode of the operation in openvm.
/// - $func3: The func3 of the operation in openvm.
/// - $func7: The func7 of the operation in openvm.
/// - $op_sym: The symbol to use for the operation.
/// - $rust_expr: A closure to get the result of the operation if target is non-zkvm.
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
