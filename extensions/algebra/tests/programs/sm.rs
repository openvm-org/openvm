block_size: Lit::Int { token: 32 }
#![feature(prelude_import)]
#![no_main]
#![no_std]
#[prelude_import]
use core::prelude::rust_2021::*;
#[macro_use]
extern crate core;
extern crate compiler_builtins as _;
use openvm_algebra_guest::{DivUnsafe, IntMod};
/// An element of the ring of integers modulo a positive integer.
/// The element is internally represented as a fixed size array of bytes.
///
/// ## Caution
/// It is not guaranteed that the integer representation is less than the modulus.
/// After any arithmetic operation, the honest host should normalize the result
/// to its canonical representation less than the modulus, but guest execution does not
/// require it.
///
/// See [`assert_reduced`](openvm_algebra_guest::IntMod::assert_reduced) and
/// [`is_reduced`](openvm_algebra_guest::IntMod::is_reduced).
#[repr(C, align(32))]
pub struct Secp256k1Coord(
    #[serde(with = "openvm_algebra_guest::BigArray")]
    [u8; 32usize],
);
#[automatically_derived]
impl ::core::clone::Clone for Secp256k1Coord {
    #[inline]
    fn clone(&self) -> Secp256k1Coord {
        Secp256k1Coord(::core::clone::Clone::clone(&self.0))
    }
}
#[automatically_derived]
impl ::core::cmp::Eq for Secp256k1Coord {
    #[inline]
    #[doc(hidden)]
    #[coverage(off)]
    fn assert_receiver_is_total_eq(&self) -> () {
        let _: ::core::cmp::AssertParamIsEq<[u8; 32usize]>;
    }
}
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl _serde::Serialize for Secp256k1Coord {
        fn serialize<__S>(
            &self,
            __serializer: __S,
        ) -> _serde::__private::Result<__S::Ok, __S::Error>
        where
            __S: _serde::Serializer,
        {
            _serde::Serializer::serialize_newtype_struct(
                __serializer,
                "Secp256k1Coord",
                {
                    #[doc(hidden)]
                    struct __SerializeWith<'__a> {
                        values: (&'__a [u8; 32usize],),
                        phantom: _serde::__private::PhantomData<Secp256k1Coord>,
                    }
                    #[automatically_derived]
                    impl<'__a> _serde::Serialize for __SerializeWith<'__a> {
                        fn serialize<__S>(
                            &self,
                            __s: __S,
                        ) -> _serde::__private::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            openvm_algebra_guest::BigArray::serialize(self.values.0, __s)
                        }
                    }
                    &__SerializeWith {
                        values: (&self.0,),
                        phantom: _serde::__private::PhantomData::<Secp256k1Coord>,
                    }
                },
            )
        }
    }
};
#[doc(hidden)]
#[allow(
    non_upper_case_globals,
    unused_attributes,
    unused_qualifications,
    clippy::absolute_paths,
)]
const _: () = {
    #[allow(unused_extern_crates, clippy::useless_attribute)]
    extern crate serde as _serde;
    #[automatically_derived]
    impl<'de> _serde::Deserialize<'de> for Secp256k1Coord {
        fn deserialize<__D>(
            __deserializer: __D,
        ) -> _serde::__private::Result<Self, __D::Error>
        where
            __D: _serde::Deserializer<'de>,
        {
            #[doc(hidden)]
            struct __Visitor<'de> {
                marker: _serde::__private::PhantomData<Secp256k1Coord>,
                lifetime: _serde::__private::PhantomData<&'de ()>,
            }
            #[automatically_derived]
            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                type Value = Secp256k1Coord;
                fn expecting(
                    &self,
                    __formatter: &mut _serde::__private::Formatter,
                ) -> _serde::__private::fmt::Result {
                    _serde::__private::Formatter::write_str(
                        __formatter,
                        "tuple struct Secp256k1Coord",
                    )
                }
                #[inline]
                fn visit_newtype_struct<__E>(
                    self,
                    __e: __E,
                ) -> _serde::__private::Result<Self::Value, __E::Error>
                where
                    __E: _serde::Deserializer<'de>,
                {
                    let __field0: [u8; 32usize] = openvm_algebra_guest::BigArray::deserialize(
                        __e,
                    )?;
                    _serde::__private::Ok(Secp256k1Coord(__field0))
                }
                #[inline]
                fn visit_seq<__A>(
                    self,
                    mut __seq: __A,
                ) -> _serde::__private::Result<Self::Value, __A::Error>
                where
                    __A: _serde::de::SeqAccess<'de>,
                {
                    let __field0 = match {
                        #[doc(hidden)]
                        struct __DeserializeWith<'de> {
                            value: [u8; 32usize],
                            phantom: _serde::__private::PhantomData<Secp256k1Coord>,
                            lifetime: _serde::__private::PhantomData<&'de ()>,
                        }
                        #[automatically_derived]
                        impl<'de> _serde::Deserialize<'de> for __DeserializeWith<'de> {
                            fn deserialize<__D>(
                                __deserializer: __D,
                            ) -> _serde::__private::Result<Self, __D::Error>
                            where
                                __D: _serde::Deserializer<'de>,
                            {
                                _serde::__private::Ok(__DeserializeWith {
                                    value: openvm_algebra_guest::BigArray::deserialize(
                                        __deserializer,
                                    )?,
                                    phantom: _serde::__private::PhantomData,
                                    lifetime: _serde::__private::PhantomData,
                                })
                            }
                        }
                        _serde::__private::Option::map(
                            _serde::de::SeqAccess::next_element::<
                                __DeserializeWith<'de>,
                            >(&mut __seq)?,
                            |__wrap| __wrap.value,
                        )
                    } {
                        _serde::__private::Some(__value) => __value,
                        _serde::__private::None => {
                            return _serde::__private::Err(
                                _serde::de::Error::invalid_length(
                                    0usize,
                                    &"tuple struct Secp256k1Coord with 1 element",
                                ),
                            );
                        }
                    };
                    _serde::__private::Ok(Secp256k1Coord(__field0))
                }
            }
            _serde::Deserializer::deserialize_newtype_struct(
                __deserializer,
                "Secp256k1Coord",
                __Visitor {
                    marker: _serde::__private::PhantomData::<Secp256k1Coord>,
                    lifetime: _serde::__private::PhantomData,
                },
            )
        }
    }
};
extern "C" {
    fn add_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    );
    fn sub_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    );
    fn mul_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    );
    fn div_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    );
    fn is_eq_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rs1: usize,
        rs2: usize,
    ) -> bool;
}
impl Secp256k1Coord {
    #[inline(always)]
    const fn from_const_u8(val: u8) -> Self {
        let mut bytes = [0; 32usize];
        bytes[0] = val;
        Self(bytes)
    }
    /// Constructor from little-endian bytes. Does not enforce the integer value of `bytes`
    /// must be less than the modulus.
    pub const fn from_const_bytes(bytes: [u8; 32usize]) -> Self {
        Self(bytes)
    }
    #[inline(always)]
    fn add_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                add_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    self as *mut Self as usize,
                    self as *const Self as usize,
                    other as *const Self as usize,
                );
            }
        }
    }
    #[inline(always)]
    fn sub_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                sub_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    self as *mut Self as usize,
                    self as *const Self as usize,
                    other as *const Self as usize,
                );
            }
        }
    }
    #[inline(always)]
    fn mul_assign_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                mul_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    self as *mut Self as usize,
                    self as *const Self as usize,
                    other as *const Self as usize,
                );
            }
        }
    }
    #[inline(always)]
    fn div_assign_unsafe_impl(&mut self, other: &Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                div_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    self as *mut Self as usize,
                    self as *const Self as usize,
                    other as *const Self as usize,
                );
            }
        }
    }
    /// SAFETY: `dst_ptr` must be a raw pointer to `&mut Self`.
    /// It will be written to only at the very end .
    #[inline(always)]
    unsafe fn add_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                add_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    dst_ptr as usize,
                    self as *const Secp256k1Coord as usize,
                    other as *const Secp256k1Coord as usize,
                );
            }
        }
    }
    /// SAFETY: `dst_ptr` must be a raw pointer to `&mut Self`.
    /// It will be written to only at the very end .
    #[inline(always)]
    unsafe fn sub_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                sub_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    dst_ptr as usize,
                    self as *const Secp256k1Coord as usize,
                    other as *const Secp256k1Coord as usize,
                );
            }
        }
    }
    /// SAFETY: `dst_ptr` must be a raw pointer to `&mut Self`.
    /// It will be written to only at the very end .
    #[inline(always)]
    unsafe fn mul_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                mul_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    dst_ptr as usize,
                    self as *const Secp256k1Coord as usize,
                    other as *const Secp256k1Coord as usize,
                );
            }
        }
    }
    #[inline(always)]
    fn div_unsafe_refs_impl(&self, other: &Self) -> Self {
        #[cfg(not(target_os = "zkvm"))]
        {
            let mut uninit: core::mem::MaybeUninit<Secp256k1Coord> = core::mem::MaybeUninit::uninit();
            unsafe {
                div_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    uninit.as_mut_ptr() as usize,
                    self as *const Secp256k1Coord as usize,
                    other as *const Secp256k1Coord as usize,
                );
            }
            unsafe { uninit.assume_init() }
        }
    }
    #[inline(always)]
    fn eq_impl(&self, other: &Self) -> bool {
        #[cfg(not(target_os = "zkvm"))]
        {
            unsafe {
                is_eq_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
                    self as *const Secp256k1Coord as usize,
                    other as *const Secp256k1Coord as usize,
                )
            }
        }
    }
}
mod algebra_impl_0 {
    use openvm_algebra_guest::IntMod;
    use super::Secp256k1Coord;
    impl IntMod for Secp256k1Coord {
        type Repr = [u8; 32usize];
        type SelfRef<'a> = &'a Self;
        const MODULUS: Self::Repr = [
            47u8, 252u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
            255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
            255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
        ];
        const ZERO: Self = Self([0; 32usize]);
        const NUM_LIMBS: usize = 32usize;
        const ONE: Self = Self::from_const_u8(1);
        fn from_repr(repr: Self::Repr) -> Self {
            Self(repr)
        }
        fn from_le_bytes(bytes: &[u8]) -> Self {
            let mut arr = [0u8; 32usize];
            arr.copy_from_slice(bytes);
            Self(arr)
        }
        fn from_be_bytes(bytes: &[u8]) -> Self {
            let mut arr = [0u8; 32usize];
            for (a, b) in arr.iter_mut().zip(bytes.iter().rev()) {
                *a = *b;
            }
            Self(arr)
        }
        fn from_u8(val: u8) -> Self {
            Self::from_const_u8(val)
        }
        fn from_u32(val: u32) -> Self {
            let mut bytes = [0; 32usize];
            bytes[..4].copy_from_slice(&val.to_le_bytes());
            Self(bytes)
        }
        fn from_u64(val: u64) -> Self {
            let mut bytes = [0; 32usize];
            bytes[..8].copy_from_slice(&val.to_le_bytes());
            Self(bytes)
        }
        fn as_le_bytes(&self) -> &[u8] {
            &(self.0)
        }
        fn to_be_bytes(&self) -> [u8; 32usize] {
            core::array::from_fn(|i| self.0[32usize - 1 - i])
        }
        fn neg_assign(&mut self) {
            unsafe {
                (Secp256k1Coord::ZERO)
                    .sub_refs_impl(self, self as *const Self as *mut Self);
            }
        }
        fn double_assign(&mut self) {
            unsafe {
                self.add_refs_impl(self, self as *const Self as *mut Self);
            }
        }
        fn square_assign(&mut self) {
            unsafe {
                self.mul_refs_impl(self, self as *const Self as *mut Self);
            }
        }
        fn double(&self) -> Self {
            self + self
        }
        fn square(&self) -> Self {
            self * self
        }
        fn cube(&self) -> Self {
            &self.square() * self
        }
        /// If `self` is not in its canonical form, the proof will fail to verify.
        /// This means guest execution will never terminate (either successfully or
        /// unsuccessfully) if `self` is not in its canonical form.
        fn assert_reduced(&self) {
            let _ = core::hint::black_box(PartialEq::eq(self, self));
        }
        fn is_reduced(&self) -> bool {
            for (x_limb, p_limb) in self.0.iter().rev().zip(Self::MODULUS.iter().rev()) {
                if x_limb < p_limb {
                    return true;
                } else if x_limb > p_limb {
                    return false;
                }
            }
            false
        }
    }
    impl<'a> core::ops::AddAssign<&'a Secp256k1Coord> for Secp256k1Coord {
        #[inline(always)]
        fn add_assign(&mut self, other: &'a Secp256k1Coord) {
            self.add_assign_impl(other);
        }
    }
    impl core::ops::AddAssign for Secp256k1Coord {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            self.add_assign_impl(&other);
        }
    }
    impl core::ops::Add for Secp256k1Coord {
        type Output = Self;
        #[inline(always)]
        fn add(mut self, other: Self) -> Self::Output {
            self += other;
            self
        }
    }
    impl<'a> core::ops::Add<&'a Secp256k1Coord> for Secp256k1Coord {
        type Output = Self;
        #[inline(always)]
        fn add(mut self, other: &'a Secp256k1Coord) -> Self::Output {
            self += other;
            self
        }
    }
    impl<'a> core::ops::Add<&'a Secp256k1Coord> for &Secp256k1Coord {
        type Output = Secp256k1Coord;
        #[inline(always)]
        fn add(self, other: &'a Secp256k1Coord) -> Self::Output {
            let mut uninit: core::mem::MaybeUninit<Secp256k1Coord> = core::mem::MaybeUninit::uninit();
            unsafe {
                self.add_refs_impl(other, uninit.as_mut_ptr());
                uninit.assume_init()
            }
        }
    }
    impl<'a> core::ops::SubAssign<&'a Secp256k1Coord> for Secp256k1Coord {
        #[inline(always)]
        fn sub_assign(&mut self, other: &'a Secp256k1Coord) {
            self.sub_assign_impl(other);
        }
    }
    impl core::ops::SubAssign for Secp256k1Coord {
        #[inline(always)]
        fn sub_assign(&mut self, other: Self) {
            self.sub_assign_impl(&other);
        }
    }
    impl core::ops::Sub for Secp256k1Coord {
        type Output = Self;
        #[inline(always)]
        fn sub(mut self, other: Self) -> Self::Output {
            self -= other;
            self
        }
    }
    impl<'a> core::ops::Sub<&'a Secp256k1Coord> for Secp256k1Coord {
        type Output = Self;
        #[inline(always)]
        fn sub(mut self, other: &'a Secp256k1Coord) -> Self::Output {
            self -= other;
            self
        }
    }
    impl<'a> core::ops::Sub<&'a Secp256k1Coord> for &'a Secp256k1Coord {
        type Output = Secp256k1Coord;
        #[inline(always)]
        fn sub(self, other: &'a Secp256k1Coord) -> Self::Output {
            let mut uninit: core::mem::MaybeUninit<Secp256k1Coord> = core::mem::MaybeUninit::uninit();
            unsafe {
                self.sub_refs_impl(other, uninit.as_mut_ptr());
                uninit.assume_init()
            }
        }
    }
    impl<'a> core::ops::MulAssign<&'a Secp256k1Coord> for Secp256k1Coord {
        #[inline(always)]
        fn mul_assign(&mut self, other: &'a Secp256k1Coord) {
            self.mul_assign_impl(other);
        }
    }
    impl core::ops::MulAssign for Secp256k1Coord {
        #[inline(always)]
        fn mul_assign(&mut self, other: Self) {
            self.mul_assign_impl(&other);
        }
    }
    impl core::ops::Mul for Secp256k1Coord {
        type Output = Self;
        #[inline(always)]
        fn mul(mut self, other: Self) -> Self::Output {
            self *= other;
            self
        }
    }
    impl<'a> core::ops::Mul<&'a Secp256k1Coord> for Secp256k1Coord {
        type Output = Self;
        #[inline(always)]
        fn mul(mut self, other: &'a Secp256k1Coord) -> Self::Output {
            self *= other;
            self
        }
    }
    impl<'a> core::ops::Mul<&'a Secp256k1Coord> for &Secp256k1Coord {
        type Output = Secp256k1Coord;
        #[inline(always)]
        fn mul(self, other: &'a Secp256k1Coord) -> Self::Output {
            let mut uninit: core::mem::MaybeUninit<Secp256k1Coord> = core::mem::MaybeUninit::uninit();
            unsafe {
                self.mul_refs_impl(other, uninit.as_mut_ptr());
                uninit.assume_init()
            }
        }
    }
    impl<'a> openvm_algebra_guest::DivAssignUnsafe<&'a Secp256k1Coord>
    for Secp256k1Coord {
        /// Undefined behaviour when denominator is not coprime to N
        #[inline(always)]
        fn div_assign_unsafe(&mut self, other: &'a Secp256k1Coord) {
            self.div_assign_unsafe_impl(other);
        }
    }
    impl openvm_algebra_guest::DivAssignUnsafe for Secp256k1Coord {
        /// Undefined behaviour when denominator is not coprime to N
        #[inline(always)]
        fn div_assign_unsafe(&mut self, other: Self) {
            self.div_assign_unsafe_impl(&other);
        }
    }
    impl openvm_algebra_guest::DivUnsafe for Secp256k1Coord {
        type Output = Self;
        /// Undefined behaviour when denominator is not coprime to N
        #[inline(always)]
        fn div_unsafe(mut self, other: Self) -> Self::Output {
            self.div_assign_unsafe_impl(&other);
            self
        }
    }
    impl<'a> openvm_algebra_guest::DivUnsafe<&'a Secp256k1Coord> for Secp256k1Coord {
        type Output = Self;
        /// Undefined behaviour when denominator is not coprime to N
        #[inline(always)]
        fn div_unsafe(mut self, other: &'a Secp256k1Coord) -> Self::Output {
            self.div_assign_unsafe_impl(other);
            self
        }
    }
    impl<'a> openvm_algebra_guest::DivUnsafe<&'a Secp256k1Coord> for &Secp256k1Coord {
        type Output = Secp256k1Coord;
        /// Undefined behaviour when denominator is not coprime to N
        #[inline(always)]
        fn div_unsafe(self, other: &'a Secp256k1Coord) -> Self::Output {
            self.div_unsafe_refs_impl(other)
        }
    }
    impl PartialEq for Secp256k1Coord {
        #[inline(always)]
        fn eq(&self, other: &Self) -> bool {
            self.eq_impl(other)
        }
    }
    impl<'a> core::iter::Sum<&'a Secp256k1Coord> for Secp256k1Coord {
        fn sum<I: Iterator<Item = &'a Secp256k1Coord>>(iter: I) -> Self {
            iter.fold(Self::ZERO, |acc, x| &acc + x)
        }
    }
    impl core::iter::Sum for Secp256k1Coord {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::ZERO, |acc, x| &acc + &x)
        }
    }
    impl<'a> core::iter::Product<&'a Secp256k1Coord> for Secp256k1Coord {
        fn product<I: Iterator<Item = &'a Secp256k1Coord>>(iter: I) -> Self {
            iter.fold(Self::ONE, |acc, x| &acc * x)
        }
    }
    impl core::iter::Product for Secp256k1Coord {
        fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::ONE, |acc, x| &acc * &x)
        }
    }
    impl core::ops::Neg for Secp256k1Coord {
        type Output = Secp256k1Coord;
        fn neg(self) -> Self::Output {
            Secp256k1Coord::ZERO - &self
        }
    }
    impl<'a> core::ops::Neg for &'a Secp256k1Coord {
        type Output = Secp256k1Coord;
        fn neg(self) -> Self::Output {
            Secp256k1Coord::ZERO - self
        }
    }
    impl core::fmt::Debug for Secp256k1Coord {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_fmt(format_args!("{0:?}", self.as_le_bytes()))
        }
    }
}
impl openvm_algebra_guest::Reduce for Secp256k1Coord {
    fn reduce_le_bytes(bytes: &[u8]) -> Self {
        let mut res = <Self as openvm_algebra_guest::IntMod>::ZERO;
        let mut base = Self::from_le_bytes(&[255u8; 32usize]);
        base += <Self as openvm_algebra_guest::IntMod>::ONE;
        for chunk in bytes.chunks(32usize).rev() {
            res = res * &base + Self::from_le_bytes(chunk);
        }
        res
    }
}
#[cfg(not(target_os = "zkvm"))]
#[link_section = ".openvm"]
#[no_mangle]
#[used]
static OPENVM_SERIALIZED_MODULUS_0: [u8; 38usize] = [
    1u8, 0u8, 32u8, 0u8, 0u8, 0u8, 47u8, 252u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8,
    255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
    255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
];
#[cfg(not(target_os = "zkvm"))]
mod openvm_intrinsics_ffi {
    #[no_mangle]
    extern "C" fn add_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    ) {}
    #[no_mangle]
    extern "C" fn sub_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    ) {}
    #[no_mangle]
    extern "C" fn mul_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    ) {}
    #[no_mangle]
    extern "C" fn div_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rd: usize,
        rs1: usize,
        rs2: usize,
    ) {}
    #[no_mangle]
    extern "C" fn is_eq_extern_func_fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f(
        rs1: usize,
        rs2: usize,
    ) -> bool {
        let mut x: u32;
        x != 0
    }
}
#[allow(non_snake_case, non_upper_case_globals)]
pub mod openvm_intrinsics_meta_do_not_type_this_by_yourself {
    pub const two_modular_limbs_list: [u8; 64usize] = [
        47u8, 252u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
        255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
        255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 47u8,
        252u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
        255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
        255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
    ];
    pub const limb_list_borders: [usize; 2usize] = [0usize, 64usize];
}
#[allow(non_snake_case)]
pub fn setup_0() {
    #[cfg(not(target_os = "zkvm"))]
    {
        let mut ptr = 0;
        match (&OPENVM_SERIALIZED_MODULUS_0[ptr], &1) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    let kind = ::core::panicking::AssertKind::Eq;
                    ::core::panicking::assert_failed(
                        kind,
                        &*left_val,
                        &*right_val,
                        ::core::option::Option::None,
                    );
                }
            }
        };
        ptr += 1;
        match (&OPENVM_SERIALIZED_MODULUS_0[ptr], &(0usize as u8)) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    let kind = ::core::panicking::AssertKind::Eq;
                    ::core::panicking::assert_failed(
                        kind,
                        &*left_val,
                        &*right_val,
                        ::core::option::Option::None,
                    );
                }
            }
        };
        ptr += 1;
        match (
            &OPENVM_SERIALIZED_MODULUS_0[ptr..ptr + 4]
                .iter()
                .rev()
                .fold(0, |acc, &x| acc * 256 + x as usize),
            &32usize,
        ) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    let kind = ::core::panicking::AssertKind::Eq;
                    ::core::panicking::assert_failed(
                        kind,
                        &*left_val,
                        &*right_val,
                        ::core::option::Option::None,
                    );
                }
            }
        };
        ptr += 4;
        let remaining = &OPENVM_SERIALIZED_MODULUS_0[ptr..];
        #[repr(C, align(32))]
        struct AlignedPlaceholder([u8; 32usize]);
        let mut uninit: core::mem::MaybeUninit<AlignedPlaceholder> = core::mem::MaybeUninit::uninit();
        unsafe {
            let mut tmp = uninit.as_mut_ptr() as usize;
        }
    }
}
pub fn setup_all_moduli() {
    setup_0();
}
use openvm::io::println;
pub fn main() {
    println("-1");
    setup_all_moduli();
    println("0");
    let mut pow = Secp256k1Coord::MODULUS;
    println("1");
    pow[0] -= 2;
    println("2");
    let mut a = Secp256k1Coord::from_u32(1234);
    println("3");
    let mut res = Secp256k1Coord::from_u32(1);
    println("4");
    let inv = res.clone().div_unsafe(&a);
    println("5");
    for pow_bit in pow {
        for j in 0..8 {
            if pow_bit & (1 << j) != 0 {
                res *= &a;
            }
            a *= a.clone();
        }
    }
    println("6");
    match (&res, &inv) {
        (left_val, right_val) => {
            if !(*left_val == *right_val) {
                let kind = ::core::panicking::AssertKind::Eq;
                ::core::panicking::assert_failed(
                    kind,
                    &*left_val,
                    &*right_val,
                    ::core::option::Option::None,
                );
            }
        }
    };
    let two = Secp256k1Coord::from_u32(2);
    let minus_two = Secp256k1Coord::from_le_bytes(&pow);
    match (&(res - &minus_two), &(inv + &two)) {
        (left_val, right_val) => {
            if !(*left_val == *right_val) {
                let kind = ::core::panicking::AssertKind::Eq;
                ::core::panicking::assert_failed(
                    kind,
                    &*left_val,
                    &*right_val,
                    ::core::option::Option::None,
                );
            }
        }
    };
    if two == minus_two {
        openvm::process::panic();
    }
}
