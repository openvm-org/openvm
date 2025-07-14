extern crate alloc;
extern crate proc_macro;

use std::sync::atomic::AtomicUsize;

use num_bigint::BigUint;
use num_prime::nt_funcs::is_prime;
use openvm_macros_common::{string_to_bytes, MacroArgs};
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, LitStr, Token,
};

static MOD_IDX: AtomicUsize = AtomicUsize::new(0);

/// This macro generates the code to setup the modulus for a given prime. Also it places the moduli
/// into a special static variable to be later extracted from the ELF and used by the VM. Usage:
/// ```
/// moduli_declare! {
///     Bls12381 { modulus = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab" },
///     Bn254 { modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583" },
/// }
/// ```
/// This creates two structs, `Bls12381` and `Bn254`, each representing the modular arithmetic class
/// (implementing `Add`, `Sub` and so on).
#[proc_macro]
pub fn moduli_declare(input: TokenStream) -> TokenStream {
    let MacroArgs { items } = parse_macro_input!(input as MacroArgs);

    let mut output = Vec::new();

    let span = proc_macro::Span::call_site();

    for item in items {
        let struct_name = item.name.to_string();
        let struct_name = syn::Ident::new(&struct_name, span.into());
        let mut modulus: Option<String> = None;
        for param in item.params {
            match param.name.to_string().as_str() {
                "modulus" => {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(value),
                        ..
                    }) = param.value
                    {
                        modulus = Some(value.value());
                    } else {
                        return syn::Error::new_spanned(
                            param.value,
                            "Expected a string literal for macro argument `modulus`",
                        )
                        .to_compile_error()
                        .into();
                    }
                }
                _ => {
                    panic!("Unknown parameter {}", param.name);
                }
            }
        }

        // Parsing the parameters is over at this point

        let mod_idx = MOD_IDX.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let modulus = modulus.expect("modulus parameter is required");
        let modulus_bytes = string_to_bytes(&modulus);
        let mut limbs = modulus_bytes.len();
        let mut block_size = 32;

        if limbs <= 32 {
            limbs = 32;
        } else if limbs <= 48 {
            limbs = 48;
            block_size = 16;
        } else {
            panic!("limbs must be at most 48");
        }

        let modulus_bytes = modulus_bytes
            .into_iter()
            .chain(vec![0u8; limbs])
            .take(limbs)
            .collect::<Vec<_>>();

        let modulus_hex = modulus_bytes
            .iter()
            .rev()
            .map(|x| format!("{:02x}", x))
            .collect::<Vec<_>>()
            .join("");
        macro_rules! create_extern_func {
            ($name:ident) => {
                let $name = syn::Ident::new(
                    &format!("{}_{}", stringify!($name), modulus_hex),
                    span.into(),
                );
            };
        }
        create_extern_func!(add_extern_func);
        create_extern_func!(sub_extern_func);
        create_extern_func!(mul_extern_func);
        create_extern_func!(div_extern_func);
        create_extern_func!(is_eq_extern_func);
        create_extern_func!(hint_sqrt_extern_func);
        create_extern_func!(hint_non_qr_extern_func);
        create_extern_func!(moduli_setup_extern_func);

        let block_size = proc_macro::Literal::usize_unsuffixed(block_size);
        let block_size = syn::Lit::new(block_size.to_string().parse::<_>().unwrap());

        let module_name = format_ident!("algebra_impl_{}", mod_idx);

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
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
            #[derive(Clone, Eq, serde::Serialize, serde::Deserialize)]
            #[repr(C, align(#block_size))]
            pub struct #struct_name(#[serde(with = "openvm_algebra_guest::BigArray")] [u8; #limbs]);

            extern "C" {
                fn #add_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #sub_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #mul_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #div_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #is_eq_extern_func(rs1: usize, rs2: usize) -> bool;
                fn #hint_sqrt_extern_func(rs1: usize);
                fn #hint_non_qr_extern_func();
                fn #moduli_setup_extern_func();
            }

            impl #struct_name {
                #[inline(always)]
                const fn from_const_u8(val: u8) -> Self {
                    let mut bytes = [0; #limbs];
                    bytes[0] = val;
                    Self(bytes)
                }

                /// Constructor from little-endian bytes. Does not enforce the integer value of `bytes`
                /// must be less than the modulus.
                pub const fn from_const_bytes(bytes: [u8; #limbs]) -> Self {
                    Self(bytes)
                }

                #[inline(always)]
                fn add_assign_impl(&mut self, other: &Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        *self = Self::from_biguint(
                            (self.as_biguint() + other.as_biguint()) % Self::modulus_biguint(),
                        );
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        unsafe {
                            #add_extern_func(
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
                        let modulus = Self::modulus_biguint();
                        *self = Self::from_biguint(
                            (self.as_biguint() + modulus.clone() - other.as_biguint()) % modulus,
                        );
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        unsafe {
                            #sub_extern_func(
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
                        *self = Self::from_biguint(
                            (self.as_biguint() * other.as_biguint()) % Self::modulus_biguint(),
                        );
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        unsafe {
                            #mul_extern_func(
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
                        let modulus = Self::modulus_biguint();
                        let inv = other.as_biguint().modinv(&modulus).unwrap();
                        *self = Self::from_biguint((self.as_biguint() * inv) % modulus);
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        unsafe {
                            #div_extern_func(
                                self as *mut Self as usize,
                                self as *const Self as usize,
                                other as *const Self as usize,
                            );
                        }
                    }
                }

                /// # Safety
                /// - `dst_ptr` must be a raw pointer to `&mut Self`. It will be written to only at the very end.
                #[inline(always)]
                unsafe fn add_refs_impl<const CHECK_SETUP: bool>(&self, other: &Self, dst_ptr: *mut Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let mut res = self.clone();
                        res += other;
                        // BEWARE order of operations: when dst_ptr = other as pointers
                        let dst = unsafe { &mut *dst_ptr };
                        *dst = res;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        #add_extern_func(
                            dst_ptr as usize,
                            self as *const #struct_name as usize,
                            other as *const #struct_name as usize,
                        );
                    }
                }

                /// SAFETY: `dst_ptr` must be a raw pointer to `&mut Self`.
                /// It will be written to only at the very end .
                #[inline(always)]
                unsafe fn sub_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let mut res = self.clone();
                        res -= other;
                        // BEWARE order of operations: when dst_ptr = other as pointers
                        let dst = unsafe { &mut *dst_ptr };
                        *dst = res;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        unsafe {
                            #sub_extern_func(
                                dst_ptr as usize,
                                self as *const #struct_name as usize,
                                other as *const #struct_name as usize,
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
                        let mut res = self.clone();
                        res *= other;
                        // BEWARE order of operations: when dst_ptr = other as pointers
                        let dst = unsafe { &mut *dst_ptr };
                        *dst = res;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        unsafe {
                            #mul_extern_func(
                                dst_ptr as usize,
                                self as *const #struct_name as usize,
                                other as *const #struct_name as usize,
                            );
                        }
                    }
                }

                #[inline(always)]
                fn div_unsafe_refs_impl(&self, other: &Self) -> Self {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let modulus = Self::modulus_biguint();
                        let inv = other.as_biguint().modinv(&modulus).unwrap();
                        Self::from_biguint((self.as_biguint() * inv) % modulus)
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #div_extern_func(
                                uninit.as_mut_ptr() as usize,
                                self as *const #struct_name as usize,
                                other as *const #struct_name as usize,
                            );
                        }
                        unsafe { uninit.assume_init() }
                    }
                }

                #[inline(always)]
                unsafe fn eq_impl<const CHECK_SETUP: bool>(&self, other: &Self) -> bool {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        self.as_le_bytes() == other.as_le_bytes()
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        #is_eq_extern_func(self as *const #struct_name as usize, other as *const #struct_name as usize)
                    }
                }

                // Helper function to call the setup instruction on first use
                #[inline(always)]
                #[cfg(target_os = "zkvm")]
                fn set_up_once() {
                    static is_setup: ::openvm_algebra_guest::once_cell::race::OnceBool = ::openvm_algebra_guest::once_cell::race::OnceBool::new();
                    is_setup.get_or_init(|| {
                        unsafe { #moduli_setup_extern_func(); }
                        true
                    });
                }
                #[inline(always)]
                #[cfg(not(target_os = "zkvm"))]
                fn set_up_once() {
                    // No-op for non-ZKVM targets
                }
            }

            // Put trait implementations in a private module to avoid conflicts
            mod #module_name {
                use openvm_algebra_guest::IntMod;

                use super::#struct_name;

                impl IntMod for #struct_name {
                    type Repr = [u8; #limbs];
                    type SelfRef<'a> = &'a Self;

                    const MODULUS: Self::Repr = [#(#modulus_bytes),*];

                    const ZERO: Self = Self([0; #limbs]);

                    const NUM_LIMBS: usize = #limbs;

                    const ONE: Self = Self::from_const_u8(1);

                    fn from_repr(repr: Self::Repr) -> Self {
                        Self(repr)
                    }

                    fn from_le_bytes(bytes: &[u8]) -> Option<Self> {
                        let elt = Self::from_le_bytes_unchecked(bytes);
                        if elt.is_reduced() {
                            Some(elt)
                        } else {
                            None
                        }
                    }

                    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
                        let elt = Self::from_be_bytes_unchecked(bytes);
                        if elt.is_reduced() {
                            Some(elt)
                        } else {
                            None
                        }
                    }

                    fn from_le_bytes_unchecked(bytes: &[u8]) -> Self {
                        let mut arr = [0u8; #limbs];
                        arr.copy_from_slice(bytes);
                        Self(arr)
                    }

                    fn from_be_bytes_unchecked(bytes: &[u8]) -> Self {
                        let mut arr = [0u8; #limbs];
                        for (a, b) in arr.iter_mut().zip(bytes.iter().rev()) {
                            *a = *b;
                        }
                        Self(arr)
                    }

                    fn from_u8(val: u8) -> Self {
                        Self::from_const_u8(val)
                    }

                    fn from_u32(val: u32) -> Self {
                        let mut bytes = [0; #limbs];
                        bytes[..4].copy_from_slice(&val.to_le_bytes());
                        Self(bytes)
                    }

                    fn from_u64(val: u64) -> Self {
                        let mut bytes = [0; #limbs];
                        bytes[..8].copy_from_slice(&val.to_le_bytes());
                        Self(bytes)
                    }

                    #[inline(always)]
                    fn as_le_bytes(&self) -> &[u8] {
                        &(self.0)
                    }

                    #[inline(always)]
                    fn to_be_bytes(&self) -> [u8; #limbs] {
                        core::array::from_fn(|i| self.0[#limbs - 1 - i])
                    }

                    #[cfg(not(target_os = "zkvm"))]
                    fn modulus_biguint() -> num_bigint::BigUint {
                        num_bigint::BigUint::from_bytes_le(&Self::MODULUS)
                    }

                    #[cfg(not(target_os = "zkvm"))]
                    fn from_biguint(biguint: num_bigint::BigUint) -> Self {
                        Self(openvm::utils::biguint_to_limbs(&biguint))
                    }

                    #[cfg(not(target_os = "zkvm"))]
                    fn as_biguint(&self) -> num_bigint::BigUint {
                        num_bigint::BigUint::from_bytes_le(self.as_le_bytes())
                    }

                    #[inline(always)]
                    fn neg_assign(&mut self) {
                        unsafe {
                            // SAFETY: we borrow self as &Self and as *mut Self but
                            // the latter will only be written to at the very end.
                            (#struct_name::ZERO).sub_refs_impl(self, self as *const Self as *mut Self);
                        }
                    }

                    #[inline(always)]
                    fn double_assign(&mut self) {
                        unsafe {
                            // SAFETY: we borrow self as &Self and as *mut Self but
                            // the latter will only be written to at the very end.
                            self.add_refs_impl::<true>(self, self as *const Self as *mut Self);
                        }
                    }

                    #[inline(always)]
                    fn square_assign(&mut self) {
                        unsafe {
                            // SAFETY: we borrow self as &Self and as *mut Self but
                            // the latter will only be written to at the very end.
                            self.mul_refs_impl(self, self as *const Self as *mut Self);
                        }
                    }

                    #[inline(always)]
                    fn double(&self) -> Self {
                        self + self
                    }

                    #[inline(always)]
                    fn square(&self) -> Self {
                        self * self
                    }

                    #[inline(always)]
                    fn cube(&self) -> Self {
                        &self.square() * self
                    }

                    /// If `self` is not in its canonical form, the proof will fail to verify.
                    /// This means guest execution will never terminate (either successfully or
                    /// unsuccessfully) if `self` is not in its canonical form.
                    // is_eq_mod enforces `self` is less than `modulus`
                    fn assert_reduced(&self) {
                        // This must not be optimized out
                        let _ = core::hint::black_box(PartialEq::eq(self, self));
                    }

                    fn is_reduced(&self) -> bool {
                        // limbs are little endian
                        for (x_limb, p_limb) in self.0.iter().rev().zip(Self::MODULUS.iter().rev()) {
                            if x_limb < p_limb {
                                return true;
                            } else if x_limb > p_limb {
                                return false;
                            }
                        }
                        // At this point, all limbs are equal
                        false
                    }

                    #[inline(always)]
                    fn set_up_once() {
                        Self::set_up_once();
                    }

                    #[inline(always)]
                    unsafe fn eq_impl<const CHECK_SETUP: bool>(&self, other: &Self) -> bool {
                        Self::eq_impl::<CHECK_SETUP>(self, other)
                    }

                    #[inline(always)]
                    unsafe fn add_ref<const CHECK_SETUP: bool>(&self, other: &Self) -> Self {
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        self.add_refs_impl::<CHECK_SETUP>(other, uninit.as_mut_ptr());
                        uninit.assume_init()
                    }
                }

                impl<'a> core::ops::AddAssign<&'a #struct_name> for #struct_name {
                    #[inline(always)]
                    fn add_assign(&mut self, other: &'a #struct_name) {
                        self.add_assign_impl(other);
                    }
                }

                impl core::ops::AddAssign for #struct_name {
                    #[inline(always)]
                    fn add_assign(&mut self, other: Self) {
                        self.add_assign_impl(&other);
                    }
                }

                impl core::ops::Add for #struct_name {
                    type Output = Self;
                    #[inline(always)]
                    fn add(mut self, other: Self) -> Self::Output {
                        self += other;
                        self
                    }
                }

                impl<'a> core::ops::Add<&'a #struct_name> for #struct_name {
                    type Output = Self;
                    #[inline(always)]
                    fn add(mut self, other: &'a #struct_name) -> Self::Output {
                        self += other;
                        self
                    }
                }

                impl<'a> core::ops::Add<&'a #struct_name> for &#struct_name {
                    type Output = #struct_name;
                    #[inline(always)]
                    fn add(self, other: &'a #struct_name) -> Self::Output {
                        // Safety: ensure setup
                        unsafe { self.add_ref::<true>(other) }
                    }
                }

                impl<'a> core::ops::SubAssign<&'a #struct_name> for #struct_name {
                    #[inline(always)]
                    fn sub_assign(&mut self, other: &'a #struct_name) {
                        self.sub_assign_impl(other);
                    }
                }

                impl core::ops::SubAssign for #struct_name {
                    #[inline(always)]
                    fn sub_assign(&mut self, other: Self) {
                        self.sub_assign_impl(&other);
                    }
                }

                impl core::ops::Sub for #struct_name {
                    type Output = Self;
                    #[inline(always)]
                    fn sub(mut self, other: Self) -> Self::Output {
                        self -= other;
                        self
                    }
                }

                impl<'a> core::ops::Sub<&'a #struct_name> for #struct_name {
                    type Output = Self;
                    #[inline(always)]
                    fn sub(mut self, other: &'a #struct_name) -> Self::Output {
                        self -= other;
                        self
                    }
                }

                impl<'a> core::ops::Sub<&'a #struct_name> for &'a #struct_name {
                    type Output = #struct_name;
                    #[inline(always)]
                    fn sub(self, other: &'a #struct_name) -> Self::Output {
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            self.sub_refs_impl(other, uninit.as_mut_ptr());
                            uninit.assume_init()
                        }
                    }
                }

                impl<'a> core::ops::MulAssign<&'a #struct_name> for #struct_name {
                    #[inline(always)]
                    fn mul_assign(&mut self, other: &'a #struct_name) {
                        self.mul_assign_impl(other);
                    }
                }

                impl core::ops::MulAssign for #struct_name {
                    #[inline(always)]
                    fn mul_assign(&mut self, other: Self) {
                        self.mul_assign_impl(&other);
                    }
                }

                impl core::ops::Mul for #struct_name {
                    type Output = Self;
                    #[inline(always)]
                    fn mul(mut self, other: Self) -> Self::Output {
                        self *= other;
                        self
                    }
                }

                impl<'a> core::ops::Mul<&'a #struct_name> for #struct_name {
                    type Output = Self;
                    #[inline(always)]
                    fn mul(mut self, other: &'a #struct_name) -> Self::Output {
                        self *= other;
                        self
                    }
                }

                impl<'a> core::ops::Mul<&'a #struct_name> for &#struct_name {
                    type Output = #struct_name;
                    #[inline(always)]
                    fn mul(self, other: &'a #struct_name) -> Self::Output {
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            self.mul_refs_impl(other, uninit.as_mut_ptr());
                            uninit.assume_init()
                        }
                    }
                }

                impl<'a> openvm_algebra_guest::DivAssignUnsafe<&'a #struct_name> for #struct_name {
                    /// Undefined behaviour when denominator is not coprime to N
                    #[inline(always)]
                    fn div_assign_unsafe(&mut self, other: &'a #struct_name) {
                        self.div_assign_unsafe_impl(other);
                    }
                }

                impl openvm_algebra_guest::DivAssignUnsafe for #struct_name {
                    /// Undefined behaviour when denominator is not coprime to N
                    #[inline(always)]
                    fn div_assign_unsafe(&mut self, other: Self) {
                        self.div_assign_unsafe_impl(&other);
                    }
                }

                impl openvm_algebra_guest::DivUnsafe for #struct_name {
                    type Output = Self;
                    /// Undefined behaviour when denominator is not coprime to N
                    #[inline(always)]
                    fn div_unsafe(mut self, other: Self) -> Self::Output {
                        self.div_assign_unsafe_impl(&other);
                        self
                    }
                }

                impl<'a> openvm_algebra_guest::DivUnsafe<&'a #struct_name> for #struct_name {
                    type Output = Self;
                    /// Undefined behaviour when denominator is not coprime to N
                    #[inline(always)]
                    fn div_unsafe(mut self, other: &'a #struct_name) -> Self::Output {
                        self.div_assign_unsafe_impl(other);
                        self
                    }
                }

                impl<'a> openvm_algebra_guest::DivUnsafe<&'a #struct_name> for &#struct_name {
                    type Output = #struct_name;
                    /// Undefined behaviour when denominator is not coprime to N
                    #[inline(always)]
                    fn div_unsafe(self, other: &'a #struct_name) -> Self::Output {
                        self.div_unsafe_refs_impl(other)
                    }
                }

                impl PartialEq for #struct_name {
                    #[inline(always)]
                    fn eq(&self, other: &Self) -> bool {
                        // Safety: must check setup
                        unsafe { self.eq_impl::<true>(other) }
                    }
                }

                impl<'a> core::iter::Sum<&'a #struct_name> for #struct_name {
                    fn sum<I: Iterator<Item = &'a #struct_name>>(iter: I) -> Self {
                        iter.fold(Self::ZERO, |acc, x| &acc + x)
                    }
                }

                impl core::iter::Sum for #struct_name {
                    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                        iter.fold(Self::ZERO, |acc, x| &acc + &x)
                    }
                }

                impl<'a> core::iter::Product<&'a #struct_name> for #struct_name {
                    fn product<I: Iterator<Item = &'a #struct_name>>(iter: I) -> Self {
                        iter.fold(Self::ONE, |acc, x| &acc * x)
                    }
                }

                impl core::iter::Product for #struct_name {
                    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                        iter.fold(Self::ONE, |acc, x| &acc * &x)
                    }
                }

                impl core::ops::Neg for #struct_name {
                    type Output = #struct_name;
                    #[inline(always)]
                    fn neg(self) -> Self::Output {
                        #struct_name::ZERO - &self
                    }
                }

                impl<'a> core::ops::Neg for &'a #struct_name {
                    type Output = #struct_name;
                    #[inline(always)]
                    fn neg(self) -> Self::Output {
                        #struct_name::ZERO - self
                    }
                }

                impl core::fmt::Debug for #struct_name {
                    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        write!(f, "{:?}", self.as_le_bytes())
                    }
                }
            }

            impl openvm_algebra_guest::Reduce for #struct_name {
                fn reduce_le_bytes(bytes: &[u8]) -> Self {
                    let mut res = <Self as openvm_algebra_guest::IntMod>::ZERO;
                    // base should be 2 ^ #limbs which exceeds what Self can represent
                    let mut base = <Self as openvm_algebra_guest::IntMod>::from_le_bytes_unchecked(&[255u8; #limbs]);
                    base += <Self as openvm_algebra_guest::IntMod>::ONE;
                    for chunk in bytes.chunks(#limbs).rev() {
                        res = res * &base + <Self as openvm_algebra_guest::IntMod>::from_le_bytes_unchecked(chunk);
                    }
                    openvm_algebra_guest::IntMod::assert_reduced(&res);
                    res
                }
            }
        });

        output.push(result);

        let modulus_biguint = BigUint::from_bytes_le(&modulus_bytes);
        let modulus_is_prime = is_prime(&modulus_biguint, None);

        if modulus_is_prime.probably() {
            // implement Field and Sqrt traits for prime moduli
            let field_and_sqrt_impl = TokenStream::from(quote::quote_spanned! { span.into() =>
                impl ::openvm_algebra_guest::Field for #struct_name {
                    const ZERO: Self = <Self as ::openvm_algebra_guest::IntMod>::ZERO;
                    const ONE: Self = <Self as ::openvm_algebra_guest::IntMod>::ONE;

                    type SelfRef<'a> = &'a Self;

                    fn double_assign(&mut self) {
                        ::openvm_algebra_guest::IntMod::double_assign(self);
                    }

                    fn square_assign(&mut self) {
                        ::openvm_algebra_guest::IntMod::square_assign(self);
                    }

                }

                impl openvm_algebra_guest::Sqrt for #struct_name {
                    // Returns a sqrt of self if it exists, otherwise None.
                    // Note that we use a hint-based approach to prove whether the square root exists.
                    // This approach works for prime moduli, but not necessarily for composite moduli,
                    // which is why we have the sqrt method in the Field trait, not the IntMod trait.
                    fn sqrt(&self) -> Option<Self> {
                        match self.honest_host_sqrt() {
                            // self is a square
                            Some(Some(sqrt)) => Some(sqrt),
                            // self is not a square
                            Some(None) => None,
                            // host is dishonest
                            None => {
                                // host is dishonest, enter infinite loop
                                loop {
                                    openvm::io::println("ERROR: Square root hint is invalid. Entering infinite loop.");
                                }
                            }
                        }
                    }
                }

                impl #struct_name {
                    // Returns None if the hint is incorrect (i.e. the host is dishonest)
                    // Returns Some(None) if the hint proves that self is not a quadratic residue
                    // Otherwise, returns Some(Some(sqrt)) where sqrt is a square root of self
                    fn honest_host_sqrt(&self) -> Option<Option<Self>> {
                        let (is_square, sqrt) = self.hint_sqrt_impl()?;

                        if is_square {
                            // ensure sqrt < modulus
                            <Self as ::openvm_algebra_guest::IntMod>::assert_reduced(&sqrt);

                            if &(&sqrt * &sqrt) == self {
                                Some(Some(sqrt))
                            } else {
                                None
                            }
                        } else {
                            // ensure sqrt < modulus
                            <Self as ::openvm_algebra_guest::IntMod>::assert_reduced(&sqrt);

                            if &sqrt * &sqrt == self * Self::get_non_qr() {
                                Some(None)
                            } else {
                                None
                            }
                        }
                    }


                    // Returns None if the hint is malformed.
                    // Otherwise, returns Some((is_square, sqrt)) where sqrt is a square root of self if is_square is true,
                    // and a square root of self * non_qr if is_square is false.
                    fn hint_sqrt_impl(&self) -> Option<(bool, Self)> {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!();
                        }
                        #[cfg(target_os = "zkvm")]
                        {
                            use ::openvm_algebra_guest::{openvm_custom_insn, openvm_rv32im_guest}; // needed for hint_store_u32! and hint_buffer_u32!

                            let is_square = core::mem::MaybeUninit::<u32>::uninit();
                            let sqrt = core::mem::MaybeUninit::<#struct_name>::uninit();
                            unsafe {
                                #hint_sqrt_extern_func(self as *const #struct_name as usize);
                                let is_square_ptr = is_square.as_ptr() as *const u32;
                                openvm_rv32im_guest::hint_store_u32!(is_square_ptr);
                                openvm_rv32im_guest::hint_buffer_u32!(sqrt.as_ptr() as *const u8, <#struct_name as ::openvm_algebra_guest::IntMod>::NUM_LIMBS / 4);
                                let is_square = is_square.assume_init();
                                if is_square == 0 || is_square == 1 {
                                    Some((is_square == 1, sqrt.assume_init()))
                                } else {
                                    None
                                }
                            }
                        }
                    }

                    // Generate a non quadratic residue by using a hint
                    fn init_non_qr() -> alloc::boxed::Box<#struct_name> {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!();
                        }
                        #[cfg(target_os = "zkvm")]
                        {
                            use ::openvm_algebra_guest::{openvm_custom_insn, openvm_rv32im_guest}; // needed for hint_buffer_u32!

                            let mut non_qr_uninit = core::mem::MaybeUninit::<Self>::uninit();
                            let mut non_qr;
                            unsafe {
                                #hint_non_qr_extern_func();
                                let ptr = non_qr_uninit.as_ptr() as *const u8;
                                openvm_rv32im_guest::hint_buffer_u32!(ptr, <Self as ::openvm_algebra_guest::IntMod>::NUM_LIMBS / 4);
                                non_qr = non_qr_uninit.assume_init();
                            }
                            // ensure non_qr < modulus
                            <Self as ::openvm_algebra_guest::IntMod>::assert_reduced(&non_qr);

                            use ::openvm_algebra_guest::{DivUnsafe, ExpBytes};
                            // construct exp = (p-1)/2 as an integer by first constraining exp = (p-1)/2 (mod p) and then exp < p
                            let exp = -<Self as ::openvm_algebra_guest::IntMod>::ONE.div_unsafe(Self::from_const_u8(2));
                            <Self as ::openvm_algebra_guest::IntMod>::assert_reduced(&exp);

                            if non_qr.exp_bytes(true, &<Self as ::openvm_algebra_guest::IntMod>::to_be_bytes(&exp)) != -<Self as ::openvm_algebra_guest::IntMod>::ONE
                            {
                                // non_qr is not a non quadratic residue, so host is dishonest
                                loop {
                                    openvm::io::println("ERROR: Non quadratic residue hint is invalid. Entering infinite loop.");
                                }
                            }

                            alloc::boxed::Box::new(non_qr)
                        }
                    }

                    // This function is public for use in tests
                    pub fn get_non_qr() -> &'static #struct_name {
                        static non_qr: ::openvm_algebra_guest::once_cell::race::OnceBox<#struct_name> = ::openvm_algebra_guest::once_cell::race::OnceBox::new();
                        &non_qr.get_or_init(Self::init_non_qr)
                    }
                }
            });

            output.push(field_and_sqrt_impl);
        }
    }

    TokenStream::from_iter(output)
}

struct ModuliDefine {
    items: Vec<LitStr>,
}

impl Parse for ModuliDefine {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let items = input.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
        Ok(Self {
            items: items.into_iter().collect(),
        })
    }
}

#[proc_macro]
pub fn moduli_init(input: TokenStream) -> TokenStream {
    let ModuliDefine { items } = parse_macro_input!(input as ModuliDefine);

    let mut externs = Vec::new();
    let mut openvm_section = Vec::new();

    // List of all modular limbs in one (that is, with a compile-time known size) array.
    let mut two_modular_limbs_flattened_list = Vec::<u8>::new();
    // List of "bars" between adjacent modular limbs sublists.
    let mut limb_list_borders = vec![0usize];

    let span = proc_macro::Span::call_site();

    for (mod_idx, item) in items.into_iter().enumerate() {
        let modulus = item.value();
        println!("[init] modulus #{} = {}", mod_idx, modulus);

        let modulus_bytes = string_to_bytes(&modulus);
        let mut limbs = modulus_bytes.len();
        let mut block_size = 32;

        if limbs <= 32 {
            limbs = 32;
        } else if limbs <= 48 {
            limbs = 48;
            block_size = 16;
        } else {
            panic!("limbs must be at most 48");
        }

        let block_size = proc_macro::Literal::usize_unsuffixed(block_size);
        let block_size = syn::Lit::new(block_size.to_string().parse::<_>().unwrap());

        let modulus_bytes = modulus_bytes
            .into_iter()
            .chain(vec![0u8; limbs])
            .take(limbs)
            .collect::<Vec<_>>();

        // We need two copies of modular limbs for Fp2 setup.
        let doubled_modulus = [modulus_bytes.clone(), modulus_bytes.clone()].concat();
        two_modular_limbs_flattened_list.extend(doubled_modulus);
        limb_list_borders.push(two_modular_limbs_flattened_list.len());

        let modulus_hex = modulus_bytes
            .iter()
            .rev()
            .map(|x| format!("{:02x}", x))
            .collect::<Vec<_>>()
            .join("");

        let serialized_modulus =
            core::iter::once(1) // 1 for "modulus"
                .chain(core::iter::once(mod_idx as u8)) // mod_idx is u8 for now (can make it u32), because we don't know the order of
                // variables in the elf
                .chain((modulus_bytes.len() as u32).to_le_bytes().iter().copied())
                .chain(modulus_bytes.iter().copied())
                .collect::<Vec<_>>();
        let serialized_name = syn::Ident::new(
            &format!("OPENVM_SERIALIZED_MODULUS_{}", mod_idx),
            span.into(),
        );
        let serialized_len = serialized_modulus.len();
        let setup_extern_func = syn::Ident::new(
            &format!("moduli_setup_extern_func_{}", modulus_hex),
            span.into(),
        );

        openvm_section.push(quote::quote_spanned! { span.into() =>
            #[cfg(target_os = "zkvm")]
            #[link_section = ".openvm"]
            #[no_mangle]
            #[used]
            static #serialized_name: [u8; #serialized_len] = [#(#serialized_modulus),*];
        });

        for op_type in ["add", "sub", "mul", "div"] {
            let func_name = syn::Ident::new(
                &format!("{}_extern_func_{}", op_type, modulus_hex),
                span.into(),
            );
            let mut chars = op_type.chars().collect::<Vec<_>>();
            chars[0] = chars[0].to_ascii_uppercase();
            let local_opcode = syn::Ident::new(
                &format!("{}Mod", chars.iter().collect::<String>()),
                span.into(),
            );
            externs.push(quote::quote_spanned! { span.into() =>
                #[no_mangle]
                extern "C" fn #func_name(rd: usize, rs1: usize, rs2: usize) {
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_algebra_guest::OPCODE,
                        funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3 as usize,
                        funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::#local_opcode as usize + #mod_idx * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                        rd = In rd,
                        rs1 = In rs1,
                        rs2 = In rs2
                    )
                }
            });
        }

        let is_eq_extern_func =
            syn::Ident::new(&format!("is_eq_extern_func_{}", modulus_hex), span.into());
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #is_eq_extern_func(rs1: usize, rs2: usize) -> bool {
                let mut x: u32;
                openvm::platform::custom_insn_r!(
                    opcode = ::openvm_algebra_guest::OPCODE,
                    funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3 as usize,
                    funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::IsEqMod as usize + #mod_idx * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                    rd = Out x,
                    rs1 = In rs1,
                    rs2 = In rs2
                );
                x != 0
            }
        });

        let hint_non_qr_extern_func = syn::Ident::new(
            &format!("hint_non_qr_extern_func_{}", modulus_hex),
            span.into(),
        );
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #hint_non_qr_extern_func() {
                openvm::platform::custom_insn_r!(
                    opcode = ::openvm_algebra_guest::OPCODE,
                    funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3 as usize,
                    funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::HintNonQr as usize + #mod_idx * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                    rd = Const "x0",
                    rs1 = Const "x0",
                    rs2 = Const "x0"
                );
            }


        });

        // This function will be defined regardless of whether the modulus is prime or not,
        // but it will be called only if the modulus is prime.
        let hint_sqrt_extern_func = syn::Ident::new(
            &format!("hint_sqrt_extern_func_{}", modulus_hex),
            span.into(),
        );
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #hint_sqrt_extern_func(rs1: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = ::openvm_algebra_guest::OPCODE,
                    funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3 as usize,
                    funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::HintSqrt as usize + #mod_idx * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                    rd = Const "x0",
                    rs1 = In rs1,
                    rs2 = Const "x0"
                );
            }
        });

        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #setup_extern_func() {
                #[cfg(target_os = "zkvm")]
                {
                    let mut ptr = 0;
                    assert_eq!(super::#serialized_name[ptr], 1);
                    ptr += 1;
                    assert_eq!(super::#serialized_name[ptr], #mod_idx as u8);
                    ptr += 1;
                    assert_eq!(super::#serialized_name[ptr..ptr+4].iter().rev().fold(0, |acc, &x| acc * 256 + x as usize), #limbs);
                    ptr += 4;
                    let remaining = &super::#serialized_name[ptr..];

                    // To avoid importing #struct_name, we create a placeholder struct with the same size and alignment.
                    #[repr(C, align(#block_size))]
                    struct AlignedPlaceholder([u8; #limbs]);

                    // We are going to use the numeric representation of the `rs2` register to distinguish the chip to setup.
                    // The transpiler will transform this instruction, based on whether `rs2` is `x0`, `x1` or `x2`, into a `SETUP_ADDSUB`, `SETUP_MULDIV` or `SETUP_ISEQ` instruction.
                    let mut uninit: core::mem::MaybeUninit<AlignedPlaceholder> = core::mem::MaybeUninit::uninit();
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_algebra_guest::OPCODE,
                        funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3,
                        funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::SetupMod as usize
                            + #mod_idx
                                * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                        rd = In uninit.as_mut_ptr(),
                        rs1 = In remaining.as_ptr(),
                        rs2 = Const "x0" // will be parsed as 0 and therefore transpiled to SETUP_ADDMOD
                    );
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_algebra_guest::OPCODE,
                        funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3,
                        funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::SetupMod as usize
                            + #mod_idx
                                * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                        rd = In uninit.as_mut_ptr(),
                        rs1 = In remaining.as_ptr(),
                        rs2 = Const "x1" // will be parsed as 1 and therefore transpiled to SETUP_MULDIV
                    );
                    unsafe {
                        // This should not be x0:
                        let mut tmp = uninit.as_mut_ptr() as usize;
                        openvm::platform::custom_insn_r!(
                            opcode = ::openvm_algebra_guest::OPCODE,
                            funct3 = ::openvm_algebra_guest::MODULAR_ARITHMETIC_FUNCT3 as usize,
                            funct7 = ::openvm_algebra_guest::ModArithBaseFunct7::SetupMod as usize
                                + #mod_idx
                                    * (::openvm_algebra_guest::ModArithBaseFunct7::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                            rd = InOut tmp,
                            rs1 = In remaining.as_ptr(),
                            rs2 = Const "x2" // will be parsed as 2 and therefore transpiled to SETUP_ISEQ
                        );
                        // rd = inout(reg) is necessary because this instruction will write to `rd` register
                    }
                }
            }
        });
    }

    let total_limbs_cnt = two_modular_limbs_flattened_list.len();
    let cnt_limbs_list_len = limb_list_borders.len();
    TokenStream::from(quote::quote_spanned! { span.into() =>
        #(#openvm_section)*
        #[allow(non_snake_case)]
        #[cfg(target_os = "zkvm")]
        mod openvm_intrinsics_ffi {
            #(#externs)*
        }
        #[allow(non_snake_case, non_upper_case_globals)]
        pub mod openvm_intrinsics_meta_do_not_type_this_by_yourself {
            pub const two_modular_limbs_list: [u8; #total_limbs_cnt] = [#(#two_modular_limbs_flattened_list),*];
            pub const limb_list_borders: [usize; #cnt_limbs_list_len] = [#(#limb_list_borders),*];
        }
    })
}
