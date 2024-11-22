#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use std::sync::atomic::AtomicUsize;

use axvm_macros_common::{string_to_bytes, Stmts};
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{parse_macro_input, Stmt};

static MOD_IDX: AtomicUsize = AtomicUsize::new(0);

/// This macro generates the code to setup the modulus for a given prime. Also it places the moduli into a special static variable to be later extracted from the ELF and used by the VM.
/// Usage:
/// ```
/// moduli_setup! {
///     Bls12381 = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
///     Bn254 = "21888242871839275222246405745257275088696311157297823662689037894645226208583";
/// }
/// ```
/// This creates two structs, `Bls12381` and `Bn254`, each representing the modular arithmetic class (implementing `Add`, `Sub` and so on).
#[proc_macro]
pub fn moduli_setup(input: TokenStream) -> TokenStream {
    let Stmts { stmts } = parse_macro_input!(input as Stmts);

    let mut output = Vec::new();

    let mut moduli = Vec::new();

    let span = proc_macro::Span::call_site();

    for stmt in stmts {
        let result: Result<TokenStream, &str> = match stmt.clone() {
            Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    if let syn::Expr::Path(path) = *assign.left {
                        let struct_name = path.path.segments[0].ident.to_string();

                        if let syn::Expr::Lit(lit) = &*assign.right {
                            if let syn::Lit::Str(str_lit) = &lit.lit {
                                let struct_name = syn::Ident::new(&struct_name, span.into());
                                let mod_idx =
                                    MOD_IDX.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                                let modulus_bytes = string_to_bytes(&str_lit.value());
                                let mut limbs = modulus_bytes.len();

                                if limbs < 32 {
                                    limbs = 32;
                                    proc_macro::Diagnostic::new(proc_macro::Level::Warning, "`limbs` has been set to 32 because it was too small; this is going to be changed once we support more flexible reads").emit();
                                }

                                // The largest power of two so that at most 10% of all space is wasted
                                let block_size =
                                    1usize << ((limbs - 1) ^ (limbs + limbs / 9)).ilog2();
                                let limbs = limbs.next_multiple_of(block_size);
                                let modulus_bytes = modulus_bytes
                                    .into_iter()
                                    .chain(vec![0u8; limbs])
                                    .take(limbs)
                                    .collect::<Vec<_>>();
                                let num_bytes = modulus_bytes.len();

                                let block_size = proc_macro::Literal::usize_unsuffixed(block_size);
                                let block_size =
                                    syn::Lit::new(block_size.to_string().parse::<_>().unwrap());

                                let serialized_modulus =
                                    core::iter::once(1) // 1 for "modulus"
                                        .chain(core::iter::once(mod_idx as u8)) // mod_idx is u8 for now (can make it u32), because we don't know the order of variables in the elf
                                        .chain(
                                            (modulus_bytes.len() as u32)
                                                .to_le_bytes()
                                                .iter()
                                                .copied(),
                                        )
                                        .chain(modulus_bytes.iter().copied())
                                        .collect::<Vec<_>>();
                                let serialized_name = syn::Ident::new(
                                    &format!("AXIOM_SERIALIZED_MODULUS_{}", mod_idx),
                                    span.into(),
                                );
                                let setup_function =
                                    syn::Ident::new(&format!("setup_{}", struct_name), span.into());
                                let serialized_len = serialized_modulus.len();

                                let module_name = format_ident!("algebra_impl_{}", mod_idx);
                                let result = TokenStream::from(
                                    quote::quote_spanned! { span.into() =>
                                        #[cfg(target_os = "zkvm")]
                                        #[link_section = ".axiom"]
                                        #[no_mangle]
                                        #[used]
                                        static #serialized_name: [u8; #serialized_len] = [#(#serialized_modulus),*];

                                        #[derive(Clone, Eq)]
                                        #[repr(C, align(#block_size))]
                                        pub struct #struct_name([u8; #limbs]);

                                        impl #struct_name {
                                            #[inline(always)]
                                            const fn from_const_u8(val: u8) -> Self {
                                                let mut bytes = [0; #limbs];
                                                bytes[0] = val;
                                                Self(bytes)
                                            }

                                            const fn from_const_bytes(bytes: [u8; #limbs]) -> Self {
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
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::AddMod as usize
                                                            + Self::MOD_IDX
                                                                * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        self as *mut Self,
                                                        self as *const Self,
                                                        other as *const Self
                                                    )
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
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::SubMod as usize
                                                            + Self::MOD_IDX
                                                                * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        self as *mut Self,
                                                        self as *const Self,
                                                        other as *const Self
                                                    )
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
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::MulMod as usize
                                                            + Self::MOD_IDX
                                                                * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        self as *mut Self,
                                                        self as *const Self,
                                                        other as *const Self
                                                    )
                                                }
                                            }

                                            #[inline(always)]
                                            fn div_assign_unsafe_impl(&mut self, other: &Self) {
                                                #[cfg(not(target_os = "zkvm"))]
                                                {
                                                    let modulus = Self::modulus_biguint();
                                                    let inv = axvm::intrinsics::uint_mod_inverse(&other.as_biguint(), &modulus);
                                                    *self = Self::from_biguint((self.as_biguint() * inv) % modulus);
                                                }
                                                #[cfg(target_os = "zkvm")]
                                                {
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::DivMod as usize
                                                            + Self::MOD_IDX
                                                                * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        self as *mut Self,
                                                        self as *const Self,
                                                        other as *const Self
                                                    )
                                                }
                                            }

                                            /// SAFETY: `dst_ptr` must be a raw pointer to `&mut Self`.
                                            /// It will be written to only at the very end .
                                            #[inline(always)]
                                            unsafe fn add_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
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
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::AddMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        dst_ptr,
                                                        self as *const #struct_name,
                                                        other as *const #struct_name
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
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::SubMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        dst_ptr,
                                                        self as *const #struct_name,
                                                        other as *const #struct_name
                                                    );
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
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::MulMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        dst_ptr,
                                                        self as *const #struct_name,
                                                        other as *const #struct_name
                                                    );
                                                }
                                            }

                                            #[inline(always)]
                                            fn div_unsafe_refs_impl(&self, other: &Self) -> Self {
                                                #[cfg(not(target_os = "zkvm"))]
                                                {
                                                    let modulus = Self::modulus_biguint();
                                                    let inv = axvm::intrinsics::uint_mod_inverse(&other.as_biguint(), &modulus);
                                                    Self::from_biguint((self.as_biguint() * inv) % modulus)
                                                }
                                                #[cfg(target_os = "zkvm")]
                                                {
                                                    let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                                                    axvm_platform::custom_insn_r!(
                                                        axvm_platform::constants::CUSTOM_1,
                                                        axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                        axvm_platform::constants::ModArithBaseFunct7::DivMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                        uninit.as_mut_ptr(),
                                                        self as *const #struct_name,
                                                        other as *const #struct_name
                                                    );
                                                    unsafe { uninit.assume_init() }
                                                }
                                            }

                                            #[inline(always)]
                                            fn eq_impl(&self, other: &Self) -> bool {
                                                #[cfg(not(target_os = "zkvm"))]
                                                {
                                                    self.as_le_bytes() == other.as_le_bytes()
                                                }
                                                #[cfg(target_os = "zkvm")]
                                                {
                                                    let mut x: u32;
                                                    unsafe {
                                                        core::arch::asm!(
                                                            ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                                                            opcode = const axvm_platform::constants::CUSTOM_1,
                                                            funct3 = const axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                            funct7 = const axvm_platform::constants::ModArithBaseFunct7::IsEqMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                            rd = out(reg) x,
                                                            rs1 = in(reg) self as *const #struct_name,
                                                            rs2 = in(reg) other as *const #struct_name
                                                        );
                                                    }
                                                    x != 0
                                                }
                                            }
                                        }

                                        // Put trait implementations in a private module to avoid conflicts
                                        mod #module_name {
                                            use axvm_algebra::IntMod;

                                            use super::#struct_name;

                                            impl axvm_algebra::IntMod for #struct_name {
                                                type Repr = [u8; #limbs];
                                                type SelfRef<'a> = &'a Self;

                                                const MOD_IDX: usize = #mod_idx;

                                                const MODULUS: Self::Repr = [#(#modulus_bytes),*];

                                                const ZERO: Self = Self([0; #limbs]);

                                                const NUM_BYTES: usize = #num_bytes;

                                                const ONE: Self = Self::from_const_u8(1);

                                                fn from_repr(repr: Self::Repr) -> Self {
                                                    Self(repr)
                                                }

                                                fn from_le_bytes(bytes: &[u8]) -> Self {
                                                    let mut arr = [0u8; #limbs];
                                                    arr.copy_from_slice(bytes);
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

                                                fn as_le_bytes(&self) -> &[u8] {
                                                    &(self.0)
                                                }

                                                #[cfg(not(target_os = "zkvm"))]
                                                fn modulus_biguint() -> num_bigint_dig::BigUint {
                                                    num_bigint_dig::BigUint::from_bytes_le(&Self::MODULUS)
                                                }

                                                #[cfg(not(target_os = "zkvm"))]
                                                fn from_biguint(biguint: num_bigint_dig::BigUint) -> Self {
                                                    Self(axvm::intrinsics::biguint_to_limbs(&biguint))
                                                }

                                                #[cfg(not(target_os = "zkvm"))]
                                                fn as_biguint(&self) -> num_bigint_dig::BigUint {
                                                    num_bigint_dig::BigUint::from_bytes_le(self.as_le_bytes())
                                                }

                                                fn neg_assign(&mut self) {
                                                    unsafe {
                                                        // SAFETY: we borrow self as &Self and as *mut Self but
                                                        // the latter will only be written to at the very end.
                                                        (#struct_name::ZERO).sub_refs_impl(self, self as *const Self as *mut Self);
                                                    }
                                                }

                                                fn double_assign(&mut self) {
                                                    unsafe {
                                                        // SAFETY: we borrow self as &Self and as *mut Self but
                                                        // the latter will only be written to at the very end.
                                                        self.add_refs_impl(self, self as *const Self as *mut Self);
                                                    }
                                                }

                                                fn square_assign(&mut self) {
                                                    unsafe {
                                                        // SAFETY: we borrow self as &Self and as *mut Self but
                                                        // the latter will only be written to at the very end.
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
                                                    let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                                                    unsafe {
                                                        self.add_refs_impl(other, uninit.as_mut_ptr());
                                                        uninit.assume_init()
                                                    }
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

                                            impl<'a> axvm_algebra::DivAssignUnsafe<&'a #struct_name> for #struct_name {
                                                /// Undefined behaviour when denominator is not coprime to N
                                                #[inline(always)]
                                                fn div_assign_unsafe(&mut self, other: &'a #struct_name) {
                                                    self.div_assign_unsafe_impl(other);
                                                }
                                            }

                                            impl axvm_algebra::DivAssignUnsafe for #struct_name {
                                                /// Undefined behaviour when denominator is not coprime to N
                                                #[inline(always)]
                                                fn div_assign_unsafe(&mut self, other: Self) {
                                                    self.div_assign_unsafe_impl(&other);
                                                }
                                            }

                                            impl axvm_algebra::DivUnsafe for #struct_name {
                                                type Output = Self;
                                                /// Undefined behaviour when denominator is not coprime to N
                                                #[inline(always)]
                                                fn div_unsafe(mut self, other: Self) -> Self::Output {
                                                    self.div_assign_unsafe_impl(&other);
                                                    self
                                                }
                                            }

                                            impl<'a> axvm_algebra::DivUnsafe<&'a #struct_name> for #struct_name {
                                                type Output = Self;
                                                /// Undefined behaviour when denominator is not coprime to N
                                                #[inline(always)]
                                                fn div_unsafe(mut self, other: &'a #struct_name) -> Self::Output {
                                                    self.div_assign_unsafe_impl(other);
                                                    self
                                                }
                                            }

                                            impl<'a> axvm_algebra::DivUnsafe<&'a #struct_name> for &#struct_name {
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
                                                    self.eq_impl(other)
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
                                                fn neg(self) -> Self::Output {
                                                    #struct_name::ZERO - &self
                                                }
                                            }

                                            impl<'a> core::ops::Neg for &'a #struct_name {
                                                type Output = #struct_name;
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

                                        impl axvm_algebra::Reduce for #struct_name {
                                            fn reduce_le_bytes(bytes: &[u8]) -> Self {
                                                let mut res = <Self as IntMod>::ZERO;
                                                // base should be 2 ^ #limbs which exceeds what Self can represent
                                                let mut base = Self::from_le_bytes(&[255u8; #limbs]);
                                                base += <Self as IntMod>::ONE;
                                                for chunk in bytes.chunks(#limbs).rev() {
                                                    res = res * &base + Self::from_le_bytes(chunk);
                                                }
                                                res
                                            }
                                        }

                                        #[allow(non_snake_case)]
                                        pub fn #setup_function() {
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                let mut ptr = 0;
                                                assert_eq!(#serialized_name[ptr], 1);
                                                ptr += 1;
                                                assert_eq!(#serialized_name[ptr], #mod_idx as u8);
                                                ptr += 1;
                                                assert_eq!(#serialized_name[ptr..ptr+4].iter().rev().fold(0, |acc, &x| acc * 256 + x as usize), #limbs);
                                                ptr += 4;
                                                let remaining = &#serialized_name[ptr..];

                                                // We are going to use the numeric representation of the `rs2` register to distinguish the chip to setup.
                                                // The transpiler will transform this instruction, based on whether `rs2` is `x0`, `x1` or `x2`, into a `SETUP_ADDSUB`, `SETUP_MULDIV` or `SETUP_ISEQ` instruction.
                                                let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                                                axvm_platform::custom_insn_r!(
                                                    axvm_platform::constants::CUSTOM_1,
                                                    axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                    axvm_platform::constants::ModArithBaseFunct7::SetupMod as usize
                                                        + #mod_idx
                                                            * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                    uninit.as_mut_ptr(),
                                                    remaining.as_ptr(),
                                                    "x0" // will be parsed as 0 and therefore transpiled to SETUP_ADDMOD
                                                );
                                                axvm_platform::custom_insn_r!(
                                                    axvm_platform::constants::CUSTOM_1,
                                                    axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                    axvm_platform::constants::ModArithBaseFunct7::SetupMod as usize
                                                        + #mod_idx
                                                            * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                    uninit.as_mut_ptr(),
                                                    remaining.as_ptr(),
                                                    "x1" // will be parsed as 1 and therefore transpiled to SETUP_MULDIV
                                                );
                                                axvm_platform::custom_insn_r!(
                                                    axvm_platform::constants::CUSTOM_1,
                                                    axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                    axvm_platform::constants::ModArithBaseFunct7::SetupMod as usize
                                                        + #mod_idx
                                                            * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                    uninit.as_mut_ptr(),
                                                    remaining.as_ptr(),
                                                    "x2" // will be parsed as 2 and therefore transpiled to SETUP_ISEQ
                                                );
                                            }
                                        }
                                    },
                                );

                                moduli.push(modulus_bytes);

                                Ok(result)
                            } else {
                                Err("Right side must be a string literal")
                            }
                        } else {
                            Err("Right side must be a string literal")
                        }
                    } else {
                        Err("Left side of assignment must be an identifier")
                    }
                } else {
                    Err("Only simple assignments are supported")
                }
            }
            _ => Err("Only assignments are supported"),
        };
        if let Err(err) = result {
            return syn::Error::new_spanned(stmt, err).to_compile_error().into();
        } else {
            output.push(result.unwrap());
        }
    }

    TokenStream::from_iter(output)
}
