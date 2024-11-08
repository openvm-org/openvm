#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Stmt,
};

struct Stmts {
    stmts: Vec<Stmt>,
}

impl Parse for Stmts {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut stmts = Vec::new();
        while !input.is_empty() {
            stmts.push(input.parse()?);
        }
        Ok(Stmts { stmts })
    }
}

fn string_to_bytes(s: &str) -> Vec<u8> {
    if s.starts_with("0x") {
        return s
            .chars()
            .skip(2)
            .filter(|c| !c.is_whitespace())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .chunks(2)
            .map(|ch| u8::from_str_radix(&ch.iter().rev().collect::<String>(), 16).unwrap())
            .collect();
    }
    let mut digits = s
        .chars()
        .map(|c| c.to_digit(10).expect("Invalid numeric literal"))
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    while !digits.is_empty() {
        let mut rem = 0u32;
        let mut new_digits = Vec::new();
        for &d in digits.iter() {
            rem = rem * 10 + d;
            new_digits.push(rem / 256);
            rem %= 256;
        }
        digits = new_digits.into_iter().skip_while(|&d| d == 0).collect();
        bytes.push(rem as u8);
    }
    bytes
}

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
    let mut mod_idx = 0usize;

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

                                let result = TokenStream::from(
                                    quote::quote_spanned! { span.into() =>

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
                                                                                fn div_assign_impl(&mut self, other: &Self) {
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

                                                                                #[inline(always)]
                                                                                fn add_refs_impl(&self, other: &Self) -> Self {
                                                                                    #[cfg(not(target_os = "zkvm"))]
                                                                                    {
                                                                                        let mut res = self.clone();
                                                                                        res += other;
                                                                                        res
                                                                                    }
                                                                                    #[cfg(target_os = "zkvm")]
                                                                                    {
                                                                                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                                                                                        axvm_platform::custom_insn_r!(
                                                                                            axvm_platform::constants::CUSTOM_1,
                                                                                            axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                                                            axvm_platform::constants::ModArithBaseFunct7::AddMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                                                            uninit.as_mut_ptr(),
                                                                                            self as *const #struct_name,
                                                                                            other as *const #struct_name
                                                                                        );
                                                                                        unsafe { uninit.assume_init() }
                                                                                    }
                                                                                }

                                                                                #[inline(always)]
                                                                                fn sub_refs_impl(&self, other: &Self) -> Self {
                                                                                    #[cfg(not(target_os = "zkvm"))]
                                                                                    {
                                                                                        let mut res = self.clone();
                                                                                        res -= other;
                                                                                        res
                                                                                    }
                                                                                    #[cfg(target_os = "zkvm")]
                                                                                    {
                                                                                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                                                                                        axvm_platform::custom_insn_r!(
                                                                                            axvm_platform::constants::CUSTOM_1,
                                                                                            axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                                                            axvm_platform::constants::ModArithBaseFunct7::SubMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                                                            uninit.as_mut_ptr(),
                                                                                            self as *const #struct_name,
                                                                                            other as *const #struct_name
                                                                                        );
                                                                                        unsafe { uninit.assume_init() }
                                                                                    }
                                                                                }

                                                                                #[inline(always)]
                                                                                fn mul_refs_impl(&self, other: &Self) -> Self {
                                                                                    #[cfg(not(target_os = "zkvm"))]
                                                                                    {
                                                                                        let mut res = self.clone();
                                                                                        res *= other;
                                                                                        res
                                                                                    }
                                                                                    #[cfg(target_os = "zkvm")]
                                                                                    {
                                                                                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                                                                                        axvm_platform::custom_insn_r!(
                                                                                            axvm_platform::constants::CUSTOM_1,
                                                                                            axvm_platform::constants::Custom1Funct3::ModularArithmetic as usize,
                                                                                            axvm_platform::constants::ModArithBaseFunct7::MulMod as usize + Self::MOD_IDX * (axvm_platform::constants::MODULAR_ARITHMETIC_MAX_KINDS as usize),
                                                                                            uninit.as_mut_ptr(),
                                                                                            self as *const #struct_name,
                                                                                            other as *const #struct_name
                                                                                        );
                                                                                        unsafe { uninit.assume_init() }
                                                                                    }
                                                                                }

                                                                                #[inline(always)]
                                                                                fn div_refs_impl(&self, other: &Self) -> Self {
                                                                                    #[cfg(not(target_os = "zkvm"))]
                                                                                    {
                                                                                        let mut res = self.clone();
                                                                                        res /= other;
                                                                                        res
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

                                                                            impl axvm::intrinsics::IntMod for #struct_name {
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
                                                                                    self.add_refs_impl(other)
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

                                                                            impl<'a> core::ops::Sub<&'a #struct_name> for &#struct_name {
                                                                                type Output = #struct_name;
                                                                                #[inline(always)]
                                                                                fn sub(self, other: &'a #struct_name) -> Self::Output {
                                                                                    self.sub_refs_impl(other)
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
                                                                                    self.mul_refs_impl(other)
                                                                                }
                                                                            }

                                                                            impl<'a> core::ops::DivAssign<&'a #struct_name> for #struct_name {
                                                                                /// Undefined behaviour when denominator is not coprime to N
                                                                                #[inline(always)]
                                                                                fn div_assign(&mut self, other: &'a #struct_name) {
                                                                                    self.div_assign_impl(other);
                                                                                }
                                                                            }

                                                                            impl core::ops::DivAssign for #struct_name {
                                                                                /// Undefined behaviour when denominator is not coprime to N
                                                                                #[inline(always)]
                                                                                fn div_assign(&mut self, other: Self) {
                                                                                    self.div_assign_impl(&other);
                                                                                }
                                                                            }

                                                                            impl core::ops::Div for #struct_name {
                                                                                type Output = Self;
                                                                                /// Undefined behaviour when denominator is not coprime to N
                                                                                #[inline(always)]
                                                                                fn div(mut self, other: Self) -> Self::Output {
                                                                                    self /= other;
                                                                                    self
                                                                                }
                                                                            }

                                                                            impl<'a> core::ops::Div<&'a #struct_name> for #struct_name {
                                                                                type Output = Self;
                                                                                /// Undefined behaviour when denominator is not coprime to N
                                                                                #[inline(always)]
                                                                                fn div(mut self, other: &'a #struct_name) -> Self::Output {
                                                                                    self /= other;
                                                                                    self
                                                                                }
                                                                            }

                                                                            impl<'a> core::ops::Div<&'a #struct_name> for &#struct_name {
                                                                                type Output = #struct_name;
                                                                                /// Undefined behaviour when denominator is not coprime to N
                                                                                #[inline(always)]
                                                                                fn div(self, other: &'a #struct_name) -> Self::Output {
                                                                                    self.div_refs_impl(other)
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
                                                                                    Self::ZERO - &self
                                                                                }
                                                                            }

                                                                            impl core::fmt::Debug for #struct_name {
                                                                                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                                                                                    write!(f, "{:?}", self.as_le_bytes())
                                                                                }
                                                                            }

                                                                            #[cfg(not(target_os = "zkvm"))]
                                                                            mod helper {
                                                                                use super::*;
                                                                                impl core::ops::Mul<u32> for #struct_name {
                                                                                    type Output = #struct_name;
                                                                                    #[inline(always)]
                                                                                    fn mul(self, other: u32) -> Self::Output {
                                                                                        let mut res = self.clone();
                                                                                        res *= #struct_name::from_u32(other);
                                                                                        res
                                                                                    }
                                                                                }

                                                                                impl core::ops::Mul<u32> for &#struct_name {
                                                                                    type Output = #struct_name;
                                                                                    #[inline(always)]
                                                                                    fn mul(self, other: u32) -> Self::Output {
                                                                                        let mut res = self.clone();
                                                                                        res *= #struct_name::from_u32(other);
                                                                                        res
                                                                                    }
                                                                                }
                                                                            }

                                                                        },
                                );

                                moduli.push(modulus_bytes);
                                mod_idx += 1;

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

    let mut serialized_moduli = (moduli.len() as u32)
        .to_le_bytes()
        .into_iter()
        .collect::<Vec<_>>();
    for modulus_bytes in moduli {
        serialized_moduli.extend((modulus_bytes.len() as u32).to_le_bytes());
        serialized_moduli.extend(modulus_bytes);
    }
    let serialized_len = serialized_moduli.len();
    // Note: this also prevents the macro from being called twice
    output.push(TokenStream::from(quote::quote! {
        #[cfg(target_os = "zkvm")]
        #[link_section = ".axiom"]
        #[no_mangle]
        #[used]
        static AXIOM_SERIALIZED_MODULI: [u8; #serialized_len] = [#(#serialized_moduli),*];
    }));

    TokenStream::from_iter(output)
}
