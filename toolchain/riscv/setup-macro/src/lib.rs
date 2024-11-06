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

#[proc_macro]
pub fn moduli_setup(input: TokenStream) -> TokenStream {
    let Stmts { stmts } = parse_macro_input!(input as Stmts);

    let mut output = Vec::new();
    let mut mod_idx = 0usize;

    let mut moduli = Vec::new();

    output.push(TokenStream::from(quote::quote! {
        #[cfg(target_os = "zkvm")]
        use core::mem::MaybeUninit;
        use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

        #[cfg(target_os = "zkvm")]
        use axvm_platform::{
            constants::{Custom1Funct3, ModArithBaseFunct7, CUSTOM_1},
            custom_insn_r,
        };

        #[cfg(not(target_os = "zkvm"))]
        use num_bigint_dig::{traits::ModInverse, BigUint, Sign, ToBigInt};

        #[cfg(not(target_os = "zkvm"))]
        use axvm::intrinsics::biguint_to_limbs;
    }));

    let string_to_bytes = |s: &str| {
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
        bytes.reverse();
        bytes
    };

    for stmt in stmts {
        let result: Result<TokenStream, &str> = match stmt.clone() {
            Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    if let syn::Expr::Path(path) = *assign.left {
                        let struct_name = path.path.segments[0].ident.to_string();

                        if let syn::Expr::Lit(lit) = &*assign.right {
                            if let syn::Lit::Str(str_lit) = &lit.lit {
                                let modulus_bytes = string_to_bytes(&str_lit.value());
                                let limbs = modulus_bytes.len();

                                let struct_name = syn::Ident::new(
                                    &struct_name,
                                    proc_macro::Span::call_site().into(),
                                );

                                // The largest power of two so that at most 10% of all space is wasted
                                let block_size = (limbs + limbs / 9 + 1).next_power_of_two() / 2;
                                let block_size = proc_macro::Literal::usize_unsuffixed(block_size);
                                let block_size =
                                    syn::Lit::new(block_size.to_string().parse::<_>().unwrap());

                                let result = TokenStream::from(quote::quote! {

                                    #[derive(Clone, Eq)]
                                    #[repr(C, align(#block_size))]
                                    pub struct #struct_name([u8; #limbs]);

                                    impl #struct_name {
                                        const MODULUS: [u8; #limbs] = [#(#modulus_bytes),*];
                                        const _MOD_IDX: usize = #mod_idx;

                                        /// The zero element of the field.
                                        pub const ZERO: Self = Self([0; #limbs]);

                                        /// Creates a new #struct_name from an array of bytes.
                                        const fn from_bytes(bytes: [u8; #limbs]) -> Self {
                                            Self(bytes)
                                        }

                                        /// Creates a new #struct_name from a u32.
                                        pub fn from_u32(val: u32) -> Self {
                                            let mut bytes = [0; #limbs];
                                            bytes[..4].copy_from_slice(&val.to_le_bytes());
                                            Self(bytes)
                                        }

                                        /// Value of this #struct_name as an array of bytes.
                                        pub fn as_bytes(&self) -> &[u8; #limbs] {
                                            &(self.0)
                                        }

                                        /// Returns MODULUS as an array of bytes.
                                        const fn modulus() -> [u8; #limbs] {
                                            Self::MODULUS
                                        }

                                        /// Creates a new #struct_name from a BigUint.
                                        #[cfg(not(target_os = "zkvm"))]
                                        pub fn from_biguint(biguint: BigUint) -> Self {
                                            Self(biguint_to_limbs(&biguint))
                                        }

                                        /// Value of this #struct_name as a BigUint.
                                        #[cfg(not(target_os = "zkvm"))]
                                        pub fn as_biguint(&self) -> BigUint {
                                            BigUint::from_bytes_le(self.as_bytes())
                                        }

                                        /// Modulus N as a BigUint.
                                        #[cfg(not(target_os = "zkvm"))]
                                        pub fn modulus_biguint() -> BigUint {
                                            BigUint::from_bytes_le(&Self::MODULUS)
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
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::AddMod as usize,
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
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::SubMod as usize,
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
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::MulMod as usize,
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
                                                let signed_inv = other.as_biguint().mod_inverse(modulus.clone()).unwrap();
                                                let inv = if signed_inv.sign() == Sign::Minus {
                                                    modulus.to_bigint().unwrap() + signed_inv
                                                } else {
                                                    signed_inv
                                                }
                                                .to_biguint()
                                                .unwrap();
                                                *self = Self::from_biguint((self.as_biguint() * inv) % modulus);
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::DivMod as usize,
                                                    self as *mut Self,
                                                    self as *const Self,
                                                    other as *const Self
                                                )
                                            }
                                        }
                                    }

                                    impl<'a> AddAssign<&'a #struct_name> for #struct_name {
                                        #[inline(always)]
                                        fn add_assign(&mut self, other: &'a #struct_name) {
                                            self.add_assign_impl(other);
                                        }
                                    }

                                    impl AddAssign for #struct_name {
                                        #[inline(always)]
                                        fn add_assign(&mut self, other: Self) {
                                            self.add_assign_impl(&other);
                                        }
                                    }

                                    impl Add for #struct_name {
                                        type Output = Self;
                                        #[inline(always)]
                                        fn add(mut self, other: Self) -> Self::Output {
                                            self += other;
                                            self
                                        }
                                    }

                                    impl<'a> Add<&'a #struct_name> for #struct_name {
                                        type Output = Self;
                                        #[inline(always)]
                                        fn add(mut self, other: &'a #struct_name) -> Self::Output {
                                            self += other;
                                            self
                                        }
                                    }

                                    impl<'a> Add<&'a #struct_name> for &#struct_name {
                                        type Output = #struct_name;
                                        #[inline(always)]
                                        fn add(self, other: &'a #struct_name) -> Self::Output {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                let mut res = self.clone();
                                                res += other;
                                                res
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::AddMod as usize,
                                                    uninit.as_mut_ptr(),
                                                    self as *const #struct_name,
                                                    other as *const #struct_name
                                                );
                                                unsafe { uninit.assume_init() }
                                            }
                                        }
                                    }

                                    impl<'a> SubAssign<&'a #struct_name> for #struct_name {
                                        #[inline(always)]
                                        fn sub_assign(&mut self, other: &'a #struct_name) {
                                            self.sub_assign_impl(other);
                                        }
                                    }

                                    impl SubAssign for #struct_name {
                                        #[inline(always)]
                                        fn sub_assign(&mut self, other: Self) {
                                            self.sub_assign_impl(&other);
                                        }
                                    }

                                    impl Sub for #struct_name {
                                        type Output = Self;
                                        #[inline(always)]
                                        fn sub(mut self, other: Self) -> Self::Output {
                                            self -= other;
                                            self
                                        }
                                    }

                                    impl<'a> Sub<&'a #struct_name> for #struct_name {
                                        type Output = Self;
                                        #[inline(always)]
                                        fn sub(mut self, other: &'a #struct_name) -> Self::Output {
                                            self -= other;
                                            self
                                        }
                                    }

                                    impl<'a> Sub<&'a #struct_name> for &#struct_name {
                                        type Output = #struct_name;
                                        #[inline(always)]
                                        fn sub(self, other: &'a #struct_name) -> Self::Output {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                let mut res = self.clone();
                                                res -= other;
                                                res
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::SubMod as usize,
                                                    uninit.as_mut_ptr(),
                                                    self as *const #struct_name,
                                                    other as *const #struct_name
                                                );
                                                unsafe { uninit.assume_init() }
                                            }
                                        }
                                    }

                                    impl<'a> MulAssign<&'a #struct_name> for #struct_name {
                                        #[inline(always)]
                                        fn mul_assign(&mut self, other: &'a #struct_name) {
                                            self.mul_assign_impl(other);
                                        }
                                    }

                                    impl MulAssign for #struct_name {
                                        #[inline(always)]
                                        fn mul_assign(&mut self, other: Self) {
                                            self.mul_assign_impl(&other);
                                        }
                                    }

                                    impl Mul for #struct_name {
                                        type Output = Self;
                                        #[inline(always)]
                                        fn mul(mut self, other: Self) -> Self::Output {
                                            self *= other;
                                            self
                                        }
                                    }

                                    impl<'a> Mul<&'a #struct_name> for #struct_name {
                                        type Output = Self;
                                        #[inline(always)]
                                        fn mul(mut self, other: &'a #struct_name) -> Self::Output {
                                            self *= other;
                                            self
                                        }
                                    }

                                    impl<'a> Mul<&'a #struct_name> for &#struct_name {
                                        type Output = #struct_name;
                                        #[inline(always)]
                                        fn mul(self, other: &'a #struct_name) -> Self::Output {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                let mut res = self.clone();
                                                res *= other;
                                                res
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::MulMod as usize,
                                                    uninit.as_mut_ptr(),
                                                    self as *const #struct_name,
                                                    other as *const #struct_name
                                                );
                                                unsafe { uninit.assume_init() }
                                            }
                                        }
                                    }

                                    impl<'a> DivAssign<&'a #struct_name> for #struct_name {
                                        /// Undefined behaviour when denominator is not coprime to N
                                        #[inline(always)]
                                        fn div_assign(&mut self, other: &'a #struct_name) {
                                            self.div_assign_impl(other);
                                        }
                                    }

                                    impl DivAssign for #struct_name {
                                        /// Undefined behaviour when denominator is not coprime to N
                                        #[inline(always)]
                                        fn div_assign(&mut self, other: Self) {
                                            self.div_assign_impl(&other);
                                        }
                                    }

                                    impl Div for #struct_name {
                                        type Output = Self;
                                        /// Undefined behaviour when denominator is not coprime to N
                                        #[inline(always)]
                                        fn div(mut self, other: Self) -> Self::Output {
                                            self /= other;
                                            self
                                        }
                                    }

                                    impl<'a> Div<&'a #struct_name> for #struct_name {
                                        type Output = Self;
                                        /// Undefined behaviour when denominator is not coprime to N
                                        #[inline(always)]
                                        fn div(mut self, other: &'a #struct_name) -> Self::Output {
                                            self /= other;
                                            self
                                        }
                                    }

                                    impl<'a> Div<&'a #struct_name> for &#struct_name {
                                        type Output = #struct_name;
                                        /// Undefined behaviour when denominator is not coprime to N
                                        #[inline(always)]
                                        fn div(self, other: &'a #struct_name) -> Self::Output {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                let mut res = self.clone();
                                                res /= other;
                                                res
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                                custom_insn_r!(
                                                    CUSTOM_1,
                                                    Custom1Funct3::ModularArithmetic as usize,
                                                    ModArithBaseFunct7::DivMod as usize,
                                                    uninit.as_mut_ptr(),
                                                    self as *const #struct_name,
                                                    other as *const #struct_name
                                                );
                                                unsafe { uninit.assume_init() }
                                            }
                                        }
                                    }

                                    impl PartialEq for #struct_name {
                                        #[inline(always)]
                                        fn eq(&self, other: &Self) -> bool {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                self.as_bytes() == other.as_bytes()
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                let mut x: u32;
                                                unsafe {
                                                    core::arch::asm!(
                                                        ".insn r {opcode}, {funct3}, {funct7}, {rd}, {rs1}, {rs2}",
                                                        opcode = const CUSTOM_1,
                                                        funct3 = const Custom1Funct3::ModularArithmetic as usize,
                                                        funct7 = const ModArithBaseFunct7::IsEqMod as usize,
                                                        rd = out(reg) x,
                                                        rs1 = in(reg) self as *const #struct_name,
                                                        rs2 = in(reg) other as *const #struct_name
                                                    );
                                                }
                                                x != 0
                                            }
                                        }
                                    }

                                    #[cfg(not(target_os = "zkvm"))]
                                    mod helper {
                                        use super::*;
                                        impl Mul<u32> for #struct_name {
                                            type Output = #struct_name;
                                            #[inline(always)]
                                            fn mul(self, other: u32) -> Self::Output {
                                                let mut res = self.clone();
                                                res *= #struct_name::from_u32(other);
                                                res
                                            }
                                        }

                                        impl Mul<u32> for &#struct_name {
                                            type Output = #struct_name;
                                            #[inline(always)]
                                            fn mul(self, other: u32) -> Self::Output {
                                                let mut res = self.clone();
                                                res *= #struct_name::from_u32(other);
                                                res
                                            }
                                        }
                                    }

                                });

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
        #[link_section = ".axiom"]
        #[no_mangle]
        #[used]
        static AXIOM_SERIALIZED_MODULI: [u8; #serialized_len] = [#(#serialized_moduli),*];
    }));

    TokenStream::from_iter(output)
}
