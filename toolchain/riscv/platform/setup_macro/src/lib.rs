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
        use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
        #[cfg(target_os = "zkvm")]
        use core::{borrow::BorrowMut, mem::MaybeUninit};

        #[cfg(not(target_os = "zkvm"))]
        use num_bigint_dig::{traits::ModInverse, BigUint, Sign, ToBigInt};
        #[cfg(not(target_os = "zkvm"))]
        use axvm::intrinsics::biguint_to_limbs;
    }));

    for stmt in stmts {
        match stmt.clone() {
            Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    if let syn::Expr::Path(path) = *assign.left {
                        let ident = path.path.segments[0].ident.to_string();

                        if let syn::Expr::Lit(lit) = &*assign.right {
                            if let syn::Lit::Str(str_lit) = &lit.lit {
                                let modulus = num_bigint::BigUint::parse_bytes(
                                    str_lit.value().as_bytes(),
                                    10,
                                )
                                .expect("Failed to parse modulus");
                                let modulus_bytes = modulus.to_bytes_le();
                                let limbs = modulus_bytes.len();

                                let struct_name = format!("IntMod_{}", ident);
                                let struct_name = syn::Ident::new(
                                    &struct_name,
                                    proc_macro::Span::call_site().into(),
                                );

                                output.push(TokenStream::from(quote::quote! {

                                    #[derive(Clone)]
                                    #[repr(C, align(32))]
                                    pub struct #struct_name([u8; #limbs]);

                                    impl #struct_name {
                                        const MODULUS: [u8; #limbs] = [#(#modulus_bytes),*];
                                        const MOD_IDX: usize = #mod_idx;

                                        /// Creates a new #struct_name from an array of bytes.
                                        pub fn from_bytes(bytes: [u8; #limbs]) -> Self {
                                            Self(bytes)
                                        }

                                        /// Value of this #struct_name as an array of bytes.
                                        pub fn as_bytes(&self) -> &[u8; #limbs] {
                                            &(self.0)
                                        }

                                        /// Creates a new #struct_name from a BigUint.
                                        #[cfg(not(target_os = "zkvm"))]
                                        pub fn from_biguint(biguint: BigUint) -> Self {
                                            Self(biguint_to_limbs(biguint))
                                        }

                                        /// Value of this #struct_name as a BigUint.
                                        #[cfg(not(target_os = "zkvm"))]
                                        pub fn as_biguint(&self) -> BigUint {
                                            BigUint::from_bytes_le(self.as_bytes())
                                        }

                                        /// Modulus N as a BigUint.
                                        #[cfg(not(target_os = "zkvm"))]
                                        pub fn modulus_biguint() -> BigUint {
                                            BigUint::from_bytes_be(&Self::MODULUS)
                                        }

                                        #[inline(always)]
                                        fn add_assign_impl(&mut self, other: &Self) {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                self.0 = biguint_to_limbs(
                                                    (self.as_biguint() + other.as_biguint()) % Self::modulus_biguint(),
                                                );
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                todo!()
                                            }
                                        }

                                        #[inline(always)]
                                        fn sub_assign_impl(&mut self, other: &Self) {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                let modulus = Self::modulus_biguint();
                                                self.0 = biguint_to_limbs(
                                                    (self.as_biguint() + modulus.clone() - other.as_biguint()) % modulus,
                                                );
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                todo!()
                                            }
                                        }

                                        #[inline(always)]
                                        fn mul_assign_impl(&mut self, other: &Self) {
                                            #[cfg(not(target_os = "zkvm"))]
                                            {
                                                self.0 = biguint_to_limbs(
                                                    (self.as_biguint() * other.as_biguint()) % Self::modulus_biguint(),
                                                );
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                todo!()
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
                                                self.0 = biguint_to_limbs((self.as_biguint() * inv) % modulus);
                                            }
                                            #[cfg(target_os = "zkvm")]
                                            {
                                                todo!()
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
                                                let ptr: *mut #struct_name = uninit.as_mut_ptr();
                                                unsafe {
                                                    *ptr = todo!();
                                                    uninit.assume_init()
                                                }
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
                                                todo!()
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
                                                todo!()
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
                                                todo!()
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
                                                todo!()
                                            }
                                        }
                                    }

                                }));

                                moduli.push(modulus_bytes);
                                mod_idx += 1;
                            } else {
                                return syn::Error::new_spanned(
                                    assign.right,
                                    "Right side must be a string literal",
                                )
                                .to_compile_error()
                                .into();
                            }
                        } else {
                            return syn::Error::new_spanned(
                                assign.right,
                                "Right side must be a string literal",
                            )
                            .to_compile_error()
                            .into();
                        }
                    } else {
                        return syn::Error::new_spanned(
                            stmt,
                            "Left side of assignment must be an identifier",
                        )
                        .to_compile_error()
                        .into();
                    }
                } else {
                    return syn::Error::new_spanned(stmt, "Only simple assignments are supported")
                        .to_compile_error()
                        .into();
                }
            }
            _ => {
                return syn::Error::new_spanned(stmt, "Only assignments are supported")
                    .to_compile_error()
                    .into();
            }
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
