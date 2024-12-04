#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use axvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, ExprPath, Path, Token,
};

/// TODO: Add documentation
#[proc_macro]
pub fn complex_declare(input: TokenStream) -> TokenStream {
    let MacroArgs { items } = parse_macro_input!(input as MacroArgs);

    let mut output = Vec::new();

    let span = proc_macro::Span::call_site();

    for item in items.into_iter() {
        let struct_name = item.name.to_string();
        let struct_name = syn::Ident::new(&struct_name, span.into());
        let mut intmod_type: Option<syn::Path> = None;
        for param in item.params {
            match param.name.to_string().as_str() {
                "mod_type" => {
                    if let syn::Expr::Path(ExprPath { path, .. }) = param.value {
                        intmod_type = Some(path)
                    } else {
                        return syn::Error::new_spanned(param.value, "Expected a type")
                            .to_compile_error()
                            .into();
                    }
                }
                _ => {
                    panic!("Unknown parameter {}", param.name);
                }
            }
        }

        let intmod_type = intmod_type.expect("mod_type parameter is required");

        macro_rules! create_extern_func {
            ($name:ident) => {
                let $name = syn::Ident::new(
                    &format!("{}_{}", stringify!($name), struct_name),
                    span.into(),
                );
            };
        }
        create_extern_func!(complex_add_extern_func);
        create_extern_func!(complex_sub_extern_func);
        create_extern_func!(complex_mul_extern_func);
        create_extern_func!(complex_div_extern_func);

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #complex_add_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #complex_sub_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #complex_mul_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #complex_div_extern_func(rd: usize, rs1: usize, rs2: usize);
            }


            /// Quadratic extension field of `#intmod_type` with irreducible polynomial `X^2 + 1`.
            /// Elements are represented as `c0 + c1 * u` where `u^2 = -1`.
            ///
            /// Memory alignment follows alignment of `#intmod_type`.
            /// Memory layout is concatenation of `c0` and `c1`.
            #[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
            #[repr(C)]
            pub struct #struct_name {
                /// Real coordinate
                pub c0: #intmod_type,
                /// Imaginary coordinate
                pub c1: #intmod_type,
            }

            impl #struct_name {
                pub const fn new(c0: #intmod_type, c1: #intmod_type) -> Self {
                    Self { c0, c1 }
                }
            }

            impl #struct_name {
                // Zero element (i.e. additive identity)
                pub const ZERO: Self = Self::new(<#intmod_type as axvm_algebra_guest::IntMod>::ZERO, <#intmod_type as axvm_algebra_guest::IntMod>::ZERO);

                // One element (i.e. multiplicative identity)
                pub const ONE: Self = Self::new(<#intmod_type as axvm_algebra_guest::IntMod>::ONE, <#intmod_type as axvm_algebra_guest::IntMod>::ZERO);

                pub fn neg_assign(&mut self) {
                    self.c0.neg_assign();
                    self.c1.neg_assign();
                }

                /// Implementation of AddAssign.
                #[inline(always)]
                fn add_assign_impl(&mut self, other: &Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        self.c0 += &other.c0;
                        self.c1 += &other.c1;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #complex_add_extern_func(
                                self as *mut Self as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                    }
                }

                /// Implementation of SubAssign.
                #[inline(always)]
                fn sub_assign_impl(&mut self, other: &Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        self.c0 -= &other.c0;
                        self.c1 -= &other.c1;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #complex_sub_extern_func(
                                self as *mut Self as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                    }
                }

                /// Implementation of MulAssign.
                #[inline(always)]
                fn mul_assign_impl(&mut self, other: &Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let (c0, c1) = (&self.c0, &self.c1);
                        let (d0, d1) = (&other.c0, &other.c1);
                        *self = Self::new(
                            c0.clone() * d0 - c1.clone() * d1,
                            c0.clone() * d1 + c1.clone() * d0,
                        );
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #complex_mul_extern_func(
                                self as *mut Self as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                    }
                }

                /// Implementation of DivAssignUnsafe.
                #[inline(always)]
                fn div_assign_unsafe_impl(&mut self, other: &Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let (c0, c1) = (&self.c0, &self.c1);
                        let (d0, d1) = (&other.c0, &other.c1);
                        let denom = <#intmod_type as axvm_algebra_guest::IntMod>::ONE.div_unsafe(d0.square() + d1.square());
                        *self = Self::new(
                            denom.clone() * (c0.clone() * d0 + c1.clone() * d1),
                            denom * &(c1.clone() * d0 - c0.clone() * d1),
                        );
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #complex_div_extern_func(
                                self as *mut Self as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                    }
                }

                /// Implementation of Add that doesn't cause zkvm to use an additional store.
                fn add_refs_impl(&self, other: &Self) -> Self {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let mut res = self.clone();
                        res.add_assign_impl(other);
                        res
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        let mut uninit: core::mem::MaybeUninit<Self> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #complex_add_extern_func(
                                uninit.as_mut_ptr() as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                        unsafe { uninit.assume_init() }
                    }
                }

                /// Implementation of Sub that doesn't cause zkvm to use an additional store.
                #[inline(always)]
                fn sub_refs_impl(&self, other: &Self) -> Self {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let mut res = self.clone();
                        res.sub_assign_impl(other);
                        res
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        let mut uninit: core::mem::MaybeUninit<Self> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #complex_sub_extern_func(
                                uninit.as_mut_ptr() as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                        unsafe { uninit.assume_init() }
                    }
                }

                /// Implementation of Mul that doesn't cause zkvm to use an additional store.
                ///
                /// SAFETY: dst_ptr must be pointer for `&mut Self`.
                /// It will only be written to at the end of the function.
                #[inline(always)]
                unsafe fn mul_refs_impl(&self, other: &Self, dst_ptr: *mut Self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let mut res = self.clone();
                        res.mul_assign_impl(other);
                        let dst = unsafe { &mut *dst_ptr };
                        *dst = res;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #complex_mul_extern_func(
                                dst_ptr as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                    }
                }

                /// Implementation of DivUnsafe that doesn't cause zkvm to use an additional store.
                #[inline(always)]
                fn div_unsafe_refs_impl(&self, other: &Self) -> Self {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let mut res = self.clone();
                        res.div_assign_unsafe_impl(other);
                        res
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        let mut uninit: core::mem::MaybeUninit<Self> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #complex_div_extern_func(
                                uninit.as_mut_ptr() as usize,
                                self as *const Self as usize,
                                other as *const Self as usize
                            );
                        }
                        unsafe { uninit.assume_init() }
                    }
                }
            }

            impl axvm_algebra_guest::field::ComplexConjugate for #struct_name {
                fn conjugate(self) -> Self {
                    Self {
                        c0: self.c0,
                        c1: -self.c1,
                    }
                }

                fn conjugate_assign(&mut self) {
                    self.c1.neg_assign();
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

            impl<'a> core::ops::Mul<&'a #struct_name> for &'a #struct_name {
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

            impl<'a> axvm_algebra_guest::DivAssignUnsafe<&'a #struct_name> for #struct_name {
                #[inline(always)]
                fn div_assign_unsafe(&mut self, other: &'a #struct_name) {
                    self.div_assign_unsafe_impl(other);
                }
            }

            impl axvm_algebra_guest::DivAssignUnsafe for #struct_name {
                #[inline(always)]
                fn div_assign_unsafe(&mut self, other: Self) {
                    self.div_assign_unsafe_impl(&other);
                }
            }

            impl axvm_algebra_guest::DivUnsafe for #struct_name {
                type Output = Self;
                #[inline(always)]
                fn div_unsafe(mut self, other: Self) -> Self::Output {
                    self = self.div_unsafe_refs_impl(&other);
                    self
                }
            }

            impl<'a> axvm_algebra_guest::DivUnsafe<&'a #struct_name> for #struct_name {
                type Output = Self;
                #[inline(always)]
                fn div_unsafe(mut self, other: &'a #struct_name) -> Self::Output {
                    self = self.div_unsafe_refs_impl(other);
                    self
                }
            }

            impl<'a> axvm_algebra_guest::DivUnsafe<&'a #struct_name> for &#struct_name {
                type Output = #struct_name;
                #[inline(always)]
                fn div_unsafe(self, other: &'a #struct_name) -> Self::Output {
                    self.div_unsafe_refs_impl(other)
                }
            }

            impl<'a> core::iter::Sum<&'a #struct_name> for #struct_name {
                fn sum<I: core::iter::Iterator<Item = &'a #struct_name>>(iter: I) -> Self {
                    iter.fold(Self::ZERO, |acc, x| &acc + x)
                }
            }

            impl core::iter::Sum for #struct_name {
                fn sum<I: core::iter::Iterator<Item = Self>>(iter: I) -> Self {
                    iter.fold(Self::ZERO, |acc, x| &acc + &x)
                }
            }

            impl<'a> core::iter::Product<&'a #struct_name> for #struct_name {
                fn product<I: core::iter::Iterator<Item = &'a #struct_name>>(iter: I) -> Self {
                    iter.fold(Self::ONE, |acc, x| &acc * x)
                }
            }

            impl core::iter::Product for #struct_name {
                fn product<I: core::iter::Iterator<Item = Self>>(iter: I) -> Self {
                    iter.fold(Self::ONE, |acc, x| &acc * &x)
                }
            }

            impl core::ops::Neg for #struct_name {
                type Output = #struct_name;
                fn neg(self) -> Self::Output {
                    Self::ZERO - &self
                }
            }

            impl core::ops::Neg for &#struct_name {
                type Output = #struct_name;
                fn neg(self) -> Self::Output {
                    #struct_name::ZERO - self
                }
            }

            impl core::fmt::Debug for #struct_name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    write!(f, "{:?} + {:?} * u", self.c0, self.c1)
                }
            }
        });
        output.push(result);
    }

    TokenStream::from_iter(output)
}

struct ComplexDefine {
    items: Vec<Path>,
}

impl Parse for ComplexDefine {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let items = input.parse_terminated(<Expr as Parse>::parse, Token![,])?;
        Ok(Self {
            items: items
                .into_iter()
                .map(|e| {
                    if let Expr::Path(p) = e {
                        p.path
                    } else {
                        panic!("expected path");
                    }
                })
                .collect(),
        })
    }
}

#[proc_macro]
pub fn complex_init(input: TokenStream) -> TokenStream {
    let ComplexDefine { items } = parse_macro_input!(input as ComplexDefine);

    let mut externs = Vec::new();
    let mut setups = Vec::new();
    let mut setup_all_complex_extensions = Vec::new();

    let span = proc_macro::Span::call_site();

    for (complex_idx, item) in items.into_iter().enumerate() {
        let str_path = item
            .segments
            .iter()
            .map(|x| x.ident.to_string())
            .collect::<Vec<_>>()
            .join("_");

        println!("[init] complex #{} = {}", complex_idx, str_path);

        for op_type in ["add", "sub", "mul", "div"] {
            let func_name = syn::Ident::new(
                &format!("complex_{}_extern_func_{}", op_type, str_path),
                span.into(),
            );
            let mut chars = op_type.chars().collect::<Vec<_>>();
            chars[0] = chars[0].to_ascii_uppercase();
            let local_opcode = syn::Ident::new(&chars.iter().collect::<String>(), span.into());
            externs.push(quote::quote_spanned! { span.into() =>
                #[no_mangle]
                extern "C" fn #func_name(rd: usize, rs1: usize, rs2: usize) {
                    axvm_platform::custom_insn_r!(
                        axvm_algebra_guest::OPCODE,
                        axvm_algebra_guest::COMPLEX_EXT_FIELD_FUNCT3,
                        axvm_algebra_guest::ComplexExtFieldBaseFunct7::#local_opcode as usize
                            + #complex_idx * (axvm_algebra_guest::ComplexExtFieldBaseFunct7::COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                        rd,
                        rs1,
                        rs2
                    )
                }
            });
        }

        let setup_function =
            syn::Ident::new(&format!("setup_complex_{}", complex_idx), span.into());

        setup_all_complex_extensions.push(quote::quote_spanned! { span.into() =>
            #setup_function();
        });
        setups.push(quote::quote_spanned! { span.into() =>
            // Inline never is necessary, as otherwise if compiler thinks it's ok to reorder, the setup result might overwrite some register in use.
            #[inline(never)]
            #[allow(non_snake_case)]
            pub fn #setup_function() {
                #[cfg(target_os = "zkvm")]
                {
                    let modulus_bytes = &axvm_intrinsics_meta_do_not_type_this_by_yourself::modular_limbs_list[axvm_intrinsics_meta_do_not_type_this_by_yourself::limb_list_borders[#complex_idx]..axvm_intrinsics_meta_do_not_type_this_by_yourself::limb_list_borders[#complex_idx + 1]];

                    // We are going to use the numeric representation of the `rs2` register to distinguish the chip to setup.
                    // The transpiler will transform this instruction, based on whether `rs2` is `x0` or `x1`, into a `SETUP_ADDSUB` or `SETUP_MULDIV` instruction.
                    let mut uninit: core::mem::MaybeUninit<[u8; axvm_intrinsics_meta_do_not_type_this_by_yourself::limb_list_borders[#complex_idx + 1] - axvm_intrinsics_meta_do_not_type_this_by_yourself::limb_list_borders[#complex_idx]]> = core::mem::MaybeUninit::uninit();
                    axvm_platform::custom_insn_r!(
                        ::axvm_algebra_guest::OPCODE,
                        ::axvm_algebra_guest::COMPLEX_EXT_FIELD_FUNCT3,
                        ::axvm_algebra_guest::ComplexExtFieldBaseFunct7::Setup as usize
                            + #complex_idx
                                * (::axvm_algebra_guest::ComplexExtFieldBaseFunct7::COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                        uninit.as_mut_ptr(),
                        modulus_bytes.as_ptr(),
                        "x0" // will be parsed as 0 and therefore transpiled to SETUP_ADDMOD
                    );
                    axvm_platform::custom_insn_r!(
                        ::axvm_algebra_guest::OPCODE,
                        ::axvm_algebra_guest::COMPLEX_EXT_FIELD_FUNCT3,
                        ::axvm_algebra_guest::ComplexExtFieldBaseFunct7::Setup as usize
                            + #complex_idx
                                * (::axvm_algebra_guest::ComplexExtFieldBaseFunct7::COMPLEX_EXT_FIELD_MAX_KINDS as usize),
                        uninit.as_mut_ptr(),
                        modulus_bytes.as_ptr(),
                        "x1" // will be parsed as 1 and therefore transpiled to SETUP_MULDIV
                    );
                }
            }
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        // #(#axiom_section)*
        #[cfg(target_os = "zkvm")]
        mod axvm_intrinsics_ffi_complex {
            #(#externs)*
        }
        #(#setups)*
        pub fn setup_all_complex_extensions() {
            #(#setup_all_complex_extensions)*
        }
    })
}

#[proc_macro]
pub fn complex_impl_field(input: TokenStream) -> TokenStream {
    let ComplexDefine { items } = parse_macro_input!(input as ComplexDefine);

    let mut output = Vec::new();

    let span = proc_macro::Span::call_site();

    for item in items.into_iter() {
        let str_path = item
            .segments
            .iter()
            .map(|x| x.ident.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let struct_name = syn::Ident::new(&str_path, span.into());

        output.push(quote::quote_spanned! { span.into() =>
            impl axvm_algebra_guest::field::Field for #struct_name {
                type SelfRef<'a>
                    = &'a Self
                where
                    Self: 'a;

                const ZERO: Self = Self::ZERO;
                const ONE: Self = Self::ONE;

                fn double_assign(&mut self) {
                    axvm_algebra_guest::field::Field::double_assign(&mut self.c0);
                    axvm_algebra_guest::field::Field::double_assign(&mut self.c1);
                }

                fn square_assign(&mut self) {
                    unsafe {
                        self.mul_refs_impl(self, self as *const Self as *mut Self);
                    }
                }
            }
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        #(#output)*
    })
}
