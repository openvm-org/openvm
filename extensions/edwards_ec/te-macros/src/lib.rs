extern crate proc_macro;

use openvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ExprPath, LitStr, Token,
};

/// This macro generates the code to setup a Twisted Edwards elliptic curve for a given modular
/// type. Also it places the curve parameters into a special static variable to be later extracted
/// from the ELF and used by the VM. Usage:
/// ```
/// te_declare! {
///     [TODO]
/// }
/// ```
///
/// For this macro to work, you must import the `elliptic_curve` crate and the
/// `openvm_ecc_guest::edwards` crate..
#[proc_macro]
pub fn te_declare(input: TokenStream) -> TokenStream {
    let MacroArgs { items } = parse_macro_input!(input as MacroArgs);

    let mut output = Vec::new();

    let span = proc_macro::Span::call_site();

    for item in items.into_iter() {
        let struct_name = item.name.to_string();
        let struct_name = syn::Ident::new(&struct_name, span.into());
        let struct_path: syn::Path = syn::parse_quote!(#struct_name);
        let mut intmod_type: Option<syn::Path> = None;
        let mut const_a: Option<syn::Expr> = None;
        let mut const_d: Option<syn::Expr> = None;
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
                "a" => {
                    const_a = Some(param.value);
                }
                "d" => {
                    const_d = Some(param.value);
                }
                _ => {
                    panic!("Unknown parameter {}", param.name);
                }
            }
        }

        let intmod_type = intmod_type.expect("mod_type parameter is required");
        let const_a = const_a.expect("constant a coefficient is required");
        let const_d = const_d.expect("constant d coefficient is required");

        macro_rules! create_extern_func {
            ($name:ident) => {
                let $name = syn::Ident::new(
                    &format!(
                        "{}_{}",
                        stringify!($name),
                        struct_path
                            .segments
                            .iter()
                            .map(|x| x.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("_")
                    ),
                    span.into(),
                );
            };
        }
        create_extern_func!(te_add_extern_func);
        create_extern_func!(te_setup_extern_func);

        let group_ops_mod_name = format_ident!("{}_ops", struct_name.to_string().to_lowercase());

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #te_add_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #te_setup_extern_func(uninit: *mut core::ffi::c_void, p1: *const u8, p2: *const u8);
            }

            #[derive(Eq, PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
            #[repr(C)]
            pub struct #struct_name {
                x: #intmod_type,
                y: #intmod_type,
            }

            impl #struct_name {
                const fn identity() -> Self {
                    Self {
                        x: <#intmod_type as openvm_algebra_guest::IntMod>::ZERO,
                        y: <#intmod_type as openvm_algebra_guest::IntMod>::ONE,
                    }
                }
                // Below are wrapper functions for the intrinsic instructions.
                // Should not be called directly.
                #[inline(always)]
                fn add_chip(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        use openvm_algebra_guest::DivUnsafe;

                        let x1y2 = p1.x.clone() * p2.y.clone();
                        let y1x2 = p1.y.clone() * p2.x.clone();
                        let x1x2 = p1.x.clone() * p2.x.clone();
                        let y1y2 = p1.y.clone() * p2.y.clone();
                        let dx1x2y1y2 = <Self as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::CURVE_D * &x1x2 * &y1y2;

                        let x3 = (x1y2 + y1x2).div_unsafe(&<#intmod_type as openvm_algebra_guest::IntMod>::ONE + &dx1x2y1y2);
                        let y3 = (y1y2 - <Self as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::CURVE_A * x1x2).div_unsafe(&<#intmod_type as openvm_algebra_guest::IntMod>::ONE - &dx1x2y1y2);

                        #struct_name { x: x3, y: y3 }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        Self::set_up_once();
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #te_add_extern_func(
                                uninit.as_mut_ptr() as usize,
                                p1 as *const #struct_name as usize,
                                p2 as *const #struct_name as usize
                            )
                        };
                        unsafe { uninit.assume_init() }
                    }
                }

                // Helper function to call the setup instruction on first use
                #[cfg(target_os = "zkvm")]
                #[inline(always)]
                fn set_up_once() {
                    static is_setup: ::openvm_ecc_guest::edwards::once_cell::race::OnceBool = ::openvm_ecc_guest::edwards::once_cell::race::OnceBool::new();
                    is_setup.get_or_init(|| {
                        let modulus_bytes = <<Self as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::Coordinate as openvm_algebra_guest::IntMod>::MODULUS;
                        let mut zero = [0u8; <<Self as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::Coordinate as openvm_algebra_guest::IntMod>::NUM_LIMBS];
                        let curve_a_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<Self as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::CURVE_A);
                        let curve_d_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<Self as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::CURVE_D);
                        let p1 = [modulus_bytes.as_ref(), curve_a_bytes.as_ref()].concat();
                        let p2 = [curve_d_bytes.as_ref(), zero.as_ref()].concat();
                        let mut uninit: core::mem::MaybeUninit<[Self; 2]> = core::mem::MaybeUninit::uninit();

                        unsafe { #te_setup_extern_func(uninit.as_mut_ptr() as *mut core::ffi::c_void, p1.as_ptr(), p2.as_ptr()); }
                        <#intmod_type as openvm_algebra_guest::IntMod>::set_up_once();
                        true
                    });
                }

                #[cfg(not(target_os = "zkvm"))]
                #[inline(always)]
                fn set_up_once() {
                    // No-op for non-ZKVM targets
                }
            }

            impl ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint for #struct_name {
                const CURVE_A: Self::Coordinate = #const_a;
                const CURVE_D: Self::Coordinate = #const_d;

                const IDENTITY: Self = Self::identity();
                type Coordinate = #intmod_type;

                /// SAFETY: assumes that #intmod_type has a memory representation
                /// such that with repr(C), two coordinates are packed contiguously.
                #[inline(always)]
                fn as_le_bytes(&self) -> &[u8] {
                    unsafe { &*core::ptr::slice_from_raw_parts(self as *const Self as *const u8, <#intmod_type as openvm_algebra_guest::IntMod>::NUM_LIMBS * 2) }
                }

                #[inline(always)]
                fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self {
                    Self { x, y }
                }

                #[inline(always)]
                fn x(&self) -> &Self::Coordinate {
                    &self.x
                }

                #[inline(always)]
                fn y(&self) -> &Self::Coordinate {
                    &self.y
                }

                #[inline(always)]
                fn x_mut(&mut self) -> &mut Self::Coordinate {
                    &mut self.x
                }

                #[inline(always)]
                fn y_mut(&mut self) -> &mut Self::Coordinate {
                    &mut self.y
                }

                #[inline(always)]
                fn into_coords(self) -> (Self::Coordinate, Self::Coordinate) {
                    (self.x, self.y)
                }

                #[inline(always)]
                fn add_impl(&self, p2: &Self) -> Self {
                    Self::add_chip(self, p2)
                }
            }

            impl core::ops::Neg for #struct_name {
                type Output = Self;

                fn neg(self) -> Self::Output {
                    #struct_name {
                        x: core::ops::Neg::neg(&self.x),
                        y: self.y,
                    }
                }
            }

            impl core::ops::Neg for &#struct_name {
                type Output = #struct_name;

                fn neg(self) -> #struct_name {
                    #struct_name {
                        x: core::ops::Neg::neg(&self.x),
                        y: self.y.clone(),
                    }
                }
            }

            mod #group_ops_mod_name {
                use ::openvm_ecc_guest::edwards::{Group, edwards::TwistedEdwardsPoint, FromCompressed, impl_te_group_ops, algebra::{IntMod, DivUnsafe, DivAssignUnsafe, ExpBytes}};
                use super::*;

                impl_te_group_ops!(#struct_name, #intmod_type);

                impl FromCompressed<#intmod_type> for #struct_name {
                    fn decompress(y: #intmod_type, rec_id: &u8) -> Option<Self> {
                        use openvm_algebra_guest::{Sqrt, DivUnsafe};
                        let x_squared = (<#intmod_type as openvm_algebra_guest::IntMod>::ONE - &y * &y).div_unsafe(<#struct_name as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::CURVE_A - &<#struct_name as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::CURVE_D * &y * &y);
                        let x = x_squared.sqrt();
                        match x {
                            None => None,
                            Some(x) => {
                                let correct_x = if x.as_le_bytes()[0] & 1 == *rec_id & 1 {
                                    x
                                } else {
                                    -x
                                };
                                // handle the case where x = 0
                                if correct_x.as_le_bytes()[0] & 1 != *rec_id & 1 {
                                    return None;
                                }
                                // In order for sqrt() to return Some, we are guaranteed that x * x == x_squared, which already proves (correct_x, y) is on the curve
                                Some(<#struct_name as ::openvm_ecc_guest::edwards::edwards::TwistedEdwardsPoint>::from_xy_unchecked(correct_x, y))
                            }
                        }
                    }
                }
            }
        });
        output.push(result);
    }

    TokenStream::from_iter(output)
}

struct TeDefine {
    items: Vec<String>,
}

impl Parse for TeDefine {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let items = input.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
        Ok(Self {
            items: items.into_iter().map(|e| e.value()).collect(),
        })
    }
}

#[proc_macro]
pub fn te_init(input: TokenStream) -> TokenStream {
    let TeDefine { items } = parse_macro_input!(input as TeDefine);

    let mut externs = Vec::new();

    let span = proc_macro::Span::call_site();

    for (ec_idx, struct_id) in items.into_iter().enumerate() {
        let add_extern_func =
            syn::Ident::new(&format!("te_add_extern_func_{struct_id}"), span.into());
        let setup_extern_func =
            syn::Ident::new(&format!("te_setup_extern_func_{struct_id}"), span.into());
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #add_extern_func(rd: usize, rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = TE_FUNCT3 as usize,
                    funct7 = TeBaseFunct7::TeAdd as usize + #ec_idx
                        * (TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize),
                    rd = In rd,
                    rs1 = In rs1,
                    rs2 = In rs2
                );
            }

            #[no_mangle]
            extern "C" fn #setup_extern_func(uninit: *mut core::ffi::c_void, p1: *const u8, p2: *const u8) {
                #[cfg(target_os = "zkvm")]
                {

                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_ecc_guest::edwards::OPCODE,
                        funct3 = ::openvm_ecc_guest::edwards::TE_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::edwards::TeBaseFunct7::TeSetup as usize
                            + #ec_idx
                                * (::openvm_ecc_guest::edwards::TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize),
                        rd = In uninit,
                        rs1 = In p1,
                        rs2 = In p2,
                    );
                }
            }
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        #[allow(non_snake_case)]
        #[cfg(target_os = "zkvm")]
        mod openvm_intrinsics_ffi_2_te {
            use ::openvm_ecc_guest::edwards::{OPCODE, TE_FUNCT3, TeBaseFunct7};

            #(#externs)*
        }
    })
}
