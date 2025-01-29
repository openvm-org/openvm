#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use openvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, ExprPath, Path, Token,
};

/// This macro generates the code to setup a Twisted Edwards elliptic curve for a given modular type. Also it places the curve parameters into a special static variable to be later extracted from the ELF and used by the VM.
/// Usage:
/// ```
/// te_declare! {
///     [TODO]
/// }
/// ```
///
/// For this macro to work, you must import the `elliptic_curve` crate and the `openvm_ecc_guest` crate..
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
        create_extern_func!(te_hint_decompress_extern_func);

        let group_ops_mod_name = format_ident!("{}_ops", struct_name.to_string().to_lowercase());

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #te_add_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #te_hint_decompress_extern_func(rs1: usize, rs2: usize);
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
                        let dx1x2y1y2 = <Self as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::CURVE_D * &x1x2 * &y1y2;

                        let x3 = (x1y2 + y1x2).div_unsafe(&<#intmod_type as openvm_algebra_guest::IntMod>::ONE + &dx1x2y1y2);
                        let y3 = (y1y2 - <Self as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::CURVE_A * x1x2).div_unsafe(&<#intmod_type as openvm_algebra_guest::IntMod>::ONE - &dx1x2y1y2);

                        #struct_name { x: x3, y: y3 }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
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
            }

            impl ::openvm_ecc_guest::edwards::TwistedEdwardsPoint for #struct_name {
                const CURVE_A: Self::Coordinate = #const_a;
                const CURVE_D: Self::Coordinate = #const_d;

                const IDENTITY: Self = Self::identity();
                type Coordinate = #intmod_type;

                /// SAFETY: assumes that #intmod_type has a memory representation
                /// such that with repr(C), two coordinates are packed contiguously.
                fn as_le_bytes(&self) -> &[u8] {
                    unsafe { &*core::ptr::slice_from_raw_parts(self as *const Self as *const u8, <#intmod_type as openvm_algebra_guest::IntMod>::NUM_LIMBS * 2) }
                }

                fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self {
                    Self { x, y }
                }

                fn x(&self) -> &Self::Coordinate {
                    &self.x
                }

                fn y(&self) -> &Self::Coordinate {
                    &self.y
                }

                fn x_mut(&mut self) -> &mut Self::Coordinate {
                    &mut self.x
                }

                fn y_mut(&mut self) -> &mut Self::Coordinate {
                    &mut self.y
                }

                fn into_coords(self) -> (Self::Coordinate, Self::Coordinate) {
                    (self.x, self.y)
                }

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
                use ::openvm_ecc_guest::{edwards::TwistedEdwardsPoint, FromCompressed, impl_te_group_ops};
                use super::*;

                impl_te_group_ops!(#struct_name, #intmod_type);

                impl FromCompressed<#intmod_type> for #struct_name {
                    fn decompress(y: #intmod_type, rec_id: &u8) -> Self {
                        let x = <#struct_name as FromCompressed<#intmod_type>>::hint_decompress(&y, rec_id);
                        // Must assert unique so we can check the parity
                        x.assert_unique();
                        assert_eq!(x.as_le_bytes()[0] & 1, *rec_id & 1);
                        <#struct_name as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::from_xy(x, y).expect("decompressed point not on curve")
                    }

                    fn hint_decompress(y: &#intmod_type, rec_id: &u8) -> #intmod_type {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!()
                        }
                        #[cfg(target_os = "zkvm")]
                        {
                            use openvm::platform as openvm_platform; // needed for hint_buffer_u32!

                            let x = core::mem::MaybeUninit::<#intmod_type>::uninit();
                            unsafe {
                                #te_hint_decompress_extern_func(y as *const _ as usize, rec_id as *const u8 as usize);
                                let ptr = x.as_ptr() as *const u8;
                                openvm_rv32im_guest::hint_buffer_u32!(ptr, <#intmod_type as openvm_algebra_guest::IntMod>::NUM_LIMBS / 4);
                                x.assume_init()
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
    items: Vec<Path>,
}

impl Parse for TeDefine {
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
pub fn te_init(input: TokenStream) -> TokenStream {
    let TeDefine { items } = parse_macro_input!(input as TeDefine);

    let mut externs = Vec::new();
    let mut setups = Vec::new();
    let mut setup_all_te_curves = Vec::new();

    let span = proc_macro::Span::call_site();

    for (ec_idx, item) in items.into_iter().enumerate() {
        let str_path = item
            .segments
            .iter()
            .map(|x| x.ident.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let add_extern_func =
            syn::Ident::new(&format!("te_add_extern_func_{}", str_path), span.into());
        let te_hint_decompress_extern_func = syn::Ident::new(
            &format!("te_hint_decompress_extern_func_{}", str_path),
            span.into(),
        );
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #add_extern_func(rd: usize, rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = TE_OPCODE,
                    funct3 = TE_FUNCT3 as usize,
                    funct7 = TeBaseFunct7::TeAdd as usize + #ec_idx
                        * (TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize),
                    rd = In rd,
                    rs1 = In rs1,
                    rs2 = In rs2
                );
            }

            #[no_mangle]
            extern "C" fn #te_hint_decompress_extern_func(rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = TE_OPCODE,
                    funct3 = TE_FUNCT3 as usize,
                    funct7 = TeBaseFunct7::TeHintDecompress as usize + #ec_idx
                        * (TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize),
                    rd = Const "x0",
                    rs1 = In rs1,
                    rs2 = In rs2
                );
            }
        });

        let setup_function = syn::Ident::new(&format!("setup_te_{}", str_path), span.into());
        setups.push(quote::quote_spanned! { span.into() =>

            #[allow(non_snake_case)]
            pub fn #setup_function() {
                #[cfg(target_os = "zkvm")]
                {
                    let modulus_bytes = <<#item as openvm_ecc_guest::edwards::TwistedEdwardsPoint>::Coordinate as openvm_algebra_guest::IntMod>::MODULUS;
                    let mut zero = [0u8; <<#item as openvm_ecc_guest::edwards::TwistedEdwardsPoint>::Coordinate as openvm_algebra_guest::IntMod>::NUM_LIMBS];
                    let curve_a_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<#item as openvm_ecc_guest::edwards::TwistedEdwardsPoint>::CURVE_A);
                    let curve_d_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<#item as openvm_ecc_guest::edwards::TwistedEdwardsPoint>::CURVE_D);
                    let p1 = [modulus_bytes.as_ref(), curve_a_bytes.as_ref()].concat();
                    let p2 = [curve_d_bytes.as_ref(), zero.as_ref()].concat();
                    let mut uninit: core::mem::MaybeUninit<[#item; 2]> = core::mem::MaybeUninit::uninit();
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_ecc_guest::TE_OPCODE,
                        funct3 = ::openvm_ecc_guest::TE_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::TeBaseFunct7::TeSetup as usize
                            + #ec_idx
                                * (::openvm_ecc_guest::TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize),
                        rd = In uninit.as_mut_ptr(),
                        rs1 = In p1.as_ptr(),
                        rs2 = In p2.as_ptr(),
                    );
                }
            }
        });

        setup_all_te_curves.push(quote::quote_spanned! { span.into() =>
            #setup_function();
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        #[cfg(target_os = "zkvm")]
        mod openvm_intrinsics_ffi_2_te {
            use ::openvm_ecc_guest::{TE_OPCODE, TE_FUNCT3, TeBaseFunct7};

            #(#externs)*
        }
        #(#setups)*
        pub fn setup_all_te_curves() {
            #(#setup_all_te_curves)*
        }
    })
}
