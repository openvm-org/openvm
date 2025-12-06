extern crate proc_macro;

use openvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ExprPath, LitStr, Token,
};

/// This macro generates the code to setup the intrinsic curve for a given point type and scalar
/// type. Also it places the curve parameters into a special static variable to be later extracted
/// from the ELF and used by the VM. Usage:
/// ```
/// curve_declare! {
///     [TODO]
/// }
/// ```
///
/// For this macro to work, you must import the `openvm_ecc_guest` crate.
#[proc_macro]
pub fn curve_declare(input: TokenStream) -> TokenStream {
    let MacroArgs { items } = parse_macro_input!(input as MacroArgs);

    let mut output = Vec::new();

    let span = proc_macro::Span::call_site();

    for item in items.into_iter() {
        let struct_name_str = item.name.to_string();
        let struct_name = syn::Ident::new(&struct_name_str, span.into());
        let mut point_type: Option<syn::Path> = None;
        let mut scalar_type: Option<syn::Path> = None;
        for param in item.params {
            match param.name.to_string().as_str() {
                // Note that point_type and scalar_type must be valid types
                "point_type" => {
                    if let syn::Expr::Path(ExprPath { path, .. }) = param.value {
                        point_type = Some(path)
                    } else {
                        return syn::Error::new_spanned(param.value, "Expected a type")
                            .to_compile_error()
                            .into();
                    }
                }
                "scalar_type" => {
                    if let syn::Expr::Path(ExprPath { path, .. }) = param.value {
                        scalar_type = Some(path)
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

        let point_type = point_type.expect("point_type parameter is required");
        let scalar_type = scalar_type.expect("scalar_type parameter is required");

        macro_rules! create_extern_func {
            ($name:ident) => {
                let $name = syn::Ident::new(
                    &format!("{}_{}", stringify!($name), struct_name_str),
                    span.into(),
                );
            };
        }
        create_extern_func!(curve_ec_mul_extern_func);
        create_extern_func!(curve_setup_extern_func);

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #curve_ec_mul_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #curve_setup_extern_func(uninit: *mut core::ffi::c_void, p1: *const u8, p2: *const u8);
            }

            #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
            #[repr(C)]
            pub struct #struct_name;
            #[allow(non_upper_case_globals)]

            impl ::openvm_ecc_guest::weierstrass::IntrinsicCurve for #struct_name {
                type Scalar = #scalar_type;
                type Point = #point_type;

                #[inline(always)]
                fn msm<const CHECK_SETUP: bool>(coeffs: &[Self::Scalar], bases: &[Self::Point]) -> Self::Point
                where
                    for<'a> &'a Self::Point: core::ops::Add<&'a Self::Point, Output = Self::Point>,
                {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        // heuristic
                        if coeffs.len() < 25 {
                            let table = ::openvm_ecc_guest::weierstrass::CachedMulTable::<Self>::new_with_prime_order(bases, 4);
                            table.windowed_mul(coeffs)
                        } else {
                            ::openvm_ecc_guest::msm(coeffs, bases)
                        }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        use core::ops::AddAssign;

                        if CHECK_SETUP {
                            Self::set_up_once();
                        }

                        let mut acc = <Self::Point as openvm_ecc_guest::Group>::IDENTITY;
                        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
                            unsafe {
                                let mut uninit: core::mem::MaybeUninit<Self::Point> =
                                    core::mem::MaybeUninit::uninit();
                                #curve_ec_mul_extern_func(
                                    uninit.as_mut_ptr() as usize,
                                    coeff as *const Self::Scalar as usize,
                                    base as *const Self::Point as usize,
                                );
                                acc.add_assign(&uninit.assume_init());
                            }
                        }
                        acc
                    }
                }

                // Helper function to call the setup instruction on first use
                #[inline(always)]
                #[cfg(target_os = "zkvm")]
                fn set_up_once() {
                    use openvm_algebra_guest::IntMod;

                    static is_setup: ::openvm_ecc_guest::once_cell::race::OnceBool = ::openvm_ecc_guest::once_cell::race::OnceBool::new();

                    is_setup.get_or_init(|| {
                        let scalar_modulus_bytes = <Self::Scalar as openvm_algebra_guest::IntMod>::MODULUS;
                        let point_modulus_bytes = <<Self::Point as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::MODULUS;
                        let p1 = scalar_modulus_bytes.as_ref();
                        let curve_a = <Self::Point as openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_A;
                        let p2 = [point_modulus_bytes.as_ref(), curve_a.as_le_bytes()].concat();
                        let mut uninit: core::mem::MaybeUninit<(Self::Scalar, Self::Point)> = core::mem::MaybeUninit::uninit();

                        unsafe { #curve_setup_extern_func(uninit.as_mut_ptr() as *mut core::ffi::c_void, p1.as_ptr(), p2.as_ptr()); }
                        <Self::Scalar as openvm_algebra_guest::IntMod>::set_up_once();
                        <Self::Point as openvm_ecc_guest::weierstrass::WeierstrassPoint>::set_up_once();
                        true
                    });
                }

                #[inline(always)]
                #[cfg(not(target_os = "zkvm"))]
                fn set_up_once() {
                    // No-op for non-ZKVM targets
                }
            }
        });
        output.push(result);
    }

    TokenStream::from_iter(output)
}

struct CurveDefine {
    items: Vec<String>,
}

impl Parse for CurveDefine {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let items = input.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
        Ok(Self {
            items: items.into_iter().map(|e| e.value()).collect(),
        })
    }
}

#[proc_macro]
pub fn curve_init(input: TokenStream) -> TokenStream {
    let CurveDefine { items } = parse_macro_input!(input as CurveDefine);

    let mut externs = Vec::new();

    let span = proc_macro::Span::call_site();

    for (curve_idx, struct_id) in items.into_iter().enumerate() {
        // Unique identifier shared by curve_declare! and curve_init! used for naming the extern
        // funcs. Currently it's just the struct type name.
        let ec_mul_extern_func = syn::Ident::new(
            &format!("curve_ec_mul_extern_func_{}", struct_id),
            span.into(),
        );
        let setup_extern_func = syn::Ident::new(
            &format!("curve_setup_extern_func_{}", struct_id),
            span.into(),
        );

        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #ec_mul_extern_func(rd: usize, rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = openvm_ecc_guest::OPCODE,
                    funct3 = openvm_ecc_guest::SW_FUNCT3 as usize,
                    funct7 = openvm_ecc_guest::SwBaseFunct7::SwEcMul as usize
                        + #curve_idx
                            * (openvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
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
                        opcode = ::openvm_ecc_guest::OPCODE,
                        funct3 = ::openvm_ecc_guest::SW_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::SwBaseFunct7::SwSetupMul as usize
                            + #curve_idx
                                * (::openvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        rd = In uninit,
                        rs1 = In p1,
                        rs2 = In p2
                    );
                }
            }
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        #[allow(non_snake_case)]
        #[cfg(target_os = "zkvm")]
        mod openvm_intrinsics_ffi_3 {
            #(#externs)*
        }
    })
}
