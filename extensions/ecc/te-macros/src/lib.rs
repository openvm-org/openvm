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
        create_extern_func!(hint_non_qr_extern_func);

        let group_ops_mod_name = format_ident!("{}_ops", struct_name.to_string().to_lowercase());

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #te_add_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #te_hint_decompress_extern_func(rs1: usize, rs2: usize);
                fn #hint_non_qr_extern_func();
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
                use ::openvm_ecc_guest::{edwards::TwistedEdwardsPoint, FromCompressed, DecompressionHint, impl_te_group_ops, algebra::{IntMod, DivUnsafe, DivAssignUnsafe, ExpBytes}};
                use super::*;

                impl_te_group_ops!(#struct_name, #intmod_type);

                impl FromCompressed<#intmod_type> for #struct_name {
                    fn decompress(y: #intmod_type, rec_id: &u8) -> Option<Self> {
                        match Self::honest_host_decompress(&y, rec_id) {
                            // successfully decompressed
                            Some(Some(ret)) => Some(ret),
                            // successfully proved that the point cannot be decompressed
                            Some(None) => None,
                            None => {
                                // host is dishonest, enter infinite loop
                                loop {
                                    openvm::io::println("ERROR: Decompression hint is invalid. Entering infinite loop.");
                                }
                            }
                        }
                    }

                    fn hint_decompress(y: &#intmod_type, rec_id: &u8) -> Option<DecompressionHint<#intmod_type>> {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!()
                        }
                        #[cfg(target_os = "zkvm")]
                        {
                            use openvm::platform as openvm_platform; // needed for hint_buffer_u32!

                            let possible = core::mem::MaybeUninit::<u32>::uninit();
                            let sqrt = core::mem::MaybeUninit::<#intmod_type>::uninit();
                            unsafe {
                                #te_hint_decompress_extern_func(y as *const _ as usize, rec_id as *const u8 as usize);
                                let possible_ptr = possible.as_ptr() as *const u32;
                                openvm_rv32im_guest::hint_store_u32!(possible_ptr);
                                openvm_rv32im_guest::hint_buffer_u32!(sqrt.as_ptr() as *const u8, <#intmod_type as openvm_algebra_guest::IntMod>::NUM_LIMBS / 4);
                                let possible = possible.assume_init();
                                if possible == 0 || possible == 1 {
                                    Some(DecompressionHint { possible: possible == 1, sqrt: sqrt.assume_init() })
                                } else {
                                    None
                                }
                            }
                        }
                    }
                }

                impl #struct_name {
                    // Returns None if the hint is incorrect (i.e. the host is dishonest)
                    // Returns Some(None) if the hint proves that the point cannot be decompressed
                    fn honest_host_decompress(y: &#intmod_type, rec_id: &u8) -> Option<Option<Self>> {
                        let hint = <#struct_name as FromCompressed<#intmod_type>>::hint_decompress(y, rec_id)?;

                        if hint.possible {
                            // ensure x < modulus
                            hint.sqrt.assert_reduced();

                            if hint.sqrt.as_le_bytes()[0] & 1 != *rec_id & 1 {
                                None
                            } else {
                                let ret = <#struct_name as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::from_xy(hint.sqrt, y.clone())?;
                                Some(Some(ret))
                            }
                        } else {
                            // ensure sqrt < modulus
                            hint.sqrt.assert_reduced();

                            let lhs = (&hint.sqrt * &hint.sqrt) * (&<#struct_name as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::CURVE_D * y * y - &<#struct_name as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::CURVE_A);
                            let rhs = y * y - &<#intmod_type as openvm_algebra_guest::IntMod>::ONE;
                            if lhs == rhs * Self::get_non_qr() {
                                Some(None)
                            } else {
                                None
                            }
                        }
                    }

                    // Generate a non quadratic residue in the coordinate field by using a hint
                    fn init_non_qr() -> alloc::boxed::Box<<Self as ::openvm_ecc_guest::edwards::TwistedEdwardsPoint>::Coordinate> {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!();
                        }
                        #[cfg(target_os = "zkvm")]
                        {
                            use openvm_algebra_guest::DivUnsafe;
                            use openvm::platform as openvm_platform; // needed for hint_buffer_u32
                            let mut non_qr_uninit = core::mem::MaybeUninit::<#intmod_type>::uninit();
                            let mut non_qr;
                            unsafe {
                                #hint_non_qr_extern_func();
                                let ptr = non_qr_uninit.as_ptr() as *const u8;
                                openvm_rv32im_guest::hint_buffer_u32!(ptr, <#intmod_type as openvm_algebra_guest::IntMod>::NUM_LIMBS / 4);
                                non_qr = non_qr_uninit.assume_init();
                            }
                            // ensure non_qr < modulus
                            non_qr.assert_reduced();

                            // construct exp = (p-1)/2 as an integer by first constraining exp = (p-1)/2 (mod p) and then exp < p
                            let exp = -<#intmod_type as openvm_algebra_guest::IntMod>::ONE.div_unsafe(#intmod_type::from_const_u8(2));
                            exp.assert_reduced();

                            if non_qr.exp_bytes(true, &exp.to_be_bytes()) != -<#intmod_type as openvm_algebra_guest::IntMod>::ONE
                            {
                                // non_qr is not a non quadratic residue, so host is dishonest
                                loop {
                                    openvm::io::println("ERROR: Non quadratic residue hint is invalid. Entering infinite loop.");
                                }
                            }

                            alloc::boxed::Box::new(non_qr)
                        }
                    }

                    pub fn get_non_qr() -> &'static #intmod_type {
                        static non_qr: ::openvm_ecc_guest::once_cell::race::OnceBox<#intmod_type> = ::openvm_ecc_guest::once_cell::race::OnceBox::new();
                        &non_qr.get_or_init(Self::init_non_qr)
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
        let hint_non_qr_extern_func = syn::Ident::new(
            &format!("hint_non_qr_extern_func_{}", str_path),
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

            #[no_mangle]
            extern "C" fn #hint_non_qr_extern_func() {
                openvm::platform::custom_insn_r!(
                    opcode = TE_OPCODE,
                    funct3 = TE_FUNCT3 as usize,
                    funct7 = TeBaseFunct7::TeHintNonQr as usize + #ec_idx
                        * (TeBaseFunct7::TWISTED_EDWARDS_MAX_KINDS as usize),
                    rd = Const "x0",
                    rs1 = Const "x0",
                    rs2 = Const "x0"
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
