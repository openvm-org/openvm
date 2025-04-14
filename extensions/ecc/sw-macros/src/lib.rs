extern crate proc_macro;

use openvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, ExprPath, Path, Token,
};

/// This macro generates the code to setup the elliptic curve for a given modular type. Also it
/// places the curve parameters into a special static variable to be later extracted from the ELF
/// and used by the VM. Usage:
/// ```
/// sw_declare! {
///     [TODO]
/// }
/// ```
///
/// For this macro to work, you must import the `elliptic_curve` crate and the `openvm_ecc_guest`
/// crate.
#[proc_macro]
pub fn sw_declare(input: TokenStream) -> TokenStream {
    let MacroArgs { items } = parse_macro_input!(input as MacroArgs);

    let mut output = Vec::new();

    let span = proc_macro::Span::call_site();

    for item in items.into_iter() {
        let struct_name = item.name.to_string();
        let struct_name = syn::Ident::new(&struct_name, span.into());
        let struct_path: syn::Path = syn::parse_quote!(#struct_name);
        let mut intmod_type: Option<syn::Path> = None;
        let mut const_a: Option<syn::Expr> = None;
        let mut const_b: Option<syn::Expr> = None;
        for param in item.params {
            match param.name.to_string().as_str() {
                // Note that mod_type must have NUM_LIMBS divisible by 4
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
                    // We currently leave it to the compiler to check if the expression is actually
                    // a constant
                    const_a = Some(param.value);
                }
                "b" => {
                    // We currently leave it to the compiler to check if the expression is actually
                    // a constant
                    const_b = Some(param.value);
                }
                _ => {
                    panic!("Unknown parameter {}", param.name);
                }
            }
        }

        let intmod_type = intmod_type.expect("mod_type parameter is required");
        // const_a is optional, default to 0
        let const_a = const_a
            .unwrap_or(syn::parse_quote!(<#intmod_type as openvm_algebra_guest::IntMod>::ZERO));
        let const_b = const_b.expect("constant b coefficient is required");

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
        create_extern_func!(sw_add_ne_extern_func);
        create_extern_func!(sw_double_extern_func);
        create_extern_func!(hint_decompress_extern_func);
        create_extern_func!(hint_non_qr_extern_func);

        let group_ops_mod_name = format_ident!("{}_ops", struct_name.to_string().to_lowercase());

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #sw_add_ne_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #sw_double_extern_func(rd: usize, rs1: usize);
                fn #hint_decompress_extern_func(rs1: usize, rs2: usize);
                fn #hint_non_qr_extern_func();
            }

            #[derive(Eq, PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
            #[repr(C)]
            pub struct #struct_name {
                x: #intmod_type,
                y: #intmod_type,
            }
            #[allow(non_upper_case_globals)]

            impl #struct_name {
                const fn identity() -> Self {
                    Self {
                        x: <#intmod_type as openvm_algebra_guest::IntMod>::ZERO,
                        y: <#intmod_type as openvm_algebra_guest::IntMod>::ZERO,
                    }
                }
                // Below are wrapper functions for the intrinsic instructions.
                // Should not be called directly.
                #[inline(always)]
                fn add_ne(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        use openvm_algebra_guest::DivUnsafe;
                        let lambda = (&p2.y - &p1.y).div_unsafe(&p2.x - &p1.x);
                        let x3 = &lambda * &lambda - &p1.x - &p2.x;
                        let y3 = &lambda * &(&p1.x - &x3) - &p1.y;
                        #struct_name { x: x3, y: y3 }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #sw_add_ne_extern_func(
                                uninit.as_mut_ptr() as usize,
                                p1 as *const #struct_name as usize,
                                p2 as *const #struct_name as usize
                            )
                        };
                        unsafe { uninit.assume_init() }
                    }
                }

                #[inline(always)]
                fn add_ne_assign(&mut self, p2: &#struct_name) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        use openvm_algebra_guest::DivUnsafe;
                        let lambda = (&p2.y - &self.y).div_unsafe(&p2.x - &self.x);
                        let x3 = &lambda * &lambda - &self.x - &p2.x;
                        let y3 = &lambda * &(&self.x - &x3) - &self.y;
                        self.x = x3;
                        self.y = y3;
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #sw_add_ne_extern_func(
                                self as *mut #struct_name as usize,
                                self as *const #struct_name as usize,
                                p2 as *const #struct_name as usize
                            )
                        };
                    }
                }

                /// Assumes that `p` is not identity.
                #[inline(always)]
                fn double_impl(p: &#struct_name) -> #struct_name {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        use openvm_algebra_guest::DivUnsafe;
                        let curve_a: #intmod_type = #const_a;
                        let two = #intmod_type::from_u8(2);
                        let lambda = (&p.x * &p.x * #intmod_type::from_u8(3) + &curve_a).div_unsafe(&p.y * &two);
                        let x3 = &lambda * &lambda - &p.x * &two;
                        let y3 = &lambda * &(&p.x - &x3) - &p.y;
                        #struct_name { x: x3, y: y3 }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #sw_double_extern_func(
                                uninit.as_mut_ptr() as usize,
                                p as *const #struct_name as usize,
                            )
                        };
                        unsafe { uninit.assume_init() }
                    }
                }

                #[inline(always)]
                fn double_assign_impl(&mut self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        *self = Self::double_impl(self);
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        unsafe {
                            #sw_double_extern_func(
                                self as *mut #struct_name as usize,
                                self as *const #struct_name as usize
                            )
                        };
                    }
                }

            }

            impl ::openvm_ecc_guest::weierstrass::WeierstrassPoint for #struct_name {
                const CURVE_A: #intmod_type = #const_a;
                const CURVE_B: #intmod_type = #const_b;
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

                fn add_ne_nonidentity(&self, p2: &Self) -> Self {
                    Self::add_ne(self, p2)
                }

                fn add_ne_assign_nonidentity(&mut self, p2: &Self) {
                    Self::add_ne_assign(self, p2);
                }

                fn sub_ne_nonidentity(&self, p2: &Self) -> Self {
                    Self::add_ne(self, &p2.clone().neg())
                }

                fn sub_ne_assign_nonidentity(&mut self, p2: &Self) {
                    Self::add_ne_assign(self, &p2.clone().neg());
                }

                fn double_nonidentity(&self) -> Self {
                    Self::double_impl(self)
                }

                fn double_assign_nonidentity(&mut self) {
                    Self::double_assign_impl(self);
                }
            }

            impl core::ops::Neg for #struct_name {
                type Output = Self;

                fn neg(self) -> Self::Output {
                    #struct_name {
                        x: self.x,
                        y: -self.y,
                    }
                }
            }

            impl core::ops::Neg for &#struct_name {
                type Output = #struct_name;

                fn neg(self) -> #struct_name {
                    #struct_name {
                        x: self.x.clone(),
                        y: core::ops::Neg::neg(&self.y),
                    }
                }
            }

            mod #group_ops_mod_name {
                use ::openvm_ecc_guest::{weierstrass::{WeierstrassPoint, FromCompressed, DecompressionHint}, impl_sw_group_ops, algebra::{IntMod, DivUnsafe, DivAssignUnsafe, ExpBytes}};
                use super::*;

                impl_sw_group_ops!(#struct_name, #intmod_type);

                impl FromCompressed<#intmod_type> for #struct_name {
                    fn decompress(x: #intmod_type, rec_id: &u8) -> Option<Self> {
                        match Self::honest_host_decompress(&x, rec_id) {
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

                    fn hint_decompress(x: &#intmod_type, rec_id: &u8) -> Option<DecompressionHint<#intmod_type>> {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!()
                        }
                        #[cfg(target_os = "zkvm")]
                        {
                            use openvm::platform as openvm_platform; // needed for hint_store_u32!

                            let possible = core::mem::MaybeUninit::<u32>::uninit();
                            let sqrt = core::mem::MaybeUninit::<#intmod_type>::uninit();
                            unsafe {
                                #hint_decompress_extern_func(x as *const _ as usize, rec_id as *const u8 as usize);
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
                    fn honest_host_decompress(x: &#intmod_type, rec_id: &u8) -> Option<Option<Self>> {
                        let hint = <#struct_name as FromCompressed<#intmod_type>>::hint_decompress(x, rec_id)?;

                        if hint.possible {
                            // ensure y < modulus
                            hint.sqrt.assert_reduced();

                            if hint.sqrt.as_le_bytes()[0] & 1 != *rec_id & 1 {
                                None
                            } else {
                                let ret = <#struct_name as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::from_xy_nonidentity(x.clone(), hint.sqrt)?;
                                Some(Some(ret))
                            }
                        } else {
                            // ensure sqrt < modulus
                            hint.sqrt.assert_reduced();

                            let alpha = (x * x * x) + (x * &<#struct_name as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_A) + &<#struct_name as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_B;
                            if &hint.sqrt * &hint.sqrt == alpha * Self::get_non_qr() {
                                Some(None)
                            } else {
                                None
                            }
                        }
                    }

                    // Generate a non quadratic residue in the coordinate field by using a hint
                    fn init_non_qr() -> alloc::boxed::Box<<Self as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate> {
                        #[cfg(not(target_os = "zkvm"))]
                        {
                            unimplemented!();
                        }
                        #[cfg(target_os = "zkvm")]
                        {
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

struct SwDefine {
    items: Vec<Path>,
}

impl Parse for SwDefine {
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
pub fn sw_init(input: TokenStream) -> TokenStream {
    let SwDefine { items } = parse_macro_input!(input as SwDefine);

    let mut externs = Vec::new();
    let mut setups = Vec::new();
    let mut setup_all_curves = Vec::new();

    let span = proc_macro::Span::call_site();

    for (ec_idx, item) in items.into_iter().enumerate() {
        let str_path = item
            .segments
            .iter()
            .map(|x| x.ident.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let add_ne_extern_func =
            syn::Ident::new(&format!("sw_add_ne_extern_func_{}", str_path), span.into());
        let double_extern_func =
            syn::Ident::new(&format!("sw_double_extern_func_{}", str_path), span.into());
        let hint_decompress_extern_func = syn::Ident::new(
            &format!("hint_decompress_extern_func_{}", str_path),
            span.into(),
        );
        let hint_non_qr_extern_func = syn::Ident::new(
            &format!("hint_non_qr_extern_func_{}", str_path),
            span.into(),
        );
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #add_ne_extern_func(rd: usize, rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = SW_FUNCT3 as usize,
                    funct7 = SwBaseFunct7::SwAddNe as usize + #ec_idx
                        * (SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd = In rd,
                    rs1 = In rs1,
                    rs2 = In rs2
                );
            }

            #[no_mangle]
            extern "C" fn #double_extern_func(rd: usize, rs1: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = SW_FUNCT3 as usize,
                    funct7 = SwBaseFunct7::SwDouble as usize + #ec_idx
                        * (SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd = In rd,
                    rs1 = In rs1,
                    rs2 = Const "x0"
                );
            }

            #[no_mangle]
            extern "C" fn #hint_decompress_extern_func(rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = SW_FUNCT3 as usize,
                    funct7 = SwBaseFunct7::HintDecompress as usize + #ec_idx
                        * (SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd = Const "x0",
                    rs1 = In rs1,
                    rs2 = In rs2
                );
            }

            #[no_mangle]
            extern "C" fn #hint_non_qr_extern_func() {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = SW_FUNCT3 as usize,
                    funct7 = SwBaseFunct7::HintNonQr as usize + #ec_idx
                        * (SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd = Const "x0",
                    rs1 = Const "x0",
                    rs2 = Const "x0"
                );
            }
        });

        let setup_function = syn::Ident::new(&format!("setup_sw_{}", str_path), span.into());
        setups.push(quote::quote_spanned! { span.into() =>
            #[allow(non_snake_case)]
            pub fn #setup_function() {
                #[cfg(target_os = "zkvm")]
                {
                    // p1 is (x1, y1), and x1 must be the modulus.
                    // y1 can be anything for SetupEcAdd, but must equal `a` for SetupEcDouble
                    let modulus_bytes = <<#item as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::MODULUS;
                    let mut one = [0u8; <<#item as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::NUM_LIMBS];
                    one[0] = 1;
                    let curve_a_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<#item as openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_A);
                    // p1 should be (p, a)
                    let p1 = [modulus_bytes.as_ref(), curve_a_bytes.as_ref()].concat();
                    // (EcAdd only) p2 is (x2, y2), and x1 - x2 has to be non-zero to avoid division over zero in add.
                    let p2 = [one.as_ref(), one.as_ref()].concat();
                    let mut uninit: core::mem::MaybeUninit<[#item; 2]> = core::mem::MaybeUninit::uninit();
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_ecc_guest::OPCODE,
                        funct3 = ::openvm_ecc_guest::SW_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::SwBaseFunct7::SwSetup as usize
                            + #ec_idx
                                * (::openvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        rd = In uninit.as_mut_ptr(),
                        rs1 = In p1.as_ptr(),
                        rs2 = In p2.as_ptr()
                    );
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_ecc_guest::OPCODE,
                        funct3 = ::openvm_ecc_guest::SW_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::SwBaseFunct7::SwSetup as usize
                            + #ec_idx
                                * (::openvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        rd = In uninit.as_mut_ptr(),
                        rs1 = In p1.as_ptr(),
                        rs2 = Const "x0" // will be parsed as 0 and therefore transpiled to SETUP_EC_DOUBLE
                    );
                }
            }
        });

        setup_all_curves.push(quote::quote_spanned! { span.into() =>
            #setup_function();
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        #[cfg(target_os = "zkvm")]
        mod openvm_intrinsics_ffi_2 {
            use ::openvm_ecc_guest::{OPCODE, SW_FUNCT3, SwBaseFunct7};

            #(#externs)*
        }
        #(#setups)*
        pub fn setup_all_curves() {
            #(#setup_all_curves)*
        }
    })
}
