#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use std::sync::atomic::AtomicUsize;

use axvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Expr, ExprPath, Path, Token,
};

static CURVE_IDX: AtomicUsize = AtomicUsize::new(0);

/// This macro generates the code to setup the elliptic curve for a given modular type. Also it places the curve parameters into a special static variable to be later extracted from the ELF and used by the VM.
/// Usage:
/// ```
/// sw_declare! {
///     [TODO]
/// }
/// ```
///
/// For this macro to work, you must import the `elliptic_curve` crate and the `axvm_ecc_guest` crate..
#[proc_macro]
pub fn sw_declare(input: TokenStream) -> TokenStream {
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
        let ec_idx = CURVE_IDX.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        macro_rules! create_extern_func {
            ($name:ident) => {
                let $name = syn::Ident::new(
                    &format!(
                        "{}_{}",
                        stringify!($name),
                        intmod_type
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

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #sw_add_ne_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #sw_double_extern_func(rd: usize, rs1: usize);
            }

            #[derive(Eq, PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
            #[repr(C)]
            pub struct #struct_name {
                pub x: #intmod_type,
                pub y: #intmod_type,
            }

            impl #struct_name {
                pub const EC_IDX: usize = #ec_idx;

                // Below are wrapper functions for the intrinsic instructions.
                // Should not be called directly.
                #[inline(always)]
                fn add_ne(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        use axvm_algebra_guest::DivUnsafe;
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
                        use axvm_algebra_guest::DivUnsafe;
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
                        use axvm_algebra_guest::DivUnsafe;
                        let two = #intmod_type::from_u8(2);
                        let lambda = &p.x * &p.x * #intmod_type::from_u8(3).div_unsafe(&p.y * &two);
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
                        use axvm_algebra_guest::DivUnsafe;
                        let two = #intmod_type::from_u8(2);
                        let lambda = &self.x * &self.x * #intmod_type::from_u8(3).div_unsafe(&self.y * &two);
                        let x3 = &lambda * &lambda - &self.x * &two;
                        let y3 = &lambda * &(&self.x - &x3) - &self.y;
                        self.x = x3;
                        self.y = y3;
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

            impl ::axvm_ecc_guest::sw::SwPoint for #struct_name {
                type Coordinate = #intmod_type;

                // Ref: https://docs.rs/k256/latest/src/k256/arithmetic/affine.rs.html#247
                fn from_encoded_point<C: elliptic_curve::Curve>(p: &elliptic_curve::sec1::EncodedPoint<C>) -> Self
                where
                    C::FieldBytesSize: elliptic_curve::sec1::ModulusSize
                {
                    match p.coordinates() {
                        elliptic_curve::sec1::Coordinates::Identity => Self::identity(),
                        elliptic_curve::sec1::Coordinates::Uncompressed { x, y } => {
                            // Sec1 bytes are in big endian.
                            let x = Self::Coordinate::from_be_bytes(x.as_ref());
                            let y = Self::Coordinate::from_be_bytes(y.as_ref());
                            // TODO: Verify that the point is on the curve

                            Self { x, y }

                        }
                        elliptic_curve::sec1::Coordinates::Compact { x } => unimplemented!(),
                        elliptic_curve::sec1::Coordinates::Compressed { x, y_is_odd } => unimplemented!(),
                    }
                }

                fn to_sec1_bytes(&self, is_compressed: bool) -> Vec<u8>
                {
                    let mut bytes = Vec::new();
                    if is_compressed {
                        let y_is_odd = self.y().as_le_bytes()[0] & 1;
                        if y_is_odd == 1 {
                            bytes.push(0x03);
                        } else {
                            bytes.push(0x02);
                        }
                        bytes.extend_from_slice(&self.x().as_be_bytes());
                    } else {
                        bytes.push(0x04);
                        bytes.extend_from_slice(&self.x().as_be_bytes());
                        bytes.extend_from_slice(&self.y().as_be_bytes());
                    }
                    bytes
                }

                fn x(&self) -> Self::Coordinate {
                    self.x.clone()
                }

                fn y(&self) -> Self::Coordinate {
                    self.y.clone()
                }
            }

            impl Group for #struct_name {
                type SelfRef<'a> = &'a Self;

                fn identity() -> Self {
                    Self {
                        x: <#intmod_type as axvm_algebra_guest::IntMod>::ZERO,
                        y: <#intmod_type as axvm_algebra_guest::IntMod>::ZERO,
                    }
                }

                fn is_identity(&self) -> bool {
                    self.x == <#intmod_type as axvm_algebra_guest::IntMod>::ZERO && self.y == <#intmod_type as axvm_algebra_guest::IntMod>::ZERO
                }

                fn double(&self) -> Self {
                    if self.is_identity() {
                        self.clone()
                    } else {
                        Self::double_impl(self)
                    }
                }

                fn double_assign(&mut self) {
                    if !self.is_identity() {
                        Self::double_assign_impl(self);
                    }
                }
            }

            impl core::ops::Add<&#struct_name> for #struct_name {
                type Output = Self;

                fn add(mut self, p2: &#struct_name) -> Self::Output {
                    self.add_assign(p2);
                    self
                }
            }

            impl core::ops::Add for #struct_name {
                type Output = Self;

                fn add(self, rhs: Self) -> Self::Output {
                    self.add(&rhs)
                }
            }

            impl core::ops::Add<&#struct_name> for &#struct_name {
                type Output = #struct_name;

                fn add(self, p2: &#struct_name) -> Self::Output {
                    if self.is_identity() {
                        p2.clone()
                    } else if p2.is_identity() {
                        self.clone()
                    } else if self.x == p2.x {
                        if &self.y + &p2.y == <#intmod_type as axvm_algebra_guest::IntMod>::ZERO {
                            #struct_name::identity()
                        } else {
                            #struct_name::double_impl(self)
                        }
                    } else {
                        #struct_name::add_ne(self, p2)
                    }
                }
            }

            impl core::ops::AddAssign<&#struct_name> for #struct_name {
                fn add_assign(&mut self, p2: &#struct_name) {
                    if self.is_identity() {
                        *self = p2.clone();
                    } else if p2.is_identity() {
                        // do nothing
                    } else if self.x == p2.x {
                        if &self.y + &p2.y == <#intmod_type as axvm_algebra_guest::IntMod>::ZERO {
                            *self = Self::identity();
                        } else {
                            Self::double_assign_impl(self);
                        }
                    } else {
                        Self::add_ne_assign(self, p2);
                    }
                }
            }

            impl core::ops::AddAssign for #struct_name {
                fn add_assign(&mut self, rhs: Self) {
                    self.add_assign(&rhs);
                }
            }

            impl core::ops::Neg for #struct_name {
                type Output = Self;

                fn neg(self) -> Self::Output {
                    Self {
                        x: self.x,
                        y: -self.y,
                    }
                }
            }

            impl core::ops::Sub<&#struct_name> for #struct_name {
                type Output = Self;

                fn sub(self, rhs: &#struct_name) -> Self::Output {
                    self.sub(rhs.clone())
                }
            }

            impl core::ops::Sub for #struct_name {
                type Output = #struct_name;

                fn sub(self, rhs: Self) -> Self::Output {
                    self.add(rhs.neg())
                }
            }

            impl core::ops::Sub<&#struct_name> for &#struct_name {
                type Output = #struct_name;

                fn sub(self, p2: &#struct_name) -> Self::Output {
                    self.add(&p2.clone().neg())
                }
            }

            impl core::ops::SubAssign<&#struct_name> for #struct_name {
                fn sub_assign(&mut self, p2: &#struct_name) {
                    self.sub_assign(p2.clone());
                }
            }

            impl core::ops::SubAssign for #struct_name {
                fn sub_assign(&mut self, rhs: Self) {
                    self.add_assign(rhs.neg());
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
        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #add_ne_extern_func(rd: usize, rs1: usize, rs2: usize) {
                axvm_platform::custom_insn_r!(
                    ::axvm_ecc_guest::OPCODE,
                    ::axvm_ecc_guest::SW_FUNCT3 as usize,
                    ::axvm_ecc_guest::SwBaseFunct7::SwAddNe as usize + #ec_idx
                        * (::axvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd,
                    rs1,
                    rs2
                );
            }

            #[no_mangle]
            extern "C" fn #double_extern_func(rd: usize, rs1: usize) {
                axvm_platform::custom_insn_r!(
                    ::axvm_ecc_guest::OPCODE,
                    ::axvm_ecc_guest::SW_FUNCT3 as usize,
                    ::axvm_ecc_guest::SwBaseFunct7::SwDouble as usize + #ec_idx
                        * (::axvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd,
                    rs1,
                    "x0"
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
                    // y1 needs to be non-zero to avoid division by zero in double.
                    let modulus_bytes = <#item as axvm_algebra_guest::IntMod>::MODULUS;
                    let mut one = [0u8; <#item as axvm_algebra_guest::IntMod>::NUM_BYTES];
                    one[0] = 1;
                    let p1 = [modulus_bytes.as_ref(), one.as_ref()].concat();
                    // (EcAdd only) p2 is (x2, y2), and x1 - x2 has to be non-zero to avoid division over zero in add.
                    let p2 = [one.as_ref(), one.as_ref()].concat();
                    let mut uninit: core::mem::MaybeUninit<[#item; 2]> = core::mem::MaybeUninit::uninit();
                    axvm_platform::custom_insn_r!(
                        ::axvm_ecc_guest::OPCODE,
                        ::axvm_ecc_guest::SW_FUNCT3 as usize,
                        ::axvm_ecc_guest::SwBaseFunct7::SwSetup as usize
                            + #ec_idx
                                * (::axvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        uninit.as_mut_ptr(),
                        p1.as_ptr(),
                        p2.as_ptr()
                    );
                    axvm_platform::custom_insn_r!(
                        ::axvm_ecc_guest::OPCODE,
                        ::axvm_ecc_guest::SW_FUNCT3 as usize,
                        ::axvm_ecc_guest::SwBaseFunct7::SwSetup as usize
                            + #ec_idx
                                * (::axvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        uninit.as_mut_ptr(),
                        p1.as_ptr(),
                        "x0" // will be parsed as 0 and therefore transpiled to SETUP_EC_DOUBLE
                    );
                }
            }
        });

        setup_all_curves.push(quote::quote_spanned! { span.into() =>
            #setup_function();
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        // #(#axiom_section)*
        #[cfg(target_os = "zkvm")]
        mod axvm_intrinsics_ffi_2 {
            #(#externs)*
        }
        #(#setups)*
        pub fn setup_all_curves() {
            #(#setup_all_curves)*
        }
    })
}
