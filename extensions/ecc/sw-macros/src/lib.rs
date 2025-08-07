extern crate proc_macro;

use openvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use quote::format_ident;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ExprPath, LitStr, Token,
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
        let struct_name_str = item.name.to_string();
        let struct_name = syn::Ident::new(&struct_name_str, span.into());
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
                    &format!("{}_{}", stringify!($name), struct_name_str),
                    span.into(),
                );
            };
        }
        create_extern_func!(sw_add_ne_extern_func);
        create_extern_func!(sw_double_extern_func);
        create_extern_func!(sw_setup_extern_func);

        let group_ops_mod_name = format_ident!("{}_ops", struct_name_str.to_lowercase());

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #sw_add_ne_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #sw_double_extern_func(rd: usize, rs1: usize);
                fn #sw_setup_extern_func(uninit: *mut core::ffi::c_void, p1: *const u8, p2: *const u8);
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
                unsafe fn add_ne<const CHECK_SETUP: bool>(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
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
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        #sw_add_ne_extern_func(
                            uninit.as_mut_ptr() as usize,
                            p1 as *const #struct_name as usize,
                            p2 as *const #struct_name as usize
                        );
                        uninit.assume_init()
                    }
                }

                #[inline(always)]
                unsafe fn add_ne_assign<const CHECK_SETUP: bool>(&mut self, p2: &#struct_name) {
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
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        #sw_add_ne_extern_func(
                            self as *mut #struct_name as usize,
                            self as *const #struct_name as usize,
                            p2 as *const #struct_name as usize
                        );
                    }
                }

                /// Assumes that `p` is not identity.
                #[inline(always)]
                unsafe fn double_impl<const CHECK_SETUP: bool>(p: &#struct_name) -> #struct_name {
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
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        #sw_double_extern_func(
                            uninit.as_mut_ptr() as usize,
                            p as *const #struct_name as usize,
                        );
                        uninit.assume_init()
                    }
                }

                // Helper function to call the setup instruction on first use
                #[inline(always)]
                #[cfg(target_os = "zkvm")]
                fn set_up_once() {
                    static is_setup: ::openvm_ecc_guest::once_cell::race::OnceBool = ::openvm_ecc_guest::once_cell::race::OnceBool::new();

                    is_setup.get_or_init(|| {
                        // p1 is (x1, y1), and x1 must be the modulus.
                        // y1 can be anything for SetupEcAdd, but must equal `a` for SetupEcDouble
                        let modulus_bytes = <<Self as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::MODULUS;
                        let mut one = [0u8; <<Self as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::NUM_LIMBS];
                        one[0] = 1;
                        let curve_a_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<#struct_name as openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_A);
                        // p1 should be (p, a)
                        let p1 = [modulus_bytes.as_ref(), curve_a_bytes.as_ref()].concat();
                        // (EcAdd only) p2 is (x2, y2), and x1 - x2 has to be non-zero to avoid division over zero in add.
                        let p2 = [one.as_ref(), one.as_ref()].concat();
                        let mut uninit: core::mem::MaybeUninit<[Self; 2]> = core::mem::MaybeUninit::uninit();

                        unsafe { #sw_setup_extern_func(uninit.as_mut_ptr() as *mut core::ffi::c_void, p1.as_ptr(), p2.as_ptr()); }
                        <#intmod_type as openvm_algebra_guest::IntMod>::set_up_once();
                        true
                    });
                }

                #[inline(always)]
                #[cfg(not(target_os = "zkvm"))]
                fn set_up_once() {
                    // No-op for non-ZKVM targets
                }

                #[inline(always)]
                fn is_identity_impl<const CHECK_SETUP: bool>(&self) -> bool {
                    use openvm_algebra_guest::IntMod;
                    // Safety: Self::set_up_once() ensures IntMod::set_up_once() has been called.
                    unsafe {
                        self.x.eq_impl::<CHECK_SETUP>(&#intmod_type::ZERO) && self.y.eq_impl::<CHECK_SETUP>(&#intmod_type::ZERO)
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
                fn set_up_once() {
                    Self::set_up_once();
                }

                #[inline]
                fn add_assign_impl<const CHECK_SETUP: bool>(&mut self, p2: &Self) {
                    use openvm_algebra_guest::IntMod;

                    if CHECK_SETUP {
                        // Call setup here so we skip it below
                        #intmod_type::set_up_once();
                    }

                    if self.is_identity_impl::<CHECK_SETUP>() {
                        *self = p2.clone();
                    } else if p2.is_identity_impl::<CHECK_SETUP>() {
                        // do nothing
                    } else if unsafe { self.x.eq_impl::<false>(&p2.x) } { // Safety: we called IntMod setup above
                        let sum_ys = unsafe { self.y.add_ref::<false>(&p2.y) };
                        // Safety: we called IntMod setup above
                        if unsafe { IntMod::eq_impl::<false>(&sum_ys, &<#intmod_type as IntMod>::ZERO) } {
                            *self = Self::identity();
                        } else {
                            unsafe {
                                self.double_assign_nonidentity::<CHECK_SETUP>();
                            }
                        }
                    } else {
                        unsafe {
                            self.add_ne_assign_nonidentity::<CHECK_SETUP>(p2);
                        }
                    }
                }

                #[inline(always)]
                fn double_assign_impl<const CHECK_SETUP: bool>(&mut self) {
                    if !self.is_identity_impl::<CHECK_SETUP>() {
                        unsafe {
                            self.double_assign_nonidentity::<CHECK_SETUP>();
                        }
                    }
                }

                #[inline(always)]
                unsafe fn add_ne_nonidentity<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self {
                    Self::add_ne::<CHECK_SETUP>(self, p2)
                }

                #[inline(always)]
                unsafe fn add_ne_assign_nonidentity<const CHECK_SETUP: bool>(&mut self, p2: &Self) {
                    Self::add_ne_assign::<CHECK_SETUP>(self, p2);
                }

                #[inline(always)]
                unsafe fn sub_ne_nonidentity<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self {
                    Self::add_ne::<CHECK_SETUP>(self, &p2.clone().neg())
                }

                #[inline(always)]
                unsafe fn sub_ne_assign_nonidentity<const CHECK_SETUP: bool>(&mut self, p2: &Self) {
                    Self::add_ne_assign::<CHECK_SETUP>(self, &p2.clone().neg());
                }

                #[inline(always)]
                unsafe fn double_nonidentity<const CHECK_SETUP: bool>(&self) -> Self {
                    Self::double_impl::<CHECK_SETUP>(self)
                }

                #[inline(always)]
                unsafe fn double_assign_nonidentity<const CHECK_SETUP: bool>(&mut self) {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        *self = Self::double_impl::<CHECK_SETUP>(self);
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        #sw_double_extern_func(
                            self as *mut #struct_name as usize,
                            self as *const #struct_name as usize
                        );
                    }
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
                use ::openvm_ecc_guest::{weierstrass::{WeierstrassPoint, FromCompressed}, impl_sw_group_ops, algebra::IntMod};
                use super::*;

                impl_sw_group_ops!(#struct_name, #intmod_type);

                impl FromCompressed<#intmod_type> for #struct_name {
                    fn decompress(x: #intmod_type, rec_id: &u8) -> Option<Self> {
                        use openvm_algebra_guest::Sqrt;
                        let y_squared = &x * &x * &x + &<#struct_name as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_A * &x + &<#struct_name as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_B;
                        let y = y_squared.sqrt();
                        match y {
                            None => None,
                            Some(y) => {
                                let correct_y = if y.as_le_bytes()[0] & 1 == *rec_id & 1 {
                                    y
                                } else {
                                    -y
                                };
                                // If y = 0 then negating y doesn't change its parity
                                if correct_y.as_le_bytes()[0] & 1 != *rec_id & 1 {
                                    return None;
                                }
                                // In order for sqrt() to return Some, we are guaranteed that y * y == y_squared, which already proves (x, correct_y) is on the curve
                                Some(<#struct_name as ::openvm_ecc_guest::weierstrass::WeierstrassPoint>::from_xy_unchecked(x, correct_y))
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

struct SwDefine {
    items: Vec<String>,
}

impl Parse for SwDefine {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let items = input.parse_terminated(<LitStr as Parse>::parse, Token![,])?;
        Ok(Self {
            items: items.into_iter().map(|e| e.value()).collect(),
        })
    }
}

#[proc_macro]
pub fn sw_init(input: TokenStream) -> TokenStream {
    let SwDefine { items } = parse_macro_input!(input as SwDefine);

    let mut externs = Vec::new();

    let span = proc_macro::Span::call_site();

    for (ec_idx, struct_id) in items.into_iter().enumerate() {
        // Unique identifier shared by sw_define! and sw_init! used for naming the extern funcs.
        // Currently it's just the struct type name.
        let add_ne_extern_func =
            syn::Ident::new(&format!("sw_add_ne_extern_func_{}", struct_id), span.into());
        let double_extern_func =
            syn::Ident::new(&format!("sw_double_extern_func_{}", struct_id), span.into());
        let setup_extern_func =
            syn::Ident::new(&format!("sw_setup_extern_func_{}", struct_id), span.into());

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
            extern "C" fn #setup_extern_func(uninit: *mut core::ffi::c_void, p1: *const u8, p2: *const u8) {
                #[cfg(target_os = "zkvm")]
                {
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_ecc_guest::OPCODE,
                        funct3 = ::openvm_ecc_guest::SW_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::SwBaseFunct7::SwSetup as usize
                            + #ec_idx
                                * (::openvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        rd = In uninit,
                        rs1 = In p1,
                        rs2 = In p2
                    );
                    openvm::platform::custom_insn_r!(
                        opcode = ::openvm_ecc_guest::OPCODE,
                        funct3 = ::openvm_ecc_guest::SW_FUNCT3 as usize,
                        funct7 = ::openvm_ecc_guest::SwBaseFunct7::SwSetup as usize
                            + #ec_idx
                                * (::openvm_ecc_guest::SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                        rd = In uninit,
                        rs1 = In p1,
                        rs2 = Const "x0" // will be parsed as 0 and therefore transpiled to SETUP_EC_DOUBLE
                    );


                }
            }
        });
    }

    TokenStream::from(quote::quote_spanned! { span.into() =>
        #[allow(non_snake_case)]
        #[cfg(target_os = "zkvm")]
        mod openvm_intrinsics_ffi_2 {
            use ::openvm_ecc_guest::{OPCODE, SW_FUNCT3, SwBaseFunct7};

            #(#externs)*
        }
    })
}
