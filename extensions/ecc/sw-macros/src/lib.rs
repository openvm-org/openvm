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
/// For this macro to work, you must import the `elliptic_curve` crate and the
/// `openvm_ecc_guest::weierstrass` crate.
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
        create_extern_func!(sw_add_proj_extern_func);
        create_extern_func!(sw_double_proj_extern_func);
        create_extern_func!(sw_setup_extern_func);

        let group_ops_mod_name = format_ident!("{}_ops", struct_name_str.to_lowercase());

        let result = TokenStream::from(quote::quote_spanned! { span.into() =>
            extern "C" {
                fn #sw_add_proj_extern_func(rd: usize, rs1: usize, rs2: usize);
                fn #sw_double_proj_extern_func(rd: usize, rs1: usize);
                fn #sw_setup_extern_func(uninit: *mut core::ffi::c_void, p1: *const u8, p2: *const u8);
            }

            #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
            #[repr(C)]
            pub struct #struct_name {
                x: #intmod_type,
                y: #intmod_type,
                z: #intmod_type,
            }
            #[allow(non_upper_case_globals)]

            impl #struct_name {
                const fn identity() -> Self {
                    Self {
                        x: <#intmod_type as openvm_algebra_guest::IntMod>::ZERO,
                        y: <#intmod_type as openvm_algebra_guest::IntMod>::ONE,
                        z: <#intmod_type as openvm_algebra_guest::IntMod>::ZERO,
                    }
                }
                // Below are wrapper functions for the intrinsic instructions.
                // Should not be called directly.
                #[inline(always)]
                fn add_proj<const CHECK_SETUP: bool>(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let curve_a: #intmod_type = #const_a;
                        let curve_b: #intmod_type = #const_b;
                        let b3 = &(&curve_b + &curve_b) + &curve_b;
                        if curve_a == <#intmod_type as openvm_algebra_guest::IntMod>::ZERO {
                            // a=0: Algorithm 7 from ePrint 2015/1060
                            let t0 = &p1.x * &p2.x;
                            let t1 = &p1.y * &p2.y;
                            let t2 = &p1.z * &p2.z;
                            let t3 = &(&(&p1.x + &p1.y) * &(&p2.x + &p2.y)) - &t0 - &t1;
                            let t4 = &(&(&p1.y + &p1.z) * &(&p2.y + &p2.z)) - &t1 - &t2;
                            let y3_temp = &(&(&p1.x + &p1.z) * &(&p2.x + &p2.z)) - &t0 - &t2;
                            let x3_coeff = &(&t0 + &t0) + &t0; // 3*t0
                            let t2_b3 = &b3 * &t2;
                            let z3_temp = &t1 + &t2_b3;
                            let t1_sub = &t1 - &t2_b3;
                            let y3_b3 = &b3 * &y3_temp;
                            let x3_out = &(&t3 * &t1_sub) - &(&t4 * &y3_b3);
                            let y3_out = &(&t1_sub * &z3_temp) + &(&y3_b3 * &x3_coeff);
                            let z3_out = &(&z3_temp * &t4) + &(&x3_coeff * &t3);
                            #struct_name { x: x3_out, y: y3_out, z: z3_out }
                        } else {
                            // General a: Algorithm 1 from ePrint 2015/1060
                            let a = &curve_a;
                            let t0 = &p1.x * &p2.x;
                            let t1 = &p1.y * &p2.y;
                            let t2 = &p1.z * &p2.z;
                            let t3 = &(&(&p1.x + &p1.y) * &(&p2.x + &p2.y)) - &t0 - &t1;
                            let t4 = &(&(&p1.x + &p1.z) * &(&p2.x + &p2.z)) - &t0 - &t2;
                            let t5 = &(&(&p1.y + &p1.z) * &(&p2.y + &p2.z)) - &t1 - &t2;
                            let z3_temp = &(&b3 * &t2) + &(a * &t4);
                            let x3_temp = &t1 - &z3_temp;
                            let z3_temp2 = &t1 + &z3_temp;
                            let y3 = &x3_temp * &z3_temp2;
                            let t1_3t0 = &(&t0 + &t0) + &t0;
                            let t2_a = a * &t2;
                            let t4_b3 = &b3 * &t4;
                            let t1_val = &t1_3t0 + &t2_a;
                            let t2_val = &(a * &(&t0 - &t2_a)) + &t4_b3;
                            let t0_res = &t1_val * &t2_val;
                            let y3 = &y3 + &t0_res;
                            let t0_res = &t5 * &t2_val;
                            let x3 = &(&t3 * &x3_temp) - &t0_res;
                            let t0_res = &t3 * &t1_val;
                            let z3 = &(&t5 * &z3_temp2) + &t0_res;
                            #struct_name { x: x3, y: y3, z: z3 }
                        }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #sw_add_proj_extern_func(
                                uninit.as_mut_ptr() as usize,
                                p1 as *const #struct_name as usize,
                                p2 as *const #struct_name as usize
                            );
                            uninit.assume_init()
                        }
                    }
                }

                /// Projective doubling.
                #[inline(always)]
                fn double_proj<const CHECK_SETUP: bool>(p: &#struct_name) -> #struct_name {
                    #[cfg(not(target_os = "zkvm"))]
                    {
                        let curve_a: #intmod_type = #const_a;
                        let curve_b: #intmod_type = #const_b;
                        let b3 = &(&curve_b + &curve_b) + &curve_b;
                        if curve_a == <#intmod_type as openvm_algebra_guest::IntMod>::ZERO {
                            // a=0: Algorithm 9 from ePrint 2015/1060
                            let t0 = &p.y * &p.y;
                            let z3 = &(&(&(&t0 + &t0) + &t0 + &t0) + &t0 + &t0 + &t0) + &t0; // 8*t0
                            let t1 = &p.y * &p.z;
                            let t2_sq = &p.z * &p.z;
                            let t2 = &b3 * &t2_sq;
                            let x3_temp = &t2 * &z3;
                            let y3 = &t0 + &t2;
                            let z3 = &t1 * &z3;
                            let t1_d = &t2 + &t2;
                            let t2 = &t1_d + &t2;
                            let t0 = &t0 - &t2;
                            let y3 = &(&t0 * &y3) + &x3_temp;
                            let t1 = &p.x * &p.y;
                            let x3 = &(&t0 * &t1) + &(&t0 * &t1); // 2 * t0 * t1
                            #struct_name { x: x3, y: y3, z: z3 }
                        } else {
                            // General a: Algorithm 3 from ePrint 2015/1060
                            let a = &curve_a;
                            let t0 = &p.x * &p.x;
                            let t1 = &p.y * &p.y;
                            let t2 = &p.z * &p.z;
                            let t3 = &(&p.x * &p.y) + &(&p.x * &p.y); // 2*x*y
                            let z3_2xz = &(&p.x * &p.z) + &(&p.x * &p.z); // 2*x*z
                            let x3_temp = a * &z3_2xz;
                            let y3_temp = &(&b3 * &t2) + &x3_temp;
                            let x3_val = &t1 - &y3_temp;
                            let y3_val = &t1 + &y3_temp;
                            let y3 = &x3_val * &y3_val;
                            let x3 = &t3 * &x3_val;
                            let z3_b3 = &b3 * &z3_2xz;
                            let t2_a = a * &t2;
                            let t3_val = &(a * &(&t0 - &t2_a)) + &z3_b3;
                            let z3_3t0 = &(&t0 + &t0) + &t0;
                            let t0_val = &(&z3_3t0 + &t2_a) * &t3_val;
                            let y3 = &y3 + &t0_val;
                            let t2_yz = &(&p.y * &p.z) + &(&p.y * &p.z); // 2*y*z
                            let t0_val = &t2_yz * &t3_val;
                            let x3 = &x3 - &t0_val;
                            let z3 = &(&(&t2_yz * &t1) + &(&t2_yz * &t1) + &(&t2_yz * &t1)) + &(&t2_yz * &t1); // 4*t2_yz*t1
                            #struct_name { x: x3, y: y3, z: z3 }
                        }
                    }
                    #[cfg(target_os = "zkvm")]
                    {
                        if CHECK_SETUP {
                            Self::set_up_once();
                        }
                        let mut uninit: core::mem::MaybeUninit<#struct_name> = core::mem::MaybeUninit::uninit();
                        unsafe {
                            #sw_double_proj_extern_func(
                                uninit.as_mut_ptr() as usize,
                                p as *const #struct_name as usize,
                            );
                            uninit.assume_init()
                        }
                    }
                }

                // Helper function to call the setup instruction on first use
                #[inline(always)]
                #[cfg(target_os = "zkvm")]
                fn set_up_once() {
                    static is_setup: ::openvm_ecc_guest::once_cell::race::OnceBool = ::openvm_ecc_guest::once_cell::race::OnceBool::new();

                    is_setup.get_or_init(|| {
                        // p1 is (modulus, a, b).
                        let modulus_bytes = <<Self as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::MODULUS;
                        let mut one = [0u8; <<Self as openvm_ecc_guest::weierstrass::WeierstrassPoint>::Coordinate as openvm_algebra_guest::IntMod>::NUM_LIMBS];
                        one[0] = 1;
                        let curve_a_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<#struct_name as openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_A);
                        let curve_b_bytes = openvm_algebra_guest::IntMod::as_le_bytes(&<#struct_name as openvm_ecc_guest::weierstrass::WeierstrassPoint>::CURVE_B);
                        // p1 should be (modulus, a, b)
                        let p1 = [modulus_bytes.as_ref(), curve_a_bytes.as_ref(), curve_b_bytes.as_ref()].concat();
                        // p2 is (1, 1, 1) placeholder
                        let p2 = [one.as_ref(), one.as_ref(), one.as_ref()].concat();
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
                    // Check z == 0
                    unsafe {
                        self.z.eq_impl::<CHECK_SETUP>(&<#intmod_type as IntMod>::ZERO)
                    }
                }
            }

            impl core::cmp::PartialEq for #struct_name {
                fn eq(&self, other: &Self) -> bool {
                    (&self.x * &other.z) == (&other.x * &self.z)
                        && (&self.y * &other.z) == (&other.y * &self.z)
                }
            }

            impl core::cmp::Eq for #struct_name {}

            impl ::openvm_ecc_guest::weierstrass::WeierstrassPoint for #struct_name {
                const CURVE_A: #intmod_type = #const_a;
                const CURVE_B: #intmod_type = #const_b;
                const IDENTITY: Self = Self::identity();
                type Coordinate = #intmod_type;

                /// SAFETY: assumes that #intmod_type has a memory representation
                /// such that with repr(C), three coordinates are packed contiguously.
                #[inline(always)]
                fn as_le_bytes(&self) -> &[u8] {
                    unsafe { &*core::ptr::slice_from_raw_parts(self as *const Self as *const u8, <#intmod_type as openvm_algebra_guest::IntMod>::NUM_LIMBS * 3) }
                }

                #[inline(always)]
                fn from_xy_unchecked(x: Self::Coordinate, y: Self::Coordinate) -> Self {
                    Self { x, y, z: <#intmod_type as openvm_algebra_guest::IntMod>::ONE }
                }

                #[inline(always)]
                fn from_xyz_unchecked(x: Self::Coordinate, y: Self::Coordinate, z: Self::Coordinate) -> Self {
                    Self { x, y, z }
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
                fn z(&self) -> &Self::Coordinate {
                    &self.z
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
                fn z_mut(&mut self) -> &mut Self::Coordinate {
                    &mut self.z
                }

                #[inline(always)]
                fn into_coords(self) -> (Self::Coordinate, Self::Coordinate, Self::Coordinate) {
                    (self.x, self.y, self.z)
                }

                #[inline(always)]
                fn set_up_once() {
                    Self::set_up_once();
                }

                #[inline]
                fn add_impl<const CHECK_SETUP: bool>(&self, p2: &Self) -> Self {
                    Self::add_proj::<CHECK_SETUP>(self, p2)
                }

                #[inline]
                fn double_impl<const CHECK_SETUP: bool>(&self) -> Self {
                    Self::double_proj::<CHECK_SETUP>(self)
                }

                fn normalize(&self) -> Self {
                    use openvm_algebra_guest::DivUnsafe;
                    if self.is_identity_impl::<true>() {
                        Self::identity()
                    } else {
                        let x = (&self.x).div_unsafe(&self.z);
                        let y = (&self.y).div_unsafe(&self.z);
                        Self { x, y, z: <#intmod_type as openvm_algebra_guest::IntMod>::ONE }
                    }
                }

                fn is_identity(&self) -> bool {
                    self.is_identity_impl::<true>()
                }
            }

            impl core::ops::Neg for #struct_name {
                type Output = Self;

                fn neg(self) -> Self::Output {
                    #struct_name {
                        x: self.x,
                        y: -self.y,
                        z: self.z,
                    }
                }
            }

            impl core::ops::Neg for &#struct_name {
                type Output = #struct_name;

                fn neg(self) -> #struct_name {
                    #struct_name {
                        x: self.x.clone(),
                        y: core::ops::Neg::neg(&self.y),
                        z: self.z.clone(),
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
        let add_proj_extern_func =
            syn::Ident::new(&format!("sw_add_proj_extern_func_{struct_id}"), span.into());
        let double_proj_extern_func = syn::Ident::new(
            &format!("sw_double_proj_extern_func_{struct_id}"),
            span.into(),
        );
        let setup_extern_func =
            syn::Ident::new(&format!("sw_setup_extern_func_{struct_id}"), span.into());

        externs.push(quote::quote_spanned! { span.into() =>
            #[no_mangle]
            extern "C" fn #add_proj_extern_func(rd: usize, rs1: usize, rs2: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = SW_FUNCT3 as usize,
                    funct7 = SwBaseFunct7::SwAddProj as usize + #ec_idx
                        * (SwBaseFunct7::SHORT_WEIERSTRASS_MAX_KINDS as usize),
                    rd = In rd,
                    rs1 = In rs1,
                    rs2 = In rs2
                );
            }

            #[no_mangle]
            extern "C" fn #double_proj_extern_func(rd: usize, rs1: usize) {
                openvm::platform::custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = SW_FUNCT3 as usize,
                    funct7 = SwBaseFunct7::SwDoubleProj as usize + #ec_idx
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
                        rs2 = Const "x0" // will be parsed as 0 and therefore transpiled to SETUP_SW_EC_DOUBLE
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
