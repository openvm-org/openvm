#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use axvm_macros_common::Stmts;
use proc_macro::TokenStream;
use syn::{parse_macro_input, Stmt};

/// This macro generates the code to setup the elliptic curve for a given modular type. Also it places the curve parameters into a special static variable to be later extracted from the ELF and used by the VM.
/// Usage:
/// ```
/// ec_setup! {
///     [TODO]
/// }
/// ```
/// This [TODO add description].
#[proc_macro]
pub fn ec_setup(input: TokenStream) -> TokenStream {
    let Stmts { stmts } = parse_macro_input!(input as Stmts);

    let mut output = Vec::new();
    let mut ec_idx = 0usize;

    let span = proc_macro::Span::call_site();

    for stmt in stmts {
        let result: Result<TokenStream, &str> = match stmt.clone() {
            Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    if let syn::Expr::Path(path) = *assign.left {
                        let struct_name = path
                            .path
                            .get_ident()
                            .expect("Left hand side must be an identifier")
                            .to_string();
                        let struct_name = syn::Ident::new(&struct_name, span.into());

                        if let syn::Expr::Path(intmod_type) = &*assign.right {
                            let result = TokenStream::from(quote::quote_spanned! { span.into() =>

                                #[derive(Eq, PartialEq, Clone)]
                                #[repr(C)]
                                pub struct #struct_name {
                                    pub x: #intmod_type,
                                    pub y: #intmod_type,
                                }

                                impl #struct_name {
                                    pub const IDENTITY: Self = Self {
                                        x: #intmod_type::ZERO,
                                        y: #intmod_type::ZERO,
                                    };

                                    pub const EC_IDX: usize = #ec_idx;

                                    pub fn is_identity(&self) -> bool {
                                        self.x == Self::IDENTITY.x && self.y == Self::IDENTITY.y
                                    }

                                    // Two points can be equal or not.
                                    pub fn add(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                                        if p1.is_identity() {
                                            p2.clone()
                                        } else if p2.is_identity() {
                                            p1.clone()
                                        } else if p1.x == p2.x {
                                            if &p1.y + &p2.y == #intmod_type::ZERO {
                                                Self::IDENTITY
                                            } else {
                                                Self::double(p1)
                                            }
                                        } else {
                                            Self::add_ne(p1, p2)
                                        }
                                    }

                                    #[inline(always)]
                                    pub fn add_ne(p1: &#struct_name, p2: &#struct_name) -> #struct_name {
                                        #[cfg(not(target_os = "zkvm"))]
                                        {
                                            let lambda = (&p2.y - &p1.y) / (&p2.x - &p1.x);
                                            let x3 = &lambda * &lambda - &p1.x - &p2.x;
                                            let y3 = &lambda * &(&p1.x - &x3) - &p1.y;
                                            #struct_name { x: x3, y: y3 }
                                        }
                                        #[cfg(target_os = "zkvm")]
                                        {
                                            let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                            custom_insn_r!(
                                                CUSTOM_1,
                                                Custom1Funct3::ShortWeierstrass as usize,
                                                SwBaseFunct7::SwAddNe as usize,
                                                uninit.as_mut_ptr(),
                                                p1 as *const #struct_name,
                                                p2 as *const #struct_name
                                            );
                                            unsafe { uninit.assume_init() }
                                        }
                                    }

                                    #[inline(always)]
                                    pub fn add_ne_assign(&mut self, p2: &#struct_name) {
                                        #[cfg(not(target_os = "zkvm"))]
                                        {
                                            let lambda = (&p2.y - &self.y) / (&p2.x - &self.x);
                                            let x3 = &lambda * &lambda - &self.x - &p2.x;
                                            let y3 = &lambda * &(&self.x - &x3) - &self.y;
                                            self.x = x3;
                                            self.y = y3;
                                        }
                                        #[cfg(target_os = "zkvm")]
                                        {
                                            custom_insn_r!(
                                                CUSTOM_1,
                                                Custom1Funct3::ShortWeierstrass as usize,
                                                SwBaseFunct7::SwAddNe as usize,
                                                self as *mut #struct_name,
                                                self as *const #struct_name,
                                                p2 as *const #struct_name
                                            );
                                        }
                                    }

                                    #[inline(always)]
                                    pub fn double(p: &#struct_name) -> #struct_name {
                                        #[cfg(not(target_os = "zkvm"))]
                                        {
                                            let lambda = &p.x * &p.x * 3 / (&p.y * 2);
                                            let x3 = &lambda * &lambda - &p.x * 2;
                                            let y3 = &lambda * &(&p.x - &x3) - &p.y;
                                            #struct_name { x: x3, y: y3 }
                                        }
                                        #[cfg(target_os = "zkvm")]
                                        {
                                            let mut uninit: MaybeUninit<#struct_name> = MaybeUninit::uninit();
                                            custom_insn_r!(
                                                CUSTOM_1,
                                                Custom1Funct3::ShortWeierstrass as usize,
                                                SwBaseFunct7::SwDouble as usize,
                                                uninit.as_mut_ptr(),
                                                p as *const #struct_name,
                                                "x0"
                                            );
                                            unsafe { uninit.assume_init() }
                                        }
                                    }

                                    #[inline(always)]
                                    pub fn double_assign(&mut self) {
                                        #[cfg(not(target_os = "zkvm"))]
                                        {
                                            let lambda = &self.x * &self.x * 3 / (&self.y * 2);
                                            let x3 = &lambda * &lambda - &self.x * 2;
                                            let y3 = &lambda * &(&self.x - &x3) - &self.y;
                                            self.x = x3;
                                            self.y = y3;
                                        }
                                        #[cfg(target_os = "zkvm")]
                                        {
                                            custom_insn_r!(
                                                CUSTOM_1,
                                                Custom1Funct3::ShortWeierstrass as usize,
                                                SwBaseFunct7::SwDouble as usize,
                                                self as *mut #struct_name,
                                                self as *const #struct_name,
                                                "x0"
                                            );
                                        }
                                    }
                                }

                            });

                            ec_idx += 1;

                            Ok(result)
                        } else {
                            Err("Right side must be a string literal")
                        }
                    } else {
                        Err("Left side of assignment must be an identifier")
                    }
                } else {
                    Err("Only simple assignments are supported")
                }
            }
            _ => Err("Only assignments are supported"),
        };
        if let Err(err) = result {
            return syn::Error::new_spanned(stmt, err).to_compile_error().into();
        } else {
            output.push(result.unwrap());
        }
    }

    // let mut serialized_moduli = (moduli.len() as u32)
    //     .to_le_bytes()
    //     .into_iter()
    //     .collect::<Vec<_>>();
    // for modulus_bytes in moduli {
    //     serialized_moduli.extend((modulus_bytes.len() as u32).to_le_bytes());
    //     serialized_moduli.extend(modulus_bytes);
    // }
    // let serialized_len = serialized_moduli.len();
    // // Note: this also prevents the macro from being called twice
    // output.push(TokenStream::from(quote::quote! {
    //     #[cfg(target_os = "zkvm")]
    //     #[link_section = ".axiom"]
    //     #[no_mangle]
    //     #[used]
    //     static AXIOM_SERIALIZED_MODULI: [u8; #serialized_len] = [#(#serialized_moduli),*];
    // }));

    TokenStream::from_iter(output)
}
