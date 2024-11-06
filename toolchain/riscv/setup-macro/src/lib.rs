extern crate proc_macro;

use proc_macro::TokenStream;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Stmt,
};

struct Stmts {
    stmts: Vec<Stmt>,
}

impl Parse for Stmts {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut stmts = Vec::new();
        while !input.is_empty() {
            stmts.push(input.parse()?);
        }
        Ok(Stmts { stmts })
    }
}

#[proc_macro]
pub fn moduli_setup(input: TokenStream) -> TokenStream {
    let Stmts { stmts } = parse_macro_input!(input as Stmts);

    let mut output = Vec::new();
    let mut mod_idx = 0usize;

    let mut moduli = Vec::new();

    output.push(TokenStream::from(quote::quote! {
        use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
        #[cfg(target_os = "zkvm")]
        use core::{borrow::BorrowMut, mem::MaybeUninit};

        #[cfg(not(target_os = "zkvm"))]
        use num_bigint_dig::{traits::ModInverse, BigUint, Sign, ToBigInt};
        #[cfg(not(target_os = "zkvm"))]
        use axvm::intrinsics::biguint_to_limbs;
    }));

    let string_to_bytes = |s: &str| {
        if s.starts_with("0x") {
            return s
                .chars()
                .skip(2)
                .filter(|c| !c.is_whitespace())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .chunks(2)
                .map(|ch| u8::from_str_radix(&ch.iter().rev().collect::<String>(), 16).unwrap())
                .collect();
        }
        let mut digits = s
            .chars()
            .map(|c| c.to_digit(10).expect("Invalid numeric literal"))
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        while !digits.is_empty() {
            let mut rem = 0u32;
            let mut new_digits = Vec::new();
            for &d in digits.iter() {
                rem = rem * 10 + d;
                new_digits.push(rem / 256);
                rem %= 256;
            }
            digits = new_digits.into_iter().skip_while(|&d| d == 0).collect();
            bytes.push(rem as u8);
        }
        bytes.reverse();
        bytes
    };

    for stmt in stmts {
        let result: Result<TokenStream, &str> = match stmt.clone() {
            Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    if let syn::Expr::Path(path) = *assign.left {
                        let struct_name = path.path.segments[0].ident.to_string();

                        if let syn::Expr::Lit(lit) = &*assign.right {
                            if let syn::Lit::Str(str_lit) = &lit.lit {
                                let modulus_bytes = string_to_bytes(&str_lit.value());
                                let limbs = modulus_bytes.len();

                                let struct_name = syn::Ident::new(
                                    &struct_name,
                                    proc_macro::Span::call_site().into(),
                                );

                                let result = TokenStream::from(quote::quote! {
                                    // placeholder
                                });

                                moduli.push(modulus_bytes);
                                mod_idx += 1;

                                Ok(result)
                            } else {
                                Err("Right side must be a string literal")
                            }
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

    let mut serialized_moduli = (moduli.len() as u32)
        .to_le_bytes()
        .into_iter()
        .collect::<Vec<_>>();
    for modulus_bytes in moduli {
        serialized_moduli.extend((modulus_bytes.len() as u32).to_le_bytes());
        serialized_moduli.extend(modulus_bytes);
    }
    let serialized_len = serialized_moduli.len();
    // Note: this also prevents the macro from being called twice
    output.push(TokenStream::from(quote::quote! {
        #[link_section = ".axiom"]
        #[no_mangle]
        #[used]
        static AXIOM_SERIALIZED_MODULI: [u8; #serialized_len] = [#(#serialized_moduli),*];
    }));

    TokenStream::from_iter(output)
}
