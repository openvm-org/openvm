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

    for stmt in stmts {
        match stmt.clone() {
            Stmt::Expr(expr, _) => {
                if let syn::Expr::Assign(assign) = expr {
                    if let syn::Expr::Path(path) = *assign.left {
                        let ident = path.path.segments[0].ident.to_string();

                        if let syn::Expr::Lit(lit) = &*assign.right {
                            if let syn::Lit::Str(str_lit) = &lit.lit {
                                let modulus = num_bigint::BigUint::parse_bytes(
                                    str_lit.value().as_bytes(),
                                    10,
                                )
                                .expect("Failed to parse modulus");
                                let modulus_bytes = modulus.to_bytes_le();
                                let limbs = modulus_bytes.len();

                                let struct_name = format!("IntMod_{}", ident);
                                let struct_name = syn::Ident::new(
                                    &struct_name,
                                    proc_macro::Span::call_site().into(),
                                );

                                output.push(TokenStream::from(quote::quote! {

                                    #[repr(C, align(8))]
                                    struct #struct_name([u8; #limbs]);

                                    impl #struct_name {
                                        const MODULUS: [u8; #limbs] = [#(#modulus_bytes),*];
                                        const MOD_IDX: usize = #mod_idx;

                                        pub fn new() -> Self {
                                            Self([0u8; #limbs])
                                        }

                                        pub fn from_bytes(bytes: &[u8]) -> Self {
                                            let mut ret = Self::new();
                                            ret.0.copy_from_slice(bytes);
                                            ret
                                        }
                                    }

                                }));

                                moduli.push(modulus_bytes);
                                mod_idx += 1;
                            } else {
                                return syn::Error::new_spanned(
                                    assign.right,
                                    "Right side must be a string literal",
                                )
                                .to_compile_error()
                                .into();
                            }
                        } else {
                            return syn::Error::new_spanned(
                                assign.right,
                                "Right side must be a string literal",
                            )
                            .to_compile_error()
                            .into();
                        }
                    } else {
                        return syn::Error::new_spanned(
                            stmt,
                            "Left side of assignment must be an identifier",
                        )
                        .to_compile_error()
                        .into();
                    }
                } else {
                    return syn::Error::new_spanned(stmt, "Only simple assignments are supported")
                        .to_compile_error()
                        .into();
                }
            }
            _ => {
                return syn::Error::new_spanned(stmt, "Only assignments are supported")
                    .to_compile_error()
                    .into();
            }
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
