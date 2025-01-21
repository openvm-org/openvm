extern crate alloc;
extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Expr, ExprLit, Fields, Lit, Meta};

#[proc_macro_derive(LocalOpcode, attributes(opcode_offset))]
pub fn local_opcode_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let mut offset = None;
    for attr in ast.attrs {
        if let Meta::NameValue(meta) = attr.meta {
            if meta.path.is_ident("opcode_offset") {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Int(lit_int),
                    ..
                }) = meta.value
                {
                    offset = Some(lit_int.base10_parse::<usize>().unwrap());
                }
            }
        }
    }
    let offset = offset.expect("opcode_offset attribute not found");

    match &ast.data {
        Data::Struct(inner) => {
            let inner = match &inner.fields {
                Fields::Unnamed(fields) => {
                    if fields.unnamed.len() != 1 {
                        panic!("Only one unnamed field is supported");
                    }
                    fields.unnamed.first().unwrap().clone()
                }
                _ => panic!("Only unnamed fields are supported"),
            };
            let inner_ty = inner.ty;

            quote! {
                impl LocalOpcode for #name {
                    const CLASS_OFFSET: usize = #offset;

                    fn from_usize(value: usize) -> Self {
                        #name(<#inner_ty as LocalOpcode>::from_usize(value))
                    }

                    fn local_usize(&self) -> usize {
                        self.0.local_usize()
                    }
                }
            }.into()
        },
        Data::Enum(_) => {
            quote! {
                impl LocalOpcode for #name {
                    const CLASS_OFFSET: usize = #offset;

                    fn from_usize(value: usize) -> Self {
                        Self::from_repr(value.try_into().unwrap())
                            .unwrap_or_else(|| panic!("Failed to convert usize {} to opcode {}", value, stringify!(#name)))
                    }

                    fn local_usize(&self) -> usize {
                        *self as usize
                    }
                }
            }.into()
        },
        Data::Union(_) => unimplemented!("Unions are not supported")
    }
}
