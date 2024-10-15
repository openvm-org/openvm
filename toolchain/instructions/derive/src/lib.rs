extern crate alloc;
extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Lit, Meta, MetaNameValue};

#[proc_macro_derive(UsizeOpcode, attributes(opcode_offset))]
pub fn usize_opcode_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let mut offset = None;
    for attr in ast.attrs {
        if let Ok(Meta::NameValue(MetaNameValue {
            path,
            lit: Lit::Int(lit_int),
            ..
        })) = attr.parse_meta()
        {
            if path.is_ident("opcode_offset") {
                offset = Some(lit_int.base10_parse::<usize>().unwrap());
            }
        }
    }
    let offset = offset.expect("opcode_offset attribute not found");

    let methods = quote! {
        impl UsizeOpcode for #name {
            fn default_offset() -> usize {
                #offset
            }

            fn from_usize(value: usize) -> Self {
                Self::from_repr(value.try_into().unwrap())
                    .unwrap_or_else(|| panic!("Failed to convert usize {} to opcode {}", value, stringify!(#name)))
            }

            fn as_usize(&self) -> usize {
                *self as usize
            }
        }
    };

    TokenStream::from(methods)
}
