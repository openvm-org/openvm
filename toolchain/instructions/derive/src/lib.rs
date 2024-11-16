extern crate alloc;
extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Lit, Meta, MetaNameValue};

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
                impl UsizeOpcode for #name {
                    fn default_offset() -> usize {
                        #offset
                    }

                    fn from_usize(value: usize) -> Self {
                        #name(<#inner_ty as UsizeOpcode>::from_usize(value))
                    }

                    fn as_usize(&self) -> usize {
                        self.0.as_usize()
                    }
                }
            }.into()
        },
        Data::Enum(_) => {
            quote! {
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
            }.into()
        },
        Data::Union(_) => unimplemented!("Unions are not supported")
    }
}
