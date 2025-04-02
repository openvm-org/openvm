use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::quote;

use crate::core::ir::{Type, BOOLEAN_TYPE_NAME};
use crate::core::type_resolution::TypeSet;
use crate::parser::metadata::ParserMetadata;
use crate::transpilation::execution::constants::{
    ident, type_to_identifier_trace_generation, type_to_rust, usize_to_field_elem, ZK_IDENTIFIER,
};

const TO_CELLS: &str = "to_cells";

impl Type {
    pub fn to_cells(&self, value: TokenStream, result: TokenStream) -> TokenStream {
        let function_name = ident(&format!(
            "{}_{}",
            TO_CELLS,
            type_to_identifier_trace_generation(self)
        ));
        quote! {
            #function_name(#value, #result);
        }
    }
    pub fn transpile_to_cells(&self, type_set: &TypeSet) -> TokenStream {
        let value = ident("value");
        let result = ident("result");
        let body = match self {
            Type::Field => quote! { #result[0] = #value; },
            Type::NamedType(name) => {
                if name == BOOLEAN_TYPE_NAME {
                    quote! { #result[0] = F::from_bool(#value) }
                } else {
                    let mut branches = vec![];
                    let algebraic_type = type_set
                        .get_algebraic_type(name, &ParserMetadata::default())
                        .unwrap();
                    for (i, variant) in algebraic_type.variants.iter().enumerate() {
                        let mut body = vec![];
                        if variant.components.len() == 1 {
                            let variant_index = usize_to_field_elem(i);
                            body.push(quote! { #result[0] = #variant_index; })
                        }
                        let mut offset = 1;
                        for (i, component) in variant.components.iter().enumerate() {
                            let size = type_set.calc_type_size(component);
                            if size > 0 {
                                let new_offset = offset + size;
                                body.push(component.to_cells(
                                    quote! { #value.#i },
                                    quote! { &mut #result[#offset..#new_offset] },
                                ));
                                offset = new_offset;
                            }
                        }
                        let constructor = ident(&variant.name);
                        branches.push(quote! {
                            #constructor(..) => {
                                #(#body)*
                            }
                        });
                    }
                    quote! {
                        match #value {
                            #(#branches)*
                        }
                    }
                }
            }
            Type::Reference(_)
            | Type::AppendablePrefix(_, _)
            | Type::ReadablePrefix(_, _)
            | Type::Array(_) => {
                let zk_identifier = ident(ZK_IDENTIFIER);
                let x = usize_to_field_elem(quote! { #value.#zk_identifier });
                quote! {
                    #result[0] = #x;
                }
            }
            Type::Unmaterialized(_) => unreachable!(),
            Type::ConstArray(elem_type, length) => {
                let elem_length = type_set.calc_type_size(elem_type);
                let inside = (0..*length).map(|i| {
                    let from = i * elem_length;
                    let to = (i + 1) * elem_length;
                    elem_type.to_cells(quote! { #value[#i] }, quote! { &mut #result[#from..#to] })
                });
                quote! {
                    #(#inside)*
                }
            }
        };
        let function_name = ident(&format!(
            "{}_{}",
            TO_CELLS,
            type_to_identifier_trace_generation(self)
        ));
        let rust_type = type_to_rust(self);
        quote! {
            fn #function_name(#value: #rust_type, #result: &mut [F]) {
                #body
            }
        }
    }
}
