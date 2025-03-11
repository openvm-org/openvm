use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    execution::constants::*,
    folder1::{ir::Type, stage1::Stage2Program},
};
use crate::execution::helpers::rust_helpers;
use crate::execution::memory::rust_memory;

pub fn rust_tracker(types_in_memory: Vec<Type>) -> TokenStream {
    let mut distinct_types = vec![];
    for tipo in types_in_memory {
        if !distinct_types
            .iter()
            .any(|other| tipo.eq_unmaterialized(other))
        {
            distinct_types.push(tipo);
        }
    }

    let mut fields = vec![];
    for tipo in distinct_types {
        let identifier = type_to_identifier(&tipo);
        let field_name = ident(&format!("{}_{}", MEMORY, identifier));
        let rust_type = type_to_rust(&tipo);
        let memory = memory_struct_name();
        fields.push(quote! {
            pub #field_name: #memory<#rust_type>,
        });
    }

    quote! {
        #[derive(Default, Debug)]
        pub struct Tracker {
            #(#fields)*
        }
    }
}

impl Stage2Program {
    pub fn transpile(&self) -> TokenStream {
        let mut functions = vec![];
        let mut types_in_memory = vec![];
        for function in self.functions.values() {
            functions.push(function.transpile(self));
            types_in_memory.extend(function.get_types_in_memory());
        }
        let mut algebraic_types = vec![];
        for algebraic_type in self.types.algebraic_types.values() {
            let mut variants = vec![];
            for variant in algebraic_type.variants.iter() {
                let name = ident(&variant.name);
                let rust_components = variant.components.iter().map(type_to_rust);
                variants.push(quote! {
                    #name(#(#rust_components),*),
                });
            }
            let name = type_name(&algebraic_type.name);
            algebraic_types.push(quote! {
                #[derive(PartialEq, Eq, Debug, Clone, Copy)]
                enum #name {
                    #(#variants)*
                }
            });
        }
        let tracker = rust_tracker(types_in_memory);
        let memory = rust_memory();
        let helpers = rust_helpers();
        
        let field_type = field_type();
        quote! {
            use std::ops::Neg;
            use openvm_stark_sdk::openvm_stark_backend::p3_field::FieldAlgebra;
            type #field_type = openvm_stark_sdk::p3_baby_bear::BabyBear;

            #tracker
            #memory
            #helpers
            #(#algebraic_types)*
            #(#functions)*
        }
    }
}
