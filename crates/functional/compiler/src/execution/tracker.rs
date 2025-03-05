use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    execution::constants::*,
    folder1::{ir::Type, stage1::Stage2Program},
};

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
        let tracker = rust_tracker(types_in_memory);
        quote! {
            #tracker
            #(#functions)*
        }
    }
}
