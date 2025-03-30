use proc_macro2::{Ident, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};

use crate::core::ir::{Type, BOOLEAN_TYPE_NAME};

pub fn ident(s: &str) -> TokenStream {
    TokenStream::from(TokenTree::Ident(Ident::new(s, Span::call_site())))
}

// types

pub fn field_type() -> TokenStream {
    ident("F")
}

pub fn reference_type() -> TokenStream {
    ident("TLRef")
}

pub fn array_type() -> TokenStream {
    ident("TLArray")
}

pub fn type_name(name: &str) -> TokenStream {
    if name == BOOLEAN_TYPE_NAME {
        quote! { bool }
    } else {
        ident(&format!("TL_{}", name))
    }
}

pub fn type_to_rust(tipo: &Type) -> TokenStream {
    match tipo {
        Type::Field => field_type(),
        Type::Reference(_) => reference_type(),
        Type::ReadablePrefix(..) => array_type(),
        Type::AppendablePrefix(..) => array_type(),
        Type::Array(..) => array_type(),
        Type::NamedType(name) => type_name(name),
        Type::Unmaterialized(inner) => type_to_rust(inner),
        Type::ConstArray(elem, length) => {
            let elem_type = type_to_rust(elem);
            quote! {
                [#elem_type; #length]
            }
        }
    }
}
pub fn type_to_identifier(tipo: &Type) -> String {
    match tipo {
        Type::Field => "F".to_string(),
        Type::Reference(inner) => format!("Ref_{}", type_to_identifier(inner)),
        Type::ReadablePrefix(inner, _) => format!("ReadablePrefix_{}", type_to_identifier(inner)),
        Type::AppendablePrefix(inner, _) => {
            format!("AppendablePrefix_{}", type_to_identifier(inner))
        }
        Type::Array(inner) => format!("Array_{}", type_to_identifier(inner)),
        Type::NamedType(name) => name.clone(),
        Type::Unmaterialized(inner) => type_to_identifier(inner),
        Type::ConstArray(elem, length) => {
            format!("ConstArray{}_{}", length, type_to_identifier(elem))
        }
    }
}

pub fn function_struct_name(name: &str) -> TokenStream {
    ident(&format!("TLFunction_{}", name))
}
pub fn memory_struct_name() -> TokenStream {
    quote! { Memory }
}
pub fn tracker_struct_name() -> TokenStream {
    quote! { Tracker }
}

// functions

pub const ISIZE_TO_FIELD_ELEM: &str = "isize_to_field_elem";
pub const TRACKER: &str = "tracker";
pub const MEMORY: &str = "memory";
pub const CREATE_REF: &str = "create_ref";
pub const DEREFERENCE: &str = "dereference";
pub const CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY: &str = "create_empty_under_construction_array";
pub const APPEND_UNDER_CONSTRUCTION_ARRAY: &str = "append_under_construction_array";
pub const ARRAY_ACCESS: &str = "array_access";
pub const STAGE: &str = "stage";

pub fn isize_to_field_elem(x: impl ToTokens) -> TokenStream {
    let function_name = ident(ISIZE_TO_FIELD_ELEM);
    quote! {
        #function_name(#x)
    }
}

pub fn create_ref(type_identifier: String, x: impl ToTokens) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", CREATE_REF, type_identifier));
    quote! {
        #tracker.#function_name(#x)
    }
}

pub fn dereference(tipo: &Type, x: impl ToTokens) -> TokenStream {
    let type_identifier = type_to_identifier(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(DEREFERENCE);
    quote! {
        #tracker.#memory.#function_name(#x)
    }
}

pub fn create_empty_under_construction_array(tipo: &Type) -> TokenStream {
    let type_identifier = type_to_identifier(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY);
    quote! {
        #tracker.#memory.#function_name()
    }
}

pub fn prepend_under_construction_array(
    tipo: &Type,
    old_array: impl ToTokens,
    elem: impl ToTokens,
) -> TokenStream {
    let type_identifier = type_to_identifier(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(APPEND_UNDER_CONSTRUCTION_ARRAY);
    quote! {
        #tracker.#memory.#function_name(#old_array, #elem)
    }
}

pub fn array_access(tipo: &Type, array: impl ToTokens, index: impl ToTokens) -> TokenStream {
    let type_identifier = type_to_identifier(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(ARRAY_ACCESS);
    quote! {
        #tracker.#memory.#function_name(#array, #index)
    }
}

pub fn execute_stage(callee: impl ToTokens, index: usize) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", STAGE, index));
    quote! {
        #callee.as_mut().as_mut().unwrap().#function_name(#tracker);
    }
}

pub fn define_stage(stage_index: usize, body: TokenStream) -> TokenStream {
    let tracker = ident(TRACKER);
    let tracker_struct = tracker_struct_name();
    let function_name = ident(&format!("{}_{}", STAGE, stage_index));
    quote! {
        pub fn #function_name(&mut self, #tracker: &mut #tracker_struct) {
            #body
        }
    }
}
