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
pub fn type_to_identifier_execution(tipo: &Type) -> String {
    type_to_identifier(tipo, true)
}

pub fn type_to_identifier_trace_generation(tipo: &Type) -> String {
    type_to_identifier(tipo, false)
}

fn type_to_identifier(tipo: &Type, pass_through_unmaterialized: bool) -> String {
    match tipo {
        Type::Field => "F".to_string(),
        Type::Reference(inner) => format!(
            "Ref_{}",
            type_to_identifier(inner, pass_through_unmaterialized)
        ),
        Type::ReadablePrefix(inner, _) => format!(
            "ReadablePrefix_{}",
            type_to_identifier(inner, pass_through_unmaterialized)
        ),
        Type::AppendablePrefix(inner, _) => {
            format!(
                "AppendablePrefix_{}",
                type_to_identifier(inner, pass_through_unmaterialized)
            )
        }
        Type::Array(inner) => format!(
            "Array_{}",
            type_to_identifier(inner, pass_through_unmaterialized)
        ),
        Type::NamedType(name) => name.clone(),
        Type::Unmaterialized(inner) => {
            let inner_id = type_to_identifier(inner, pass_through_unmaterialized);
            if pass_through_unmaterialized {
                inner_id
            } else {
                format!("Unmaterialized_{}", inner_id)
            }
        }
        Type::ConstArray(elem, length) => {
            format!(
                "ConstArray{}_{}",
                length,
                type_to_identifier(elem, pass_through_unmaterialized)
            )
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
pub const CALL_COUNTER: &str = "call_counter";
pub const CREATE_REF: &str = "create_ref";
pub const DEREFERENCE: &str = "dereference";
pub const CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY: &str = "create_empty_under_construction_array";
pub const APPEND_UNDER_CONSTRUCTION_ARRAY: &str = "append_under_construction_array";
pub const ARRAY_ACCESS: &str = "array_access";
pub const STAGE: &str = "stage";
pub const FUNCTION_ID: &str = "FUNCTION_ID";
pub const ZK_IDENTIFIER: &str = "zk_identifier";
pub const CALC_ZK_IDENTIFIER: &str = "calc_zk_identifier";

pub const GET_REFERENCE_MULTIPLICITY: &str = "get_reference_multiplicity";
pub const GET_ARRAY_MULTIPLICITY: &str = "get_array_multiplicity";

pub fn isize_to_field_elem(x: impl ToTokens) -> TokenStream {
    let function_name = ident(ISIZE_TO_FIELD_ELEM);
    quote! {
        #function_name(#x)
    }
}

pub fn usize_to_field_elem(x: impl ToTokens) -> TokenStream {
    quote! {
        F::from_canonical_usize(#x)
    }
}

pub fn create_ref(
    type_identifier: String,
    x: impl ToTokens,
    zk_identifier: TokenStream,
) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", CREATE_REF, type_identifier));
    quote! {
        #tracker.#function_name(#x, #zk_identifier)
    }
}

pub fn dereference(tipo: &Type, x: impl ToTokens) -> TokenStream {
    let type_identifier = type_to_identifier_execution(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(DEREFERENCE);
    quote! {
        #tracker.#memory.#function_name(#x)
    }
}

pub fn create_empty_under_construction_array(
    tipo: &Type,
    zk_identifier: TokenStream,
) -> TokenStream {
    let type_identifier = type_to_identifier_execution(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY);
    quote! {
        #tracker.#memory.#function_name(#zk_identifier)
    }
}

pub fn append_under_construction_array(
    tipo: &Type,
    old_array: impl ToTokens,
    elem: impl ToTokens,
) -> TokenStream {
    let type_identifier = type_to_identifier_execution(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(APPEND_UNDER_CONSTRUCTION_ARRAY);
    quote! {
        #tracker.#memory.#function_name(#old_array, #elem)
    }
}

pub fn array_access(tipo: &Type, array: impl ToTokens, index: impl ToTokens) -> TokenStream {
    let type_identifier = type_to_identifier_execution(tipo);
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

pub fn get_call_index(function_name: &String, function_field: TokenStream) -> TokenStream {
    let tracker = ident(TRACKER);
    let tracker_field = ident(&format!("{}_{}", function_name, CALL_COUNTER));
    quote! {
        #function_field = #tracker.#tracker_field;
        #tracker.#tracker_field += 1;
    }
}

pub fn calc_zk_identifier(statement_index: usize) -> TokenStream {
    let function_name = ident(CALC_ZK_IDENTIFIER);
    quote! {
        self.#function_name(#statement_index)
    }
}
