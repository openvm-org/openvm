use proc_macro2::{Ident, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};

use crate::{
    execution::transpilation::VariableNamer,
    folder1::{file3::FlattenedFunction, function_resolution::Stage, ir::Type},
};

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

pub fn under_construction_array_type() -> TokenStream {
    ident("UnderConstructionArray")
}

pub fn type_name(name: &str) -> TokenStream {
    ident(&format!("TL_{}", name))
}

pub fn type_to_rust(tipo: &Type) -> TokenStream {
    match tipo {
        Type::Field => field_type(),
        Type::Reference(_) => reference_type(),
        Type::Array(_) => array_type(),
        Type::UnderConstructionArray(_) => under_construction_array_type(),
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
        Type::Array(inner) => format!("Array_{}", type_to_identifier(inner)),
        Type::UnderConstructionArray(inner) => {
            format!("UnderConstructionArray_{}", type_to_identifier(inner))
        }
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
pub const EQ_TO_BOOL: &str = "eq_to_bool";
pub const TRACKER: &str = "tracker";
pub const MEMORY: &str = "memory";
pub const CREATE_REF: &str = "create_ref";
pub const DEREFERENCE: &str = "dereference";
pub const CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY: &str = "create_empty_under_construction_array";
pub const APPEND_UNDER_CONSTRUCTION_ARRAY: &str = "append_under_construction_array";
pub const FINALIZE_ARRAY: &str = "finalize_array";
pub const ARRAY_ACCESS: &str = "array_access";
pub const STAGE: &str = "stage";

pub fn isize_to_field_elem(x: impl ToTokens) -> TokenStream {
    let function_name = ident(ISIZE_TO_FIELD_ELEM);
    quote! {
        #function_name(#x)
    }
}

pub fn eq_to_bool(left: impl ToTokens, right: impl ToTokens) -> TokenStream {
    let function_name = ident(EQ_TO_BOOL);
    quote! {
        #function_name(#left, #right)
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

pub fn finalize_array(tipo: &Type, under_construction: impl ToTokens) -> TokenStream {
    let type_identifier = type_to_identifier(tipo);
    let tracker = ident(TRACKER);
    let memory = ident(&format!("{}_{}", MEMORY, type_identifier));
    let function_name = ident(FINALIZE_ARRAY);
    quote! {
        #tracker.#memory.#function_name(#under_construction)
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

pub fn execute_stage(
    callee: impl ToTokens,
    in_arguments: &Vec<TokenStream>,
    out_arguments: &Vec<TokenStream>,
    index: usize,
) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", STAGE, index));
    quote! {
        let (#(#out_arguments),*) = #callee.#function_name(#tracker, #(#in_arguments),*);
    }
}

pub fn define_stage(
    function: &FlattenedFunction,
    stage_index: usize,
    namer: &VariableNamer,
    body: TokenStream,
) -> TokenStream {
    let tracker = ident(TRACKER);
    let tracker_struct = tracker_struct_name();
    let function_name = ident(&format!("{}_{}", STAGE, stage_index));

    let stage = function.stages[stage_index];

    let in_argument_names = (stage.start..stage.mid).map(|index| namer.in_argument_name(index));
    let in_argument_types = function.arguments[stage.start..stage.mid]
        .iter()
        .map(|argument| type_to_rust(argument.get_type()));
    let out_argument_types = function.arguments[stage.mid..stage.end]
        .iter()
        .map(|argument| type_to_rust(argument.get_type()));
    quote! {
        fn #function_name(#tracker: &mut #tracker_struct, #(#in_argument_names: #in_argument_types),*) -> (#(#out_argument_types),*) {
            #body
        }
    }
}
