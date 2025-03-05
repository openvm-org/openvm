use proc_macro2::{Ident, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};

pub fn ident(s: &str) -> TokenStream {
    TokenStream::from(TokenTree::Ident(Ident::new(s, Span::call_site())))
}

// functions

const ISIZE_TO_FIELD_ELEM: &str = "isize_to_field_elem";
const EQ_TO_BOOL: &str = "eq_to_bool";
const TRACKER: &str = "tracker";
const CREATE_REF: &str = "create_ref";
const DEREFERENCE: &str = "dereference";
const CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY: &str = "create_empty_under_construction_array";
const APPEND_UNDER_CONSTRUCTION_ARRAY: &str = "append_under_construction_array";
const FINALIZE_ARRAY: &str = "finalize_array";
const ARRAY_ACCESS: &str = "array_access";

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

pub fn dereference(type_identifier: String, x: impl ToTokens) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", DEREFERENCE, type_identifier));
    quote! {
        #tracker.#function_name(#x)
    }
}

pub fn create_empty_under_construction_array(type_identifier: String) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!(
        "{}_{}",
        CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY, type_identifier
    ));
    quote! {
        #tracker.#function_name()
    }
}

pub fn prepend_under_construction_array(
    type_identifier: String,
    old_array: impl ToTokens,
    elem: impl ToTokens,
) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!(
        "{}_{}",
        APPEND_UNDER_CONSTRUCTION_ARRAY, type_identifier
    ));
    quote! {
        #tracker.#function_name(#old_array, #elem)
    }
}

pub fn finalize_array(type_identifier: String, under_construction: impl ToTokens) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", FINALIZE_ARRAY, type_identifier));
    quote! {
        #tracker.#function_name(#under_construction)
    }
}

pub fn array_access(
    type_identifier: String,
    array: impl ToTokens,
    index: impl ToTokens,
) -> TokenStream {
    let tracker = ident(TRACKER);
    let function_name = ident(&format!("{}_{}", ARRAY_ACCESS, type_identifier));
    quote! {
        #tracker.#function_name(#array, #index)
    }
}
