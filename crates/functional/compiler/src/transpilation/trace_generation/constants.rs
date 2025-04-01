use crate::transpilation::execution::constants::ident;
use proc_macro2::TokenStream;
use quote::quote;

pub fn trace_set_struct_name() -> TokenStream {
    quote! { TraceSet }
}

pub fn trace_width_const_name() -> TokenStream {
    quote! { TRACE_WIDTH }
}

pub fn num_references_const_name() -> TokenStream {
    quote! { NUM_REFERENCES }
}

pub fn trace_set_field_name(function_name: &str) -> TokenStream {
    ident(&format!("{}_trace", function_name))
}

pub fn generate_trace_function_name() -> TokenStream {
    quote! { generate_trace }
}

pub fn max_trace_height_const_name() -> TokenStream {
    quote! { MAX_TRACE_HEIGHT }
}
