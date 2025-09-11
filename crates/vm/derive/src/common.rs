use quote::{format_ident, quote};
use syn::{Ident, ItemFn};

/// Extract the first two generic type parameters (F and CTX) from function generics
pub fn extract_f_and_ctx_types(generics: &syn::Generics) -> (syn::Ident, syn::Ident) {
    let mut type_params = generics.params.iter().filter_map(|param| {
        if let syn::GenericParam::Type(type_param) = param {
            Some(&type_param.ident)
        } else {
            None
        }
    });

    let f_type = type_params
        .next()
        .expect("Function must have at least one type parameter (F)")
        .clone();
    let ctx_type = type_params
        .next()
        .expect("Function must have at least two type parameters (F and CTX)")
        .clone();

    (f_type, ctx_type)
}

/// Build a list of generic arguments for function calls
pub fn build_generic_args(generics: &syn::Generics) -> Vec<proc_macro2::TokenStream> {
    generics
        .params
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(type_param) => {
                let ident = &type_param.ident;
                quote! { #ident }
            }
            syn::GenericParam::Lifetime(lifetime) => {
                let lifetime = &lifetime.lifetime;
                quote! { #lifetime }
            }
            syn::GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                quote! { #ident }
            }
        })
        .collect()
}

/// Generate handler name from function name:
/// If original ends with `_impl`, replace with `_handler`, else append `_handler` suffix.
pub fn handler_name_from_fn(fn_name: &Ident) -> Ident {
    let new_name_str = fn_name
        .to_string()
        .strip_suffix("_impl")
        .map(|base| format!("{base}_handler"))
        .unwrap_or_else(|| format!("{fn_name}_handler"));
    format_ident!("{}", new_name_str)
}

/// Check if function returns Result type
pub fn returns_result_type(input_fn: &ItemFn) -> bool {
    match &input_fn.sig.output {
        syn::ReturnType::Type(_, ty) => {
            matches!(**ty, syn::Type::Path(ref path) if path.path.segments.last().is_some_and(|seg| seg.ident == "Result"))
        }
        _ => false,
    }
}
