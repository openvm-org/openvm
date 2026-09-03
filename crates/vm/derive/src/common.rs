use quote::{format_ident, quote};
use syn::{FnArg, GenericArgument, Ident, ItemFn, PathArguments, Type};

/// Extract the execution context type parameter from the function's `VmExecState` argument.
///
/// Looking at the argument type is more robust than relying on generic parameter order or on a
/// particular trait-bound spelling. In particular, bounds may be written in a `where` clause.
pub fn extract_ctx_type(input_fn: &ItemFn) -> syn::Result<Ident> {
    let ctx_type = input_fn
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(arg) => Some(arg.ty.as_ref()),
            FnArg::Receiver(_) => None,
        })
        .find_map(|ty| {
            let Type::Reference(reference) = ty else {
                return None;
            };
            reference.mutability?;
            let Type::Path(state_type) = reference.elem.as_ref() else {
                return None;
            };
            let state_segment = state_type.path.segments.last()?;
            if state_segment.ident != "VmExecState" {
                return None;
            }
            let PathArguments::AngleBracketed(arguments) = &state_segment.arguments else {
                return None;
            };
            match arguments.args.iter().nth(1)? {
                GenericArgument::Type(ty) => Some(ty),
                _ => None,
            }
        })
        .ok_or_else(|| {
            syn::Error::new_spanned(
                &input_fn.sig,
                "create_handler requires a `&mut VmExecState<_, Ctx>` argument",
            )
        })?;

    let Type::Path(ctx_path) = ctx_type else {
        return Err(syn::Error::new_spanned(
            ctx_type,
            "the VmExecState context must be a generic type parameter",
        ));
    };
    let Some(ctx_ident) = ctx_path.path.get_ident() else {
        return Err(syn::Error::new_spanned(
            ctx_type,
            "the VmExecState context must be a generic type parameter",
        ));
    };
    if !input_fn
        .sig
        .generics
        .type_params()
        .any(|param| param.ident == *ctx_ident)
    {
        return Err(syn::Error::new_spanned(
            ctx_type,
            "the VmExecState context must be a generic type parameter",
        ));
    }

    Ok(ctx_ident.clone())
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

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::extract_ctx_type;

    #[test]
    fn extracts_context_without_a_field_parameter() {
        let function = parse_quote! {
            unsafe fn execute_impl<Ctx: ExecutionCtxTrait>(
                pre_compute: *const u8,
                state: &mut VmExecState<GuestMemory, Ctx>,
            ) {}
        };

        assert_eq!(extract_ctx_type(&function).unwrap(), "Ctx");
    }

    #[test]
    fn extracts_context_independent_of_order_and_bound_location() {
        let function = parse_quote! {
            unsafe fn execute_impl<Op, Ctx, F>(
                pre_compute: *const u8,
                state: &mut crate::arch::VmExecState<GuestMemory, Ctx>,
            ) where
                Ctx: MeteredExecutionCtxTrait,
            {}
        };

        assert_eq!(extract_ctx_type(&function).unwrap(), "Ctx");
    }

    #[test]
    fn rejects_a_non_generic_context() {
        let function = parse_quote! {
            unsafe fn execute_impl(
                pre_compute: *const u8,
                state: &mut VmExecState<GuestMemory, ConcreteCtx>,
            ) {}
        };

        assert!(extract_ctx_type(&function).is_err());
    }

    #[test]
    fn rejects_an_immutable_execution_state() {
        let function = parse_quote! {
            unsafe fn execute_impl<Ctx>(
                pre_compute: *const u8,
                state: &VmExecState<GuestMemory, Ctx>,
            ) {}
        };

        assert!(extract_ctx_type(&function).is_err());
    }
}
