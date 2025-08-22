use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, ItemFn};

/// Implementation of the TCO handler generation logic.
/// This is called from the proc macro attribute in lib.rs.
pub fn tco_impl(item: TokenStream) -> TokenStream {
    // Parse the input function
    let input_fn = parse_macro_input!(item as ItemFn);

    // Extract information from the function
    let fn_name = &input_fn.sig.ident;
    let generics = &input_fn.sig.generics;
    let where_clause = &generics.where_clause;

    // Extract the first two generic type parameters (F and CTX)
    let (f_type, ctx_type) = extract_f_and_ctx_types(generics);
    // Derive new function name:
    // If original ends with `_impl`, replace with `_tco_handler`, else append suffix.
    let new_name_str = fn_name
        .to_string()
        .strip_suffix("_impl")
        .map(|base| format!("{base}_tco_handler"))
        .unwrap_or_else(|| format!("{fn_name}_tco_handler"));
    let handler_name = format_ident!("{}", new_name_str);

    // Build the generic parameters for the handler, preserving all original generics
    let handler_generics = generics.clone();

    // Build the function call with all the generics
    let generic_args = build_generic_args(generics);
    let execute_call = if generic_args.is_empty() {
        quote! { #fn_name(pre_compute, exec_state) }
    } else {
        quote! { #fn_name::<#(#generic_args),*>(pre_compute, exec_state) }
    };

    // Generate the TCO handler function
    let handler_fn = quote! {
        #[inline(never)]
        unsafe fn #handler_name #handler_generics (
            interpreter: &::openvm_circuit::arch::interpreter::InterpretedInstance<#f_type, #ctx_type>,
            exec_state: &mut ::openvm_circuit::arch::VmExecState<
                #f_type,
                ::openvm_circuit::system::memory::online::GuestMemory,
                #ctx_type,
            >,
        )
        #where_clause
        {
            use ::openvm_circuit::arch::ExecutionError;

            let pre_compute = interpreter.get_pre_compute(exec_state.vm_state.pc);
            #execute_call;

            if exec_state.exit_code.is_err() {
                // stop execution
                return;
            }
            if #ctx_type::should_suspend(exec_state) {
                return;
            }
            // exec_state.pc should have been updated by execute_impl at this point
            let next_handler = interpreter.get_handler(exec_state.vm_state.pc);
            if next_handler.is_none() {
                exec_state.exit_code = Err(ExecutionError::PcOutOfBounds (exec_state.vm_state.pc));
                return;
            }
            let next_handler = next_handler.unwrap_unchecked();

            // NOTE: `become` is a keyword that requires Rust Nightly.
            // It is part of the explicit tail calls RFC: <https://github.com/rust-lang/rust/issues/112788>
            // which is still incomplete.
            become next_handler(interpreter, exec_state)
        }
    };

    // Return both the original function and the new handler
    let output = quote! {
        #input_fn

        #handler_fn
    };

    TokenStream::from(output)
}

fn extract_f_and_ctx_types(generics: &syn::Generics) -> (syn::Ident, syn::Ident) {
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

fn build_generic_args(generics: &syn::Generics) -> Vec<proc_macro2::TokenStream> {
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
