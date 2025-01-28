use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Used for defining the guest's entrypoint and main function.
///
/// When `#![no_main]` is used, the programs entrypoint and main function is left undefined. The
/// `entry` attribute is required to indicate the main function and link it to an entrypoint provided
/// by the `openvm` crate.
///
/// When `std` is enabled, the entrypoint will be linked automatically and this macro is not
/// required.
///
/// # Example
///
/// ```ignore
/// #![no_main]
/// #![no_std]
///
/// #[openvm::main]
/// fn main() { }
/// ```
#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let fn_name = &input.sig.ident;
    let fn_block = &input.block;
    let fn_attrs = &input.attrs;
    let fn_vis = &input.vis;
    let fn_sig = &input.sig;

    // Here, implement the transformation logic you need.
    // For demonstration, let's assume we want to print a message before the function executes.

    let expanded = quote! {
        // Type check the given path
        #[cfg(all(not(feature = "std"), target_os = "zkvm"))]
        const ZKVM_ENTRY: fn() = #fn_name;

        // Include generated main in a module so we don't conflict
        // with any other definitions of "main" in this file.
        #[cfg(all(not(feature = "std"), target_os = "zkvm"))]
        mod zkvm_generated_main {
            #[no_mangle]
            fn main() {
                super::ZKVM_ENTRY()
            }
        }

        #(#fn_attrs)*
        #fn_vis #fn_sig {
            #fn_block
        }
    };

    TokenStream::from(expanded)
}
