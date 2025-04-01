use proc_macro2::TokenStream;
use quote::quote;

use crate::transpilation::execution::constants::{field_type, ident, ISIZE_TO_FIELD_ELEM};

pub fn rust_helpers() -> TokenStream {
    let isize_to_field_elem = ident(ISIZE_TO_FIELD_ELEM);

    let field_type = field_type();

    quote! {
        pub fn #isize_to_field_elem(x: isize) -> #field_type {
            let base = #field_type::from_canonical_usize(x.unsigned_abs());
            if x >= 0 {
                base
            } else {
                base.neg()
            }
        }
    }
}
