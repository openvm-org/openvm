use proc_macro2::TokenStream;
use quote::quote;

use crate::execution::constants::{field_type, ident, type_name, EQ_TO_BOOL, ISIZE_TO_FIELD_ELEM};

pub fn rust_helpers() -> TokenStream {
    let isize_to_field_elem = ident(ISIZE_TO_FIELD_ELEM);
    let eq_to_bool = ident(EQ_TO_BOOL);

    let field_type = field_type();
    let bool_type = type_name("Bool");

    quote! {
        pub fn #isize_to_field_elem(x: isize) -> #field_type {
            let base = #field_type::from_canonical_usize(x.unsigned_abs());
            if x >= 0 {
                base
            } else {
                base.neg()
            }
        }

        pub fn #eq_to_bool<T: Eq>(x: T, y: T) -> #bool_type {
            if x == y {
                #bool_type::True()
            } else {
                #bool_type::False()
            }
        }
    }
}
