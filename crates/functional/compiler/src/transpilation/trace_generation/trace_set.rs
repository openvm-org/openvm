use proc_macro2::TokenStream;
use quote::quote;

use crate::core::stage1::Stage2Program;
use crate::transpilation::execution::constants::*;
use crate::transpilation::trace_generation::constants::*;

pub fn rust_trace_set(program: &Stage2Program) -> TokenStream {
    let mut fields = vec![];
    for function_name in program.functions.keys() {
        let field_name = trace_set_field_name(function_name);
        fields.push(quote! {
            pub #field_name: Vec<F>,
        });
    }
    let struct_name = trace_set_struct_name();
    let struct_declaration = quote! {
        #[derive(Default, Debug)]
        pub struct #struct_name {
            #(#fields)*
        }
    };

    let mut init_calls = vec![];
    for function_name in program.functions.keys() {
        let field_name = trace_set_field_name(function_name);
        let function_struct_name = function_struct_name(function_name);
        let width = trace_width_const_name();
        let call_counter = ident(&format!("{}_{}", function_name, CALL_COUNTER));
        init_calls.push(quote! {
            #field_name: Self::init_trace(tracker.#call_counter, #function_struct_name::#width),
        });
    }

    let tracker_struct = tracker_struct_name();
    let impl_block = quote! {
        impl #struct_name {
            pub fn new(tracker: &#tracker_struct) -> Self {
                Self {
                    #(#init_calls)*
                }
            }
            pub fn init_trace(num_calls: usize, width: usize) -> Vec<F> {
                let height = num_calls.next_power_of_two();
                vec![F::ZERO; height * width]
            }
        }
    };

    quote! {
        #struct_declaration
        #impl_block
    }
}
