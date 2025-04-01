use crate::air::constructor::{AirConstructor, AirSet};
use crate::core::stage1::Stage2Program;
use crate::transpilation::trace_generation::constants::max_trace_height_const_name;
use crate::transpilation::trace_generation::trace_set::rust_trace_set;
use proc_macro2::TokenStream;
use quote::quote;

pub mod constants;
pub mod function;
pub mod trace_set;
pub mod types;

impl Stage2Program {
    pub fn transpile_trace_generation(&self, air_set: &AirSet) -> TokenStream {
        let mut functions = vec![];
        for function in self.functions.values() {
            let (representation_table, air_constructor) = air_set.get_air(&function.name);
            functions.push(function.transpile_trace_generation_impl(
                representation_table,
                air_constructor,
                self,
            ));
        }
        let trace_set = rust_trace_set(self);
        let max_trace_height_name = max_trace_height_const_name();
        let max_trace_height_value = AirConstructor::MAX_HEIGHT;
        quote! {
            const #max_trace_height_name: usize = #max_trace_height_value;
            #trace_set
            #(#functions)*
        }
    }
}
