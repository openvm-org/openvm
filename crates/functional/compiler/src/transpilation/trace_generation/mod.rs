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
        let mut declaration_types = vec![];
        for function in self.functions.values() {
            declaration_types.extend(function.get_declaration_types());
            let (representation_table, air_constructor) = air_set.get_air(&function.name);
            functions.push(function.transpile_trace_generation_impl(
                representation_table,
                air_constructor,
                self,
            ));
        }
        
        let mut distinct_types = vec![];
        for tipo in declaration_types {
            if self.types.calc_type_size(&tipo) > 0
                && !distinct_types
                .iter()
                .any(|other| tipo.eq(other))
            {
                distinct_types.push(tipo);
            }
        }
        
        let mut type_conversions = vec![];
        for tipo in distinct_types {
            type_conversions.push(tipo.transpile_to_cells(&self.types));
        }
        
        let trace_set = rust_trace_set(self);
        let max_trace_height_name = max_trace_height_const_name();
        let max_trace_height_value = AirConstructor::MAX_HEIGHT;
        quote! {
            const #max_trace_height_name: usize = #max_trace_height_value;
            #trace_set
            #(#functions)*
            #(#type_conversions)*
        }
    }
}
