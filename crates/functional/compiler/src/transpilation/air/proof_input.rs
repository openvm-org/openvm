use proc_macro2::TokenStream;
use quote::quote;

use crate::core::stage1::Stage2Program;
use crate::transpilation::air::air_struct::air_struct_name;
use crate::transpilation::execution::constants::{function_struct_name, ident};
use crate::transpilation::trace_generation::constants::{
    trace_set_field_name, trace_set_struct_name, trace_width_const_name,
};

pub fn rust_proof_input(program: &Stage2Program) -> TokenStream {
    let trace_set_struct_name = trace_set_struct_name();
    let trace_set = quote! { trace_set };
    let airs = quote! { airs };
    let inputs = quote! { inputs };

    let mut functions = vec![];
    for function_name in program.functions.keys() {
        let function_struct_name = function_struct_name(function_name);
        let air_struct_name = air_struct_name(function_name);
        let trace_set_field_name = trace_set_field_name(function_name);
        let width = trace_width_const_name();
        functions.push(quote! {
            if !#trace_set.#trace_set_field_name.is_empty() {
                #airs.push(Arc::new(#air_struct_name::default()));
                #inputs.push(AirProofInput::simple_no_pis(RowMajorMatrix::new(#trace_set.#trace_set_field_name, #function_struct_name::#width)));
            }
        });
    }

    quote! {
        use std::sync::Arc;
        use openvm_stark_backend::{
            prover::types::AirProofInput, AirRef,
        };
        use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;
        use openvm_stark_sdk::p3_blake3::Blake3;

        type SC = openvm_stark_sdk::config::baby_bear_bytehash::BabyBearByteHashConfig<Blake3>;

        pub struct ProofInput {
            pub #airs: Vec<AirRef<SC>>,
            pub #inputs: Vec<AirProofInput<SC>>,
        }
        impl ProofInput {
            pub fn new(#trace_set: #trace_set_struct_name) -> Self {
                let mut #airs: Vec<AirRef<SC>> = vec![];
                let mut #inputs: Vec<AirProofInput<SC>> = vec![];
                #(#functions)*
                Self { #airs, #inputs }
            }
        }
    }
}
