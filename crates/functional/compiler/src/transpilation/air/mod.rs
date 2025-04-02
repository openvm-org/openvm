use crate::air::constructor::AirSet;
use proc_macro2::TokenStream;
use quote::quote;

pub mod air_struct;
pub mod constraints;
pub mod proof_input;

impl<'a> AirSet<'a> {
    pub fn transpile_airs(&self) -> TokenStream {
        let mut airs = vec![];
        for (name, (_, air_constructor)) in self.airs.iter() {
            airs.push(air_constructor.transpile_air_struct(name));
        }
        quote! {
            use openvm_stark_backend::{
                interaction::InteractionBuilder,
                p3_air::{Air, AirBuilder, BaseAir},
                p3_field::Field,
                p3_matrix::Matrix,
        rap::{BaseAirWithPublicValues, PartitionedBaseAir},
            };
                        #(#airs)*
                    }
    }
}
