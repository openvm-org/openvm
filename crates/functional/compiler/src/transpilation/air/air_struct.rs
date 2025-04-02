use crate::air::constructor::AirConstructor;
use crate::transpilation::execution::constants::ident;
use proc_macro2::TokenStream;
use quote::quote;

pub fn air_struct_name(name: &str) -> TokenStream {
    ident(&format!("Air_{}", name))
}

impl AirConstructor {
    pub fn transpile_air_struct(&self, name: &str) -> TokenStream {
        let struct_name = air_struct_name(name);
        let width = self.width();
        let constraints = self.transpile_constraints();
        quote! {
            #[derive(Clone, Copy, Debug, Default)]
            pub struct #struct_name;

            impl<F: Field> BaseAir<F> for #struct_name {
                fn width(&self) -> usize {
                    #width
                }
            }

            impl<F: Field> BaseAirWithPublicValues<F> for #struct_name {}
            impl<F: Field> PartitionedBaseAir<F> for #struct_name {}

            impl<AB: InteractionBuilder> Air<AB> for #struct_name {
                fn eval(&self, builder: &mut AB) {
                    #constraints
                }
            }
        }
    }
}
