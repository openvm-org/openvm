use proc_macro2::TokenStream;
use quote::quote;

use crate::execution::constants::*;

pub fn rust_memory() -> TokenStream {
    let memory_struct = memory_struct_name();
    let reference_struct = reference_type();
    let array_struct = array_type();

    let create_ref = ident(CREATE_REF);
    let dereference = ident(DEREFERENCE);
    let create_empty_under_construction_array = ident(CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY);
    let append_under_construction_array = ident(APPEND_UNDER_CONSTRUCTION_ARRAY);
    let finalize_array = ident(FINALIZE_ARRAY);
    let array_access = ident(ARRAY_ACCESS);

    quote! {
        #[derive(Clone, Copy, Debug)]
        pub struct #reference_struct(usize);
        #[derive(Clone, Copy, Debug)]
        pub struct #array_struct(usize);

        #[derive(Default, Debug)]
        pub struct #memory_struct<T: Copy + Clone> {
            pub references: Vec<T>,
            pub reference_timestamps: Vec<usize>,
            pub arrays: Vec<Vec<T>>,
            pub array_timestamps: Vec<usize>,
        }

        impl<T: Copy + Clone> #memory_struct<T> {
            pub fn #create_ref(&mut self, value: T) -> #reference_struct {
                let index = self.references.len();
                self.references.push(value);
                self.reference_timestamps.push(0);
                #reference_struct(index)
            }

            pub fn #dereference(&self, reference: #reference_struct) -> T {
                self.references[reference.0]
            }

            pub fn #create_empty_under_construction_array(&mut self) -> #array_struct {
                let index = self.arrays.len();
                self.arrays.push(vec![]);
                self.array_timestamps.push(0);
                #array_struct(index)
            }

            pub fn #append_under_construction_array(&mut self, array: #array_struct, value: T) {
                self.arrays[array.0].push(value);
            }

            pub fn #finalize_array(&mut self, array: #array_struct) -> #array_struct {
                array
            }

            pub fn #array_access(&self, array: #array_struct, index: usize) -> T {
                self.arrays[array.0][index]
            }
        }
    }
}
