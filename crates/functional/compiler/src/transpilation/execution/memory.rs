use proc_macro2::TokenStream;
use quote::quote;

use crate::transpilation::execution::constants::*;

pub fn rust_memory() -> TokenStream {
    let memory_struct = memory_struct_name();
    let reference_struct = reference_type();
    let array_struct = array_type();

    let create_ref = ident(CREATE_REF);
    let dereference = ident(DEREFERENCE);
    let create_empty_under_construction_array = ident(CREATE_EMPTY_UNDER_CONSTRUCTION_ARRAY);
    let append_under_construction_array = ident(APPEND_UNDER_CONSTRUCTION_ARRAY);
    let array_access = ident(ARRAY_ACCESS);
    let zk_identifier = ident(ZK_IDENTIFIER);
    let get_reference_multiplicity = ident(GET_REFERENCE_MULTIPLICITY);
    let get_array_multiplicity = ident(GET_ARRAY_MULTIPLICITY);

    quote! {
        #[derive(Clone, Copy, Default, Debug)]
        pub struct #reference_struct {
            execution_index: usize,
            #zk_identifier: usize,
        }
        #[derive(Clone, Copy, Default, Debug)]
        pub struct #array_struct {
            execution_index: usize,
            #zk_identifier: usize,
        }

        #[derive(Default, Debug)]
        pub struct #memory_struct<T: Copy + Clone> {
            pub references: Vec<T>,
            pub reference_num_accesses: Vec<usize>,
            pub arrays: Vec<Vec<T>>,
            pub array_num_accesses: Vec<Vec<usize>>,
        }

        impl<T: Copy + Clone> #memory_struct<T> {
            pub fn #create_ref(&mut self, value: T, #zk_identifier: usize) -> #reference_struct {
                let index = self.references.len();
                self.references.push(value);
                self.reference_num_accesses.push(0);
                #reference_struct {
                    execution_index: index,
                    #zk_identifier,
                }
            }

            pub fn #dereference(&mut self, reference: #reference_struct) -> T {
                self.reference_num_accesses[reference.execution_index] += 1;
                self.references[reference.execution_index]
            }

            pub fn #create_empty_under_construction_array(&mut self, #zk_identifier: usize) -> #array_struct {
                let index = self.arrays.len();
                self.arrays.push(vec![]);
                self.array_num_accesses.push(vec![]);
                #array_struct {
                    execution_index: index,
                    #zk_identifier,
                }
            }

            pub fn #append_under_construction_array(&mut self, array: #array_struct, value: T) -> (usize, #array_struct) {
                self.arrays[array.execution_index].push(value);
                self.array_num_accesses[array.execution_index].push(0);
                (self.array_num_accesses[array.execution_index].len() - 1, array)
            }

            pub fn #array_access(&mut self, array: #array_struct, index: F) -> T {
                let index = index.as_canonical_u32() as usize;
                self.array_num_accesses[array.execution_index][index] += 1;
                self.arrays[array.execution_index][index]
            }

            pub fn #get_reference_multiplicity(&self, reference: #reference_struct) -> usize {
                self.reference_num_accesses[reference.execution_index]
            }

            pub fn #get_array_multiplicity(&self, array: #array_struct, index: usize) -> usize {
                self.array_num_accesses[array.execution_index][index]
            }
        }
    }
}
