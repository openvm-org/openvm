extern crate alloc;
extern crate proc_macro;

use itertools::multiunzip;
use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, Fields};

#[proc_macro_derive(InstructionExecutor)]
pub fn instruction_executor_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let generics = &ast.generics;
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    match &ast.data {
        Data::Struct(inner) => {
            // Check if the struct has only one unnamed field
            let inner_ty = match &inner.fields {
                Fields::Unnamed(fields) => {
                    if fields.unnamed.len() != 1 {
                        panic!("Only one unnamed field is supported");
                    }
                    fields.unnamed.first().unwrap().ty.clone()
                }
                _ => panic!("Only unnamed fields are supported"),
            };
            // Assume F is already generic of the field.
            // Also assume this is used in VM crate.
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause
                .predicates
                .push(syn::parse_quote! { #inner_ty: crate::arch::InstructionExecutor<F> });
            quote! {
                impl #impl_generics crate::arch::InstructionExecutor<F> for #name #ty_generics #where_clause {
                    fn execute(
                        &mut self,
                        instruction: axvm_instructions::instruction::Instruction<F>,
                        from_state: crate::arch::ExecutionState<u32>,
                    ) -> crate::arch::Result<crate::arch::ExecutionState<u32>> {
                        self.0.execute(instruction, from_state)
                    }

                    fn get_opcode_name(&self, opcode: usize) -> String {
                        self.0.get_opcode_name(opcode)
                    }
                }
            }
            .into()
        }
        Data::Enum(e) => {
            let variants = e
                .variants
                .iter()
                .map(|variant| {
                    let variant_name = &variant.ident;

                    let mut fields = variant.fields.iter();
                    let field = fields.next().unwrap();
                    assert!(fields.next().is_none(), "Only one field is supported");
                    (variant_name, field)
                })
                .collect::<Vec<_>>();
            let (execute_arms, get_opcode_name_arms): (Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                    let field_ty = &field.ty;
                    let execute_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as crate::arch::InstructionExecutor<F>>::execute(x, instruction, from_state)
                    };
                    let get_opcode_name_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as crate::arch::InstructionExecutor<F>>::get_opcode_name(x, opcode)
                    };

                    (execute_arm, get_opcode_name_arm)
                }));
            quote! {
                impl #impl_generics crate::arch::InstructionExecutor<F> for #name #ty_generics {
                    fn execute(
                        &mut self,
                        instruction: axvm_instructions::instruction::Instruction<F>,
                        from_state: crate::arch::ExecutionState<u32>,
                    ) -> crate::arch::Result<crate::arch::ExecutionState<u32>> {
                        match self {
                            #(#execute_arms,)*
                        }
                    }

                    fn get_opcode_name(&self, opcode: usize) -> String {
                        match self {
                            #(#get_opcode_name_arms,)*
                        }
                    }
                }
            }
            .into()
        }
        Data::Union(_) => unimplemented!("Unions are not supported"),
    }
}
