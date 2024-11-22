extern crate alloc;
extern crate proc_macro;

use itertools::multiunzip;
use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, Fields, GenericParam};

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
            // Use full path ::axvm_circuit... so it can be used either within or outside the vm crate.
            // Assume F is already generic of the field.
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause.predicates.push(
                syn::parse_quote! { #inner_ty: ::axvm_circuit::arch::InstructionExecutor<F> },
            );
            quote! {
                impl #impl_generics crate::arch::InstructionExecutor<F> for #name #ty_generics #where_clause {
                    fn execute(
                        &mut self,
                        instruction: ::axvm_circuit::arch::instructions::instruction::Instruction<F>,
                        from_state: ::axvm_circuit::arch::ExecutionState<u32>,
                    ) -> ::axvm_circuit::arch::Result<::axvm_circuit::arch::ExecutionState<u32>> {
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
            let first_ty_generic = ast
                .generics
                .params
                .first()
                .and_then(|param| match param {
                    GenericParam::Type(type_param) => Some(&type_param.ident),
                    _ => None,
                })
                .expect("First generic must be type for Field");
            // Use full path ::axvm_circuit... so it can be used either within or outside the vm crate.
            // Assume F is already generic of the field.
            let (execute_arms, get_opcode_name_arms): (Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                    let field_ty = &field.ty;
                    let execute_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::axvm_circuit::arch::InstructionExecutor<#first_ty_generic>>::execute(x, instruction, from_state)
                    };
                    let get_opcode_name_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::axvm_circuit::arch::InstructionExecutor<#first_ty_generic>>::get_opcode_name(x, opcode)
                    };

                    (execute_arm, get_opcode_name_arm)
                }));
            quote! {
                impl #impl_generics ::axvm_circuit::arch::InstructionExecutor<#first_ty_generic> for #name #ty_generics {
                    fn execute(
                        &mut self,
                        instruction: ::axvm_circuit::arch::instructions::instruction::Instruction<#first_ty_generic>,
                        from_state: ::axvm_circuit::arch::ExecutionState<u32>,
                    ) -> ::axvm_circuit::arch::Result<::axvm_circuit::arch::ExecutionState<u32>> {
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

/// Derives `AnyEnum` trait on an enum type.
/// By default an enum arm will just return `self` as `&dyn Any`.
///
/// Use the `#[any_enum]` field attribute to specify that the
/// arm itself implements `AnyEnum` and should call the inner `as_any_kind` method.
#[proc_macro_derive(AnyEnum, attributes(any_enum))]
pub fn any_enum_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let generics = &ast.generics;
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    match &ast.data {
        Data::Enum(e) => {
            let variants = e
                .variants
                .iter()
                .map(|variant| {
                    let variant_name = &variant.ident;

                    // Check if the variant has #[any_enum] attribute
                    let is_enum = variant
                        .attrs
                        .iter()
                        .any(|attr| attr.path().is_ident("any_enum"));
                    let mut fields = variant.fields.iter();
                    let field = fields.next().unwrap();
                    assert!(fields.next().is_none(), "Only one field is supported");
                    (variant_name, field, is_enum)
                })
                .collect::<Vec<_>>();
            let arms: Vec<_> =
                variants.iter().map(|(variant_name, field, is_enum)| {
                    let field_ty = &field.ty;

                    if *is_enum {
                        // Call the inner trait impl
                        quote! {
                            #name::#variant_name(x) => <#field_ty as ::axvm_circuit::arch::AnyEnum>::as_any_kind(x)
                        }
                    } else {
                        quote! {
                            #name::#variant_name(x) => x
                        }
                    }
                }).collect();
            quote! {
                impl #impl_generics ::axvm_circuit::arch::AnyEnum for #name #ty_generics {
                    fn as_any_kind(&self) -> &dyn Any {
                        match self {
                            #(#arms,)*
                        }
                    }
                }
            }
            .into()
        }
        _ => unimplemented!("Only enums are supported"),
    }
}
