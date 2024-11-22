// AlignedBorrow is copied from valida-derive under MIT license
extern crate alloc;
extern crate proc_macro;

use itertools::multiunzip;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, GenericParam};

#[proc_macro_derive(AlignedBorrow)]
pub fn aligned_borrow_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    // Get first generic which must be type (ex. `T`) for input <T, N: NumLimbs, const M: usize>
    let type_generic = ast
        .generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(type_param) => &type_param.ident,
            _ => panic!("Expected first generic to be a type"),
        })
        .next()
        .expect("Expected at least one generic");

    // Get generics after the first (ex. `N: NumLimbs, const M: usize`)
    // We need this because when we assert the size, we want to substitute u8 for T.
    let non_first_generics = ast
        .generics
        .params
        .iter()
        .skip(1)
        .filter_map(|param| match param {
            GenericParam::Type(type_param) => Some(&type_param.ident),
            GenericParam::Const(const_param) => Some(&const_param.ident),
            _ => None,
        })
        .collect::<Vec<_>>();

    // Get impl generics (`<T, N: NumLimbs, const M: usize>`), type generics (`<T, N>`), where clause (`where T: Clone`)
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();

    let methods = quote! {
        impl #impl_generics core::borrow::Borrow<#name #type_generics> for [#type_generic] #where_clause {
            fn borrow(&self) -> &#name #type_generics {
                debug_assert_eq!(self.len(), #name::#type_generics::width());
                let (prefix, shorts, _suffix) = unsafe { self.align_to::<#name #type_generics>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert_eq!(shorts.len(), 1);
                &shorts[0]
            }
        }

        impl #impl_generics core::borrow::BorrowMut<#name #type_generics> for [#type_generic] #where_clause {
            fn borrow_mut(&mut self) -> &mut #name #type_generics {
                debug_assert_eq!(self.len(), #name::#type_generics::width());
                let (prefix, shorts, _suffix) = unsafe { self.align_to_mut::<#name #type_generics>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert_eq!(shorts.len(), 1);
                &mut shorts[0]
            }
        }

        impl #impl_generics #name #type_generics {
            pub const fn width() -> usize {
                std::mem::size_of::<#name<u8 #(, #non_first_generics)*>>()
            }
        }
    };

    TokenStream::from(methods)
}

#[proc_macro_derive(Chip)]
pub fn chip_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let generics = &ast.generics;
    let (_impl_generics, ty_generics, _where_clause) = generics.split_for_impl();

    match &ast.data {
        Data::Struct(inner) => {
            let generics = &ast.generics;
            let mut new_generics = generics.clone();
            new_generics
                .params
                .push(syn::parse_quote! { SC: ax_stark_backend::config::StarkGenericConfig });
            let (impl_generics, _, _) = new_generics.split_for_impl();

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
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause
                .predicates
                .push(syn::parse_quote! { #inner_ty: ax_stark_backend::Chip<SC> });
            quote! {
                impl #impl_generics ax_stark_backend::Chip<SC> for #name #ty_generics #where_clause {
                    fn air(&self) -> std::sync::Arc<dyn ax_stark_backend::rap::AnyRap<SC>> {
                        self.0.air()
                    }
                    fn generate_air_proof_input(self) -> ax_stark_backend::prover::types::AirProofInput<SC> {
                        self.0.generate_air_proof_input()
                    }
                    fn generate_air_proof_input_with_id(self, air_id: usize) -> (usize, ax_stark_backend::prover::types::AirProofInput<SC>) {
                        self.0.generate_air_proof_input_with_id(air_id)
                    }
                }
            }.into()
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

            let (air_arms, generate_air_proof_input_arms, generate_air_proof_input_with_id_arms): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                let air_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ax_stark_backend::Chip<SC>>::air(x)
                };
                let generate_air_proof_input_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ax_stark_backend::Chip<SC>>::generate_air_proof_input(x)
                };
                let generate_air_proof_input_with_id_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ax_stark_backend::Chip<SC>>::generate_air_proof_input_with_id(x, air_id)
                };
                (air_arm, generate_air_proof_input_arm, generate_air_proof_input_with_id_arm)
            }));

            // Attach an extra generic SC: StarkGenericConfig to the impl_generics
            let generics = &ast.generics;
            let mut new_generics = generics.clone();
            new_generics
                .params
                .push(syn::parse_quote! { SC: ax_stark_backend::config::StarkGenericConfig });
            let (impl_generics, _, _) = new_generics.split_for_impl();

            // Implement Chip whenever the inner type implements Chip
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause.predicates.push(syn::parse_quote! { ax_stark_backend::config::Domain<SC>: ax_stark_backend::p3_commit::PolynomialSpace<Val = F>
            });

            quote! {
                impl #impl_generics ax_stark_backend::Chip<SC> for #name #ty_generics #where_clause {
                    fn air(&self) -> std::sync::Arc<dyn ax_stark_backend::rap::AnyRap<SC>> {
                        match self {
                            #(#air_arms,)*
                        }
                    }
                    fn generate_air_proof_input(self) -> ax_stark_backend::prover::types::AirProofInput<SC> {
                        match self {
                            #(#generate_air_proof_input_arms,)*
                        }
                    }
                    fn generate_air_proof_input_with_id(self, air_id: usize) -> (usize, ax_stark_backend::prover::types::AirProofInput<SC>) {
                        match self {
                            #(#generate_air_proof_input_with_id_arms,)*
                        }
                    }
                }
            }.into()
        }
        Data::Union(_) => unimplemented!("Unions are not supported"),
    }
}

#[proc_macro_derive(ChipUsageGetter)]
pub fn chip_usage_getter_derive(input: TokenStream) -> TokenStream {
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
            // Implement ChipUsageGetter whenever the inner type implements ChipUsageGetter
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause
                .predicates
                .push(syn::parse_quote! { #inner_ty: ax_stark_backend::ChipUsageGetter });
            quote! {
                impl #impl_generics ax_stark_backend::ChipUsageGetter for #name #ty_generics #where_clause {
                    fn air_name(&self) -> String {
                        self.0.air_name()
                    }
                    fn current_trace_height(&self) -> usize {
                        self.0.current_trace_height()
                    }
                    fn trace_width(&self) -> usize {
                        self.0.trace_width()
                    }
                }
            }
            .into()
        }
        Data::Enum(e) => {
            let (air_name_arms, current_trace_height_arms, trace_width_arms): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(e.variants.iter().map(|variant| {
                    let variant_name = &variant.ident;
                    let air_name_arm = quote! {
                    #name::#variant_name(x) => ax_stark_backend::ChipUsageGetter::air_name(x)
                };
                    let current_trace_height_arm = quote! {
                    #name::#variant_name(x) => ax_stark_backend::ChipUsageGetter::current_trace_height(x)
                };
                    let trace_width_arm = quote! {
                    #name::#variant_name(x) => ax_stark_backend::ChipUsageGetter::trace_width(x)
                };
                    (air_name_arm, current_trace_height_arm, trace_width_arm)
                }));

            quote! {
                impl #impl_generics ax_stark_backend::ChipUsageGetter for #name #ty_generics {
                    fn air_name(&self) -> String {
                        match self {
                            #(#air_name_arms,)*
                        }
                    }
                    fn current_trace_height(&self) -> usize {
                        match self {
                            #(#current_trace_height_arms,)*
                        }
                    }
                    fn trace_width(&self) -> usize {
                        match self {
                            #(#trace_width_arms,)*
                        }
                    }

                }
            }
            .into()
        }
        Data::Union(_) => unimplemented!("Unions are not supported"),
    }
}
