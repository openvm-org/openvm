// AlignedBorrow is copied from valida-derive under MIT license
extern crate alloc;
extern crate proc_macro;

use itertools::multiunzip;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, GenericParam, LitStr, Meta};

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

    // Get impl generics (`<T, N: NumLimbs, const M: usize>`), type generics (`<T, N>`), where
    // clause (`where T: Clone`)
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

/// `S` is the type the derive macro is being called on
/// Implements Borrow<S> and BorrowMut<S> for [u8]
/// [u8] has to have (checked via `debug_assert!`s)
/// - at least size_of(S) length
/// - at least align_of(S) alignment
#[proc_macro_derive(AlignedBytesBorrow)]
pub fn aligned_bytes_borrow_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    // Get impl generics, type generics, where clause
    // Note, need to add the new type generic to the `impl_generics`
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();

    let methods = quote! {
        impl #impl_generics core::borrow::Borrow<#name #type_generics> for [u8]
        where
            #where_clause
        {
            fn borrow(&self) -> &#name #type_generics {
                use core::mem::{align_of, size_of_val};
                debug_assert!(size_of_val(self) >= core::mem::size_of::<#name #type_generics>());
                debug_assert_eq!(self.as_ptr() as usize % align_of::<#name #type_generics>(), 0);
                unsafe { &*(self.as_ptr() as *const #name #type_generics) }
            }
        }

        impl #impl_generics core::borrow::BorrowMut<#name #type_generics> for [u8]
        where
            #where_clause
        {
            fn borrow_mut(&mut self) -> &mut #name #type_generics {
                use core::mem::{align_of, size_of_val};
                debug_assert!(size_of_val(self) >= core::mem::size_of::<#name #type_generics>());
                debug_assert_eq!(self.as_ptr() as usize % align_of::<#name #type_generics>(), 0);
                unsafe { &mut *(self.as_mut_ptr() as *mut #name #type_generics) }
            }
        }
    };

    TokenStream::from(methods)
}

#[proc_macro_derive(Chip, attributes(chip))]
pub fn chip_derive(input: TokenStream) -> TokenStream {
    // Parse the attributes from the struct or enum
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let generics = &ast.generics;
    let (_impl_generics, ty_generics, _where_clause) = generics.split_for_impl();

    match &ast.data {
        Data::Struct(inner) => {
            let generics = &ast.generics;
            let mut new_generics = generics.clone();
            new_generics.params.push(syn::parse_quote! { R });
            new_generics
                .params
                .push(syn::parse_quote! { PB: openvm_stark_backend::prover::hal::ProverBackend });
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
                .push(syn::parse_quote! { #inner_ty: openvm_stark_backend::Chip<R, PB> });
            quote! {
                impl #impl_generics openvm_stark_backend::Chip<R, PB> for #name #ty_generics #where_clause {
                    fn generate_proving_ctx(&self, records: R) -> openvm_stark_backend::prover::types::AirProvingContext<PB> {
                        self.0.generate_proving_ctx(records)
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

            let (generate_proving_ctx_arms, where_predicates): (Vec<_>, Vec<_>) =
                variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                let generate_proving_ctx_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as openvm_stark_backend::Chip<R, PB>>::generate_proving_ctx(x, records)
                };
                let where_predicate =
                    syn::parse_quote! { #field_ty: openvm_stark_backend::Chip<R, PB> };
                (generate_proving_ctx_arm, where_predicate)
            }).collect();

            // Attach extra generics R and PB to the impl_generics
            let generics = &ast.generics;
            let mut new_generics = generics.clone();
            new_generics.params.push(syn::parse_quote! { R });
            new_generics
                .params
                .push(syn::parse_quote! { PB: openvm_stark_backend::prover::hal::ProverBackend });
            let (impl_generics, _, _) = new_generics.split_for_impl();

            // Implement Chip whenever the inner type implements Chip
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            for predicate in where_predicates {
                where_clause.predicates.push(predicate);
            }
            let attributes = ast.attrs.iter().find(|&attr| attr.path().is_ident("chip"));
            if let Some(attr) = attributes {
                let mut fail_flag = false;

                match &attr.meta {
                    Meta::List(meta_list) => {
                        meta_list
                            .parse_nested_meta(|meta| {
                                if meta.path.is_ident("where") {
                                    let value = meta.value()?; // this parses the `=`
                                    let s: LitStr = value.parse()?;
                                    let where_value = s.value();
                                    where_clause.predicates.push(syn::parse_str(&where_value)?);
                                } else {
                                    fail_flag = true;
                                }
                                Ok(())
                            })
                            .unwrap();
                    }
                    _ => fail_flag = true,
                }
                if fail_flag {
                    return syn::Error::new(
                        name.span(),
                        "Only `#[chip(where = ...)]` format is supported",
                    )
                    .to_compile_error()
                    .into();
                }
            }

            quote! {
                impl #impl_generics openvm_stark_backend::Chip<R, PB> for #name #ty_generics #where_clause {
                    fn generate_proving_ctx(&self, records: R) -> openvm_stark_backend::prover::types::AirProvingContext<PB> {
                        match self {
                            #(#generate_proving_ctx_arms,)*
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
                .push(syn::parse_quote! { #inner_ty: openvm_stark_backend::ChipUsageGetter });
            quote! {
                impl #impl_generics openvm_stark_backend::ChipUsageGetter for #name #ty_generics #where_clause {
                    fn air_name(&self) -> String {
                        self.0.air_name()
                    }
                    fn constant_trace_height(&self) -> Option<usize> {
                        self.0.constant_trace_height()
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
            let (air_name_arms, constant_trace_height_arms, current_trace_height_arms, trace_width_arms): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(e.variants.iter().map(|variant| {
                    let variant_name = &variant.ident;
                    let air_name_arm = quote! {
                    #name::#variant_name(x) => openvm_stark_backend::ChipUsageGetter::air_name(x)
                };
                    let constant_trace_height_arm = quote! {
                    #name::#variant_name(x) => openvm_stark_backend::ChipUsageGetter::constant_trace_height(x)
                };
                    let current_trace_height_arm = quote! {
                    #name::#variant_name(x) => openvm_stark_backend::ChipUsageGetter::current_trace_height(x)
                };
                    let trace_width_arm = quote! {
                    #name::#variant_name(x) => openvm_stark_backend::ChipUsageGetter::trace_width(x)
                };
                    (air_name_arm, constant_trace_height_arm, current_trace_height_arm, trace_width_arm)
                }));

            quote! {
                impl #impl_generics openvm_stark_backend::ChipUsageGetter for #name #ty_generics {
                    fn air_name(&self) -> String {
                        match self {
                            #(#air_name_arms,)*
                        }
                    }
                    fn constant_trace_height(&self) -> Option<usize> {
                        match self {
                            #(#constant_trace_height_arms,)*
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

#[proc_macro_derive(BytesStateful)]
pub fn bytes_stateful_derive(input: TokenStream) -> TokenStream {
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
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause
                .predicates
                .push(syn::parse_quote! { #inner_ty: ::openvm_stark_backend::Stateful<Vec<u8>> });

            quote! {
                impl #impl_generics ::openvm_stark_backend::Stateful<Vec<u8>> for #name #ty_generics #where_clause {
                    fn load_state(&mut self, state: Vec<u8>) {
                        self.0.load_state(state)
                    }

                    fn store_state(&self) -> Vec<u8> {
                        self.0.store_state()
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
            // Use full path ::openvm_stark_backend... so it can be used either within or outside
            // the vm crate.
            let (load_state_arms, store_state_arms): (Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                    let field_ty = &field.ty;
                    let load_state_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_stark_backend::Stateful<Vec<u8>>>::load_state(x, state)
                    };
                    let store_state_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_stark_backend::Stateful<Vec<u8>>>::store_state(x)
                    };

                    (load_state_arm, store_state_arm)
                }));
            quote! {
                impl #impl_generics ::openvm_stark_backend::Stateful<Vec<u8>> for #name #ty_generics {
                    fn load_state(&mut self, state: Vec<u8>) {
                        match self {
                            #(#load_state_arms,)*
                        }
                    }

                    fn store_state(&self) -> Vec<u8> {
                        match self {
                            #(#store_state_arms,)*
                        }
                    }
                }
            }
            .into()
        }
        _ => unimplemented!(),
    }
}
