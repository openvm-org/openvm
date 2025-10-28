extern crate alloc;
extern crate proc_macro;

use itertools::{multiunzip, Itertools};
use proc_macro::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse_quote, punctuated::Punctuated, spanned::Spanned, Data, DataStruct, Field, Fields,
    GenericParam, Ident, Meta, Token,
};

mod common;
#[cfg(not(feature = "tco"))]
mod nontco;
#[cfg(feature = "tco")]
mod tco;

#[proc_macro_derive(PreflightExecutor)]
pub fn preflight_executor_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let generics = &ast.generics;
    let (_, ty_generics, _) = generics.split_for_impl();

    let default_ty_generic = Ident::new("F", proc_macro2::Span::call_site());
    let mut new_generics = generics.clone();
    new_generics.params.push(syn::parse_quote! { RA });
    let field_ty_generic = generics
        .params
        .first()
        .and_then(|param| match param {
            GenericParam::Type(type_param) => Some(&type_param.ident),
            _ => None,
        })
        .unwrap_or_else(|| {
            new_generics.params.push(syn::parse_quote! { F });
            &default_ty_generic
        });

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
            let where_clause = new_generics.make_where_clause();
            where_clause.predicates.push(
                syn::parse_quote! { #inner_ty: ::openvm_circuit::arch::PreflightExecutor<#field_ty_generic, RA> },
            );
            let (impl_generics, _, where_clause) = new_generics.split_for_impl();
            quote! {
                impl #impl_generics ::openvm_circuit::arch::PreflightExecutor<#field_ty_generic, RA> for #name #ty_generics #where_clause {
                    fn execute(
                        &self,
                        state: ::openvm_circuit::arch::VmStateMut<#field_ty_generic, ::openvm_circuit::system::memory::online::TracingMemory, RA>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<#field_ty_generic>,
                    ) -> Result<(), ::openvm_circuit::arch::ExecutionError> {
                        self.0.execute(state, instruction)
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
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate.
            let (execute_arms, get_opcode_name_arms, where_predicates): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                    let field_ty = &field.ty;
                    let execute_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::PreflightExecutor<#field_ty_generic, RA>>::execute(x, state, instruction)
                    };
                    let get_opcode_name_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::PreflightExecutor<#field_ty_generic, RA>>::get_opcode_name(x, opcode)
                    };
                    let where_predicate = syn::parse_quote! {
                        #field_ty: ::openvm_circuit::arch::PreflightExecutor<#field_ty_generic, RA>
                    };
                    (execute_arm, get_opcode_name_arm, where_predicate)
                }));
            let where_clause = new_generics.make_where_clause();
            for predicate in where_predicates {
                where_clause.predicates.push(predicate);
            }
            // Don't use these ty_generics because it might have extra "F"
            let (impl_generics, _, where_clause) = new_generics.split_for_impl();
            quote! {
                impl #impl_generics ::openvm_circuit::arch::PreflightExecutor<#field_ty_generic, RA> for #name #ty_generics #where_clause {
                    fn execute(
                        &self,
                        state: ::openvm_circuit::arch::VmStateMut<#field_ty_generic, ::openvm_circuit::system::memory::online::TracingMemory, RA>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<#field_ty_generic>,
                    ) -> Result<(), ::openvm_circuit::arch::ExecutionError> {
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

#[proc_macro_derive(Executor)]
pub fn executor_derive(input: TokenStream) -> TokenStream {
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
                .push(syn::parse_quote! { #inner_ty: ::openvm_circuit::arch::Executor<F> });

            // We use the macro's feature to decide whether to generate the impl or not. This avoids
            // the target crate needing the "tco" feature defined.
            #[cfg(feature = "tco")]
            let handler = quote! {
                fn handler<Ctx>(
                    &self,
                    pc: u32,
                    inst: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                    data: &mut [u8],
                ) -> Result<::openvm_circuit::arch::Handler<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                where
                    Ctx: ::openvm_circuit::arch::execution_mode::ExecutionCtxTrait, {
                    self.0.handler(pc, inst, data)
                }
            };
            #[cfg(not(feature = "tco"))]
            let handler = quote! {};

            quote! {
                impl #impl_generics ::openvm_circuit::arch::Executor<F> for #name #ty_generics #where_clause {
                    #[inline(always)]
                    fn pre_compute_size(&self) -> usize {
                        self.0.pre_compute_size()
                    }
                    #[cfg(not(feature = "tco"))]
                    #[inline(always)]
                    fn pre_compute<Ctx>(
                        &self,
                        pc: u32,
                        inst: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::ExecutionCtxTrait, {
                        self.0.pre_compute(pc, inst, data)
                    }

                    #handler
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
            let default_ty_generic = Ident::new("F", proc_macro2::Span::call_site());
            let mut new_generics = generics.clone();
            let first_ty_generic = ast
                .generics
                .params
                .first()
                .and_then(|param| match param {
                    GenericParam::Type(type_param) => Some(&type_param.ident),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    new_generics.params.push(syn::parse_quote! { F });
                    &default_ty_generic
                });
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let (pre_compute_size_arms, pre_compute_arms, _handler_arms, where_predicates): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multiunzip(variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                let pre_compute_size_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::Executor<#first_ty_generic>>::pre_compute_size(x)
                };
                let pre_compute_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::Executor<#first_ty_generic>>::pre_compute(x, pc, instruction, data)
                };
                let handler_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::Executor<#first_ty_generic>>::handler(x, pc, instruction, data)
                };
                let where_predicate = syn::parse_quote! {
                    #field_ty: ::openvm_circuit::arch::Executor<#first_ty_generic>
                };
                (pre_compute_size_arm, pre_compute_arm, handler_arm, where_predicate)
            }));
            let where_clause = new_generics.make_where_clause();
            for predicate in where_predicates {
                where_clause.predicates.push(predicate);
            }
            // We use the macro's feature to decide whether to generate the impl or not. This avoids
            // the target crate needing the "tco" feature defined.
            #[cfg(feature = "tco")]
            let handler = quote! {
                fn handler<Ctx>(
                    &self,
                    pc: u32,
                    instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                    data: &mut [u8],
                ) -> Result<::openvm_circuit::arch::Handler<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                where
                    Ctx: ::openvm_circuit::arch::execution_mode::ExecutionCtxTrait, {
                    match self {
                        #(#_handler_arms,)*
                    }
                }
            };
            #[cfg(not(feature = "tco"))]
            let handler = quote! {};

            // Don't use these ty_generics because it might have extra "F"
            let (impl_generics, _, where_clause) = new_generics.split_for_impl();

            quote! {
                impl #impl_generics ::openvm_circuit::arch::Executor<#first_ty_generic> for #name #ty_generics #where_clause {
                    #[inline(always)]
                    fn pre_compute_size(&self) -> usize {
                        match self {
                            #(#pre_compute_size_arms,)*
                        }
                    }

                    #[cfg(not(feature = "tco"))]
                    #[inline(always)]
                    fn pre_compute<Ctx>(
                        &self,
                        pc: u32,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::ExecutionCtxTrait, {
                        match self {
                            #(#pre_compute_arms,)*
                        }
                    }

                    #handler
                }
            }
            .into()
        }
        Data::Union(_) => unimplemented!("Unions are not supported"),
    }
}

#[proc_macro_derive(MeteredExecutor)]
pub fn metered_executor_derive(input: TokenStream) -> TokenStream {
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
            // crate.
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause
                .predicates
                .push(syn::parse_quote! { #inner_ty: ::openvm_circuit::arch::MeteredExecutor<F> });

            // We use the macro's feature to decide whether to generate the impl or not. This avoids
            // the target crate needing the "tco" feature defined.
            #[cfg(feature = "tco")]
            let metered_handler = quote! {
                fn metered_handler<Ctx>(
                    &self,
                    chip_idx: usize,
                    pc: u32,
                    inst: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                    data: &mut [u8],
                ) -> Result<::openvm_circuit::arch::Handler<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                where
                    Ctx: ::openvm_circuit::arch::execution_mode::MeteredExecutionCtxTrait, {
                    self.0.metered_handler(chip_idx, pc, inst, data)
                }
            };
            #[cfg(not(feature = "tco"))]
            let metered_handler = quote! {};

            quote! {
                impl #impl_generics ::openvm_circuit::arch::MeteredExecutor<F> for #name #ty_generics #where_clause {
                    #[inline(always)]
                    fn metered_pre_compute_size(&self) -> usize {
                        self.0.metered_pre_compute_size()
                    }
                    #[cfg(not(feature = "tco"))]
                    #[inline(always)]
                    fn metered_pre_compute<Ctx>(
                        &self,
                        chip_idx: usize,
                        pc: u32,
                        inst: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::MeteredExecutionCtxTrait, {
                        self.0.metered_pre_compute(chip_idx, pc, inst, data)
                    }
                    #metered_handler
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
            let default_ty_generic = Ident::new("F", proc_macro2::Span::call_site());
            let mut new_generics = generics.clone();
            let first_ty_generic = ast
                .generics
                .params
                .first()
                .and_then(|param| match param {
                    GenericParam::Type(type_param) => Some(&type_param.ident),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    new_generics.params.push(syn::parse_quote! { F });
                    &default_ty_generic
                });
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let (pre_compute_size_arms, metered_pre_compute_arms, _metered_handler_arms, where_predicates): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = multiunzip(variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                let pre_compute_size_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::MeteredExecutor<#first_ty_generic>>::metered_pre_compute_size(x)
                };
                let metered_pre_compute_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::MeteredExecutor<#first_ty_generic>>::metered_pre_compute(x, chip_idx, pc, instruction, data)
                };
                let metered_handler_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::MeteredExecutor<#first_ty_generic>>::metered_handler(x, chip_idx, pc, instruction, data)
                };
                let where_predicate = syn::parse_quote! {
                    #field_ty: ::openvm_circuit::arch::MeteredExecutor<#first_ty_generic>
                };
                (pre_compute_size_arm, metered_pre_compute_arm, metered_handler_arm, where_predicate)
            }));
            let where_clause = new_generics.make_where_clause();
            for predicate in where_predicates {
                where_clause.predicates.push(predicate);
            }
            // Don't use these ty_generics because it might have extra "F"
            let (impl_generics, _, where_clause) = new_generics.split_for_impl();

            // We use the macro's feature to decide whether to generate the impl or not. This avoids
            // the target crate needing the "tco" feature defined.
            #[cfg(feature = "tco")]
            let metered_handler = quote! {
                fn metered_handler<Ctx>(
                    &self,
                    chip_idx: usize,
                    pc: u32,
                    instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                    data: &mut [u8],
                ) -> Result<::openvm_circuit::arch::Handler<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                where
                    Ctx: ::openvm_circuit::arch::execution_mode::MeteredExecutionCtxTrait,
                {
                    match self {
                        #(#_metered_handler_arms,)*
                    }
                }
            };
            #[cfg(not(feature = "tco"))]
            let metered_handler = quote! {};

            quote! {
                impl #impl_generics ::openvm_circuit::arch::MeteredExecutor<#first_ty_generic> for #name #ty_generics #where_clause {
                    #[inline(always)]
                    fn metered_pre_compute_size(&self) -> usize {
                        match self {
                            #(#pre_compute_size_arms,)*
                        }
                    }

                    #[cfg(not(feature = "tco"))]
                    #[inline(always)]
                    fn metered_pre_compute<Ctx>(
                        &self,
                        chip_idx: usize,
                        pc: u32,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>, ::openvm_circuit::arch::StaticProgramError>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::MeteredExecutionCtxTrait, {
                        match self {
                            #(#metered_pre_compute_arms,)*
                        }
                    }

                    #metered_handler
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
            let (arms, arms_mut): (Vec<_>, Vec<_>) =
                variants.iter().map(|(variant_name, field, is_enum)| {
                    let field_ty = &field.ty;

                    if *is_enum {
                        // Call the inner trait impl
                        (quote! {
                            #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::AnyEnum>::as_any_kind(x)
                        },
                        quote! {
                            #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::AnyEnum>::as_any_kind_mut(x)
                        })
                    } else {
                        (quote! {
                            #name::#variant_name(x) => x
                        },
                        quote! {
                            #name::#variant_name(x) => x
                        })
                    }
                }).unzip();
            quote! {
                impl #impl_generics ::openvm_circuit::arch::AnyEnum for #name #ty_generics {
                    fn as_any_kind(&self) -> &dyn std::any::Any {
                        match self {
                            #(#arms,)*
                        }
                    }

                    fn as_any_kind_mut(&mut self) -> &mut dyn std::any::Any {
                        match self {
                            #(#arms_mut,)*
                        }
                    }
                }
            }
            .into()
        }
        _ => syn::Error::new(name.span(), "Only enums are supported")
            .to_compile_error()
            .into(),
    }
}

#[proc_macro_derive(VmConfig, attributes(config, extension))]
pub fn vm_generic_config_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = &ast.ident;

    match &ast.data {
        syn::Data::Struct(inner) => match generate_config_traits_impl(name, inner) {
            Ok(tokens) => tokens,
            Err(err) => err.to_compile_error().into(),
        },
        _ => syn::Error::new(name.span(), "Only structs are supported")
            .to_compile_error()
            .into(),
    }
}

fn generate_config_traits_impl(name: &Ident, inner: &DataStruct) -> syn::Result<TokenStream> {
    let gen_name_with_uppercase_idents = |ident: &Ident| {
        let mut name = ident.to_string().chars().collect::<Vec<_>>();
        assert!(name[0].is_lowercase(), "Field name must not be capitalized");
        let res_lower = Ident::new(&name.iter().collect::<String>(), Span::call_site().into());
        name[0] = name[0].to_ascii_uppercase();
        let res_upper = Ident::new(&name.iter().collect::<String>(), Span::call_site().into());
        (res_lower, res_upper)
    };

    let fields = match &inner.fields {
        Fields::Named(named) => named.named.iter().collect(),
        Fields::Unnamed(_) => {
            return Err(syn::Error::new(
                name.span(),
                "Only named fields are supported",
            ))
        }
        Fields::Unit => vec![],
    };

    let source_field = fields
        .iter()
        .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("config")))
        .exactly_one()
        .map_err(|_| {
            syn::Error::new(
                name.span(),
                "Exactly one field must have the #[config] attribute",
            )
        })?;
    let (source_name, source_name_upper) =
        gen_name_with_uppercase_idents(source_field.ident.as_ref().unwrap());

    let extensions = fields
        .iter()
        .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("extension")))
        .cloned()
        .collect::<Vec<_>>();

    let mut executor_enum_fields = Vec::new();
    let mut create_executors = Vec::new();
    let mut create_airs = Vec::new();
    let mut execution_where_predicates: Vec<syn::WherePredicate> = Vec::new();
    let mut circuit_where_predicates: Vec<syn::WherePredicate> = Vec::new();

    let source_field_ty = source_field.ty.clone();

    for e in extensions.iter() {
        let (ext_field_name, ext_name_upper) =
            gen_name_with_uppercase_idents(e.ident.as_ref().expect("field must be named"));
        let executor_type = parse_executor_type(e, false)?;
        executor_enum_fields.push(quote! {
            #[any_enum]
            #ext_name_upper(#executor_type),
        });
        create_executors.push(quote! {
            let inventory: ::openvm_circuit::arch::ExecutorInventory<Self::Executor> = inventory.extend::<F, _, _>(&self.#ext_field_name)?;
        });
        let extension_ty = e.ty.clone();
        execution_where_predicates.push(parse_quote! {
            #extension_ty: ::openvm_circuit::arch::VmExecutionExtension<F, Executor = #executor_type>
        });
        create_airs.push(quote! {
            inventory.start_new_extension();
            ::openvm_circuit::arch::VmCircuitExtension::extend_circuit(&self.#ext_field_name, &mut inventory)?;
        });
        circuit_where_predicates.push(parse_quote! {
            #extension_ty: ::openvm_circuit::arch::VmCircuitExtension<SC>
        });
    }

    // The config type always needs <F> due to SystemExecutor
    let source_executor_type = parse_executor_type(source_field, true)?;
    execution_where_predicates.push(parse_quote! {
        #source_field_ty: ::openvm_circuit::arch::VmExecutionConfig<F, Executor = #source_executor_type>
    });
    circuit_where_predicates.push(parse_quote! {
        #source_field_ty: ::openvm_circuit::arch::VmCircuitConfig<SC>
    });
    let execution_where_clause = quote! { where #(#execution_where_predicates),* };
    let circuit_where_clause = quote! { where #(#circuit_where_predicates),* };

    let executor_type = Ident::new(&format!("{}Executor", name), name.span());

    let token_stream = TokenStream::from(quote! {
        #[derive(
            Clone,
            ::derive_more::derive::From,
            ::openvm_circuit::derive::AnyEnum,
            ::openvm_circuit::derive::Executor,
            ::openvm_circuit::derive::MeteredExecutor,
            ::openvm_circuit::derive::PreflightExecutor,
        )]
        pub enum #executor_type<F: openvm_stark_backend::p3_field::Field> {
            #[any_enum]
            #source_name_upper(#source_executor_type),
            #(#executor_enum_fields)*
        }

        impl<F: openvm_stark_backend::p3_field::Field> ::openvm_circuit::arch::VmExecutionConfig<F> for #name #execution_where_clause {
            type Executor = #executor_type<F>;

            fn create_executors(
                &self,
            ) -> Result<::openvm_circuit::arch::ExecutorInventory<Self::Executor>, ::openvm_circuit::arch::ExecutorInventoryError> {
                let inventory = self.#source_name.create_executors()?.transmute::<Self::Executor>();
                #(#create_executors)*
                Ok(inventory)
            }
        }

        impl<SC: openvm_stark_backend::config::StarkGenericConfig> ::openvm_circuit::arch::VmCircuitConfig<SC> for #name #circuit_where_clause {
            fn create_airs(
                &self,
            ) -> Result<::openvm_circuit::arch::AirInventory<SC>, ::openvm_circuit::arch::AirInventoryError> {
                let mut inventory = self.#source_name.create_airs()?;
                #(#create_airs)*
                Ok(inventory)
            }
        }

        impl AsRef<SystemConfig> for #name {
            fn as_ref(&self) -> &SystemConfig {
                self.#source_name.as_ref()
            }
        }

        impl AsMut<SystemConfig> for #name {
            fn as_mut(&mut self) -> &mut SystemConfig {
                self.#source_name.as_mut()
            }
        }
    });
    Ok(token_stream)
}

// Parse the executor name as either
// `{type_name}Executor` or whatever the attribute `executor = ` specifies
// Also determines whether the executor type needs generic parameters
fn parse_executor_type(
    f: &Field,
    default_needs_generics: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    // TRACKING ISSUE:
    // We cannot just use <e.ty.to_token_stream() as VmExecutionExtension<F>>::Executor because of this: <https://github.com/rust-lang/rust/issues/85576>
    let mut executor_type = None;
    // Do not unwrap the Result until needed
    let executor_name = syn::parse_str::<Ident>(&format!("{}Executor", f.ty.to_token_stream()));

    if let Some(attr) = f
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("extension") || attr.path().is_ident("config"))
    {
        match attr.meta {
            Meta::Path(_) => {}
            Meta::NameValue(_) => {
                return Err(syn::Error::new(
                    f.ty.span(),
                    "Only `#[config]`, `#[extension]`, `#[config(...)]` or `#[extension(...)]` formats are supported",
                ))
            }
            _ => {
                let nested = attr
                    .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) => {
                            if nv.path.is_ident("executor") {
                                executor_type = match nv.value {
                                    syn::Expr::Lit(syn::ExprLit {
                                        lit: syn::Lit::Str(lit_str), ..
                                    }) => {
                                        let executor_type: syn::Type = syn::parse_str(&lit_str.value())?;
                                        Some(quote! { #executor_type })
                                    },
                                    syn::Expr::Path(path) => {
                                        // Handle identifier paths like `executor = MyExecutor`
                                        Some(path.to_token_stream())
                                    },
                                    _ => {
                                        return Err(syn::Error::new(
                                            nv.value.span(),
                                            "executor value must be a string literal or identifier"
                                        ));
                                    }
                                };
                            } else if nv.path.is_ident("generics") {
                                // Parse boolean value for generics
                                let value_str = nv.value.to_token_stream().to_string();
                                let needs_generics = match value_str.as_str() {
                                    "true" => true,
                                    "false" => false,
                                    _ => return Err(syn::Error::new(
                                        nv.value.span(),
                                        "generics attribute must be either true or false"
                                    ))
                                };
                                let executor_name = executor_name.clone()?;
                                executor_type = Some(if needs_generics {
                                    quote! { #executor_name<F> }
                                } else {
                                    quote! { #executor_name }
                                });
                            } else {
                                return Err(syn::Error::new(nv.span(), "only executor and generics keys are supported"));
                            }
                        }
                        _ => {
                            return Err(syn::Error::new(meta.span(), "only name = value format is supported"));
                        }
                    }
                }
            }
        }
    }
    if let Some(executor_type) = executor_type {
        Ok(executor_type)
    } else {
        let executor_name = executor_name?;
        Ok(if default_needs_generics {
            quote! { #executor_name<F> }
        } else {
            quote! { #executor_name }
        })
    }
}

/// An attribute procedural macro for creating TCO (Tail Call Optimization) handlers.
///
/// This macro generates a handler function that wraps an execute implementation
/// with tail call optimization using the `become` keyword. It extracts the generics
/// and where clauses from the original function.
///
/// # Usage
///
/// Place this attribute above a function definition:
/// ```
/// #[create_tco_handler]
/// unsafe fn execute_e1_impl<F: PrimeField32, CTX, const B_IS_IMM: bool>(
///     pre_compute: &[u8],
///     state: &mut VmExecState<F, GuestMemory, CTX>,
/// ) where
///     CTX: ExecutionCtxTrait,
/// {
///     // function body
/// }
/// ```
///
/// This will generate a TCO handler function with the same generics and where clauses.
///
/// # Safety
///
/// Do not use this macro if your function wants to terminate execution without error with a
/// specific error code. The handler generated by this macro assumes that execution should continue
/// unless the execute_impl returns an error. This is done for performance to skip an exit code
/// check.
#[proc_macro_attribute]
pub fn create_handler(_attr: TokenStream, item: TokenStream) -> TokenStream {
    #[cfg(feature = "tco")]
    {
        tco::tco_impl(item)
    }
    #[cfg(not(feature = "tco"))]
    {
        nontco::nontco_impl(item)
    }
}
