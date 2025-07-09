extern crate alloc;
extern crate proc_macro;

use itertools::{multiunzip, Itertools};
use proc_macro::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    punctuated::Punctuated, spanned::Spanned, Data, DataStruct, Field, Fields, GenericParam, Ident,
    Meta, Token,
};

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
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause.predicates.push(
                syn::parse_quote! { #inner_ty: ::openvm_circuit::arch::InstructionExecutor<F> },
            );
            quote! {
                impl #impl_generics ::openvm_circuit::arch::InstructionExecutor<F> for #name #ty_generics #where_clause {
                    fn execute(
                        &mut self,
                        state: ::openvm_circuit::arch::VmStateMut<F, ::openvm_circuit::system::memory::online::TracingMemory,
                        ::openvm_circuit::arch::MatrixRecordArena<F>>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                    ) -> ::openvm_circuit::arch::Result<()> {
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
            let default_ty_generic = Ident::new("F", proc_macro2::Span::call_site());
            let mut new_generics = generics.clone();
            let field_ty_generic = ast
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
            let (execute_arms, get_opcode_name_arms, where_predicates): (Vec<_>, Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                    let field_ty = &field.ty;
                    let execute_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InstructionExecutor<#field_ty_generic>>::execute(x, state, instruction)
                    };
                    let get_opcode_name_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InstructionExecutor<#field_ty_generic>>::get_opcode_name(x, opcode)
                    };
                    let where_predicate = syn::parse_quote! {
                        #field_ty: ::openvm_circuit::arch::InstructionExecutor<#field_ty_generic>
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
                impl #impl_generics ::openvm_circuit::arch::InstructionExecutor<#field_ty_generic> for #name #ty_generics #where_clause {
                    fn execute(
                        &mut self,
                        state: ::openvm_circuit::arch::VmStateMut<#field_ty_generic, ::openvm_circuit::system::memory::online::TracingMemory, ::openvm_circuit::arch::MatrixRecordArena<#field_ty_generic>>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<#field_ty_generic>,
                    ) -> ::openvm_circuit::arch::Result<()> {
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

#[proc_macro_derive(InsExecutorE1)]
pub fn ins_executor_e1_executor_derive(input: TokenStream) -> TokenStream {
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
                .push(syn::parse_quote! { #inner_ty: ::openvm_circuit::arch::InsExecutorE1<F> });
            quote! {
                impl #impl_generics ::openvm_circuit::arch::InsExecutorE1<F> for #name #ty_generics #where_clause {
                    fn execute_e1<Ctx>(
                        &self,
                        state: &mut ::openvm_circuit::arch::VmStateMut<F, ::openvm_circuit::system::memory::online::GuestMemory, Ctx>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                    ) -> ::openvm_circuit::arch::Result<()>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::E1E2ExecutionCtx,
                    {
                        self.0.execute_e1(state, instruction)
                    }

                    fn execute_metered(
                        &self,
                        state: &mut ::openvm_circuit::arch::VmStateMut<F, ::openvm_circuit::system::memory::online::GuestMemory, ::openvm_circuit::arch::execution_mode::metered::MeteredCtx>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        chip_index: usize,
                    ) -> ::openvm_circuit::arch::Result<()> {
                        self.0.execute_metered(state, instruction, chip_index)
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
            let (execute_e1_arms, execute_metered_arms, where_predicates): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                let execute_e1_arm= quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic>>::execute_e1(x, state, instruction)
                };
                let execute_metered_arm =quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic>>::execute_metered(x, state, instruction, chip_index)
                };
                let where_predicate = syn::parse_quote! {
                    #field_ty: ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic>
                };
                (execute_e1_arm, execute_metered_arm, where_predicate)
            }));
            let where_clause = new_generics.make_where_clause();
            for predicate in where_predicates {
                where_clause.predicates.push(predicate);
            }
            // Don't use these ty_generics because it might have extra "F"
            let (impl_generics, _, where_clause) = new_generics.split_for_impl();

            quote! {
                impl #impl_generics ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic> for #name #ty_generics #where_clause {
                    fn execute_e1<Ctx>(
                        &self,
                        state: &mut ::openvm_circuit::arch::VmStateMut<F,::openvm_circuit::system::memory::online::GuestMemory, Ctx>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<#first_ty_generic>,
                    ) -> ::openvm_circuit::arch::Result<()>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::E1E2ExecutionCtx,
                    {
                        match self {
                            #(#execute_e1_arms,)*
                        }
                    }

                    fn execute_metered(
                        &self,
                        state: &mut ::openvm_circuit::arch::VmStateMut<F, ::openvm_circuit::system::memory::online::GuestMemory, ::openvm_circuit::arch::execution_mode::metered::MeteredCtx>,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<#first_ty_generic>,
                        chip_index: usize,
                    ) -> ::openvm_circuit::arch::Result<()> {
                        match self {
                            #(#execute_metered_arms,)*
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
        .clone()
        .expect("Exactly one field must have the #[config] attribute");
    let (source_name, source_name_upper) =
        gen_name_with_uppercase_idents(source_field.ident.as_ref().unwrap());

    let extensions = fields
        .iter()
        .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("extension")))
        .cloned()
        .collect::<Vec<_>>();

    let mut executor_enum_fields = Vec::new();
    let mut create_executors = Vec::new();
    let mut create_circuit = Vec::new();
    let mut create_chip_complex = Vec::new();
    for e in extensions.iter() {
        let (ext_field_name, ext_name_upper) =
            gen_name_with_uppercase_idents(e.ident.as_ref().unwrap());
        let (executor_name, needs_generics) = parse_executor_name(e)?;
        let executor_type = if needs_generics {
            quote! { #executor_name<F> }
        } else {
            quote! { #executor_name }
        };
        executor_enum_fields.push(quote! {
            #[any_enum]
            #ext_name_upper(#executor_type),
        });
        create_executors.push(quote! {
            let inventory: ::openvm_circuit::arch::ExecutorInventory<Self::Executor> = inventory.extend::<F, _, _>(&self.#ext_field_name)?;
        });
        create_circuit.push(quote! {
            inventory.start_new_extension();
            ::openvm_circuit::arch::VmCircuitExtension::extend_circuit(&self.#ext_field_name, &mut inventory)?;
        });
        create_chip_complex.push(quote! {
            inventory.start_new_extension();
            ::openvm_circuit::arch::VmProverExtension::extend_prover(&self.#ext_field_name, &mut inventory)?;
        })
    }

    let (source_executor_name, source_needs_generics) = parse_executor_name(source_field)?;
    let source_executor_type = if source_needs_generics {
        quote! { #source_executor_name<F> }
    } else {
        quote! { #source_executor_name }
    };
    let executor_type = Ident::new(&format!("{}Executor", name), name.span());

    let token_stream = TokenStream::from(quote! {
        #[derive(
            Clone,
            ::openvm_circuit::derive::InstructionExecutor,
            ::openvm_circuit::derive::InsExecutorE1,
            ::derive_more::derive::From,
            ::openvm_circuit::derive::AnyEnum
        )]
        pub enum #executor_type<F: Field> {
            #[any_enum]
            #source_name_upper(#source_executor_type),
            #(#executor_enum_fields)*
        }

        impl<F: PrimeField32> ::openvm_circuit::arch::VmExecutionConfig<F> for #name {
            type Executor = #executor_type<F>;

            fn create_executors(
                &self,
            ) -> Result<::openvm_circuit::arch::ExecutorInventory<Self::Executor>, ::openvm_circuit::arch::ExecutorInventoryError> {
                let inventory = self.#source_name.create_executors()?;
                #(#create_executors)*
                Ok(inventory)
            }
        }
    });
    Ok(token_stream)
}

// Parse the executor name as either
// `{type_name}Executor` or whatever the attribute `executor = ` specifies
// Also determines whether the executor type needs generic parameters
fn parse_executor_name(f: &Field) -> syn::Result<(Ident, bool)> {
    // TRACKING ISSUE:
    // We cannot just use <e.ty.to_token_stream() as VmExecutionExtension<F>>::Executor because of this: <https://github.com/rust-lang/rust/issues/85576>
    let mut executor_name = Ident::new(
        &format!("{}Executor", f.ty.to_token_stream()),
        Span::call_site().into(),
    );
    let mut needs_generics = true; // Default to true for backward compatibility

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
                    .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                    .unwrap();
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) => {
                            if nv.path.is_ident("executor") {
                                executor_name = Ident::new(
                                    &nv.value.to_token_stream().to_string(),
                                    Span::call_site().into(),
                                );
                                Ok(())
                            } else if nv.path.is_ident("generics") {
                                // Parse boolean value for generics
                                let value_str = nv.value.to_token_stream().to_string();
                                needs_generics = match value_str.as_str() {
                                    "true" => true,
                                    "false" => false,
                                    _ => return Err(syn::Error::new(
                                        nv.value.span(),
                                        "generics attribute must be either true or false"
                                    ))
                                };
                                Ok(())
                            } else {
                                Err("only executor and generics keys are supported")
                            }
                        }
                        _ => Err("only name = value format is supported"),
                    }
                    .expect("wrong attributes format");
                }
            }
        }
    };
    Ok((executor_name, needs_generics))
}
