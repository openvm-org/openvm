extern crate alloc;
extern crate proc_macro;

use itertools::{multiunzip, Itertools};
use proc_macro::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{Data, Fields, GenericParam, Ident};

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
            let (arms, arms_mut): (Vec<_>, Vec<_>) =
                variants.iter().map(|(variant_name, field, is_enum)| {
                    let field_ty = &field.ty;

                    if *is_enum {
                        // Call the inner trait impl
                        (quote! {
                            #name::#variant_name(x) => <#field_ty as ::axvm_circuit::arch::AnyEnum>::as_any_kind(x)
                        },
                        quote! {
                            #name::#variant_name(x) => <#field_ty as ::axvm_circuit::arch::AnyEnum>::as_any_kind_mut(x)
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
                impl #impl_generics ::axvm_circuit::arch::AnyEnum for #name #ty_generics {
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

#[proc_macro_derive(VmGenericConfig, attributes(system, extension))]
pub fn vm_generic_config_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = &ast.ident;

    match &ast.data {
        syn::Data::Struct(inner) => {
            let fields = match &inner.fields {
                Fields::Named(named) => named.named.iter().collect(),
                Fields::Unnamed(_) => {
                    return syn::Error::new(name.span(), "Only named fields are supported")
                        .to_compile_error()
                        .into();
                }
                Fields::Unit => vec![],
            };

            let system = fields
                .iter()
                .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("system")))
                .exactly_one()
                .expect("Exactly one field must have #[system] attribute");
            let mut system_name = system
                .ident
                .clone()
                .unwrap()
                .to_string()
                .chars()
                .collect::<Vec<_>>();
            system_name[0] = system_name[0].to_ascii_uppercase();
            let system_name = system_name.iter().collect::<String>();
            let system_name = Ident::new(&system_name, Span::call_site().into());
            let system_executor = Ident::new(
                &format!(
                    "{}Executor",
                    system
                        .ty
                        .to_token_stream()
                        .to_string()
                        .strip_suffix("Config")
                        .expect("Struct name must end with Config")
                ),
                Span::call_site().into(),
            );

            let extensions = fields
                .iter()
                .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("extension")))
                .cloned()
                .collect::<Vec<_>>();

            let extension_enums = extensions
                .into_iter()
                .map(|e| {
                    let mut field_name = e
                        .ident
                        .clone()
                        .unwrap()
                        .to_string()
                        .chars()
                        .collect::<Vec<_>>();
                    field_name[0] = field_name[0].to_ascii_uppercase();
                    let field_name = field_name.iter().collect::<String>();
                    let field_name = Ident::new(&field_name, Span::call_site().into());
                    let type_name = e.ty.to_token_stream().to_string();
                    let type_name =
                        Ident::new(&format!("{}Executor", type_name), Span::call_site().into());
                    quote! {
                        #[any_enum]
                        #field_name(#type_name<F>),
                    }
                })
                .collect::<Vec<_>>();

            let mut enum_name = String::from(
                name.to_string()
                    .strip_suffix("Config")
                    .expect("Struct name must end with Config"),
            );
            enum_name.push_str("Executor");
            let enum_name = Ident::new(&enum_name, name.span());
            TokenStream::from(quote! {
                #[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
                pub enum #enum_name<F: PrimeField32> {
                    #[any_enum]
                    #system_name(#system_executor<F>),
                    #(#extension_enums)*
                }
            })
        }
        _ => syn::Error::new(name.span(), "Only structs are supported")
            .to_compile_error()
            .into(),
    }
}
