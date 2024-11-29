extern crate alloc;
extern crate proc_macro;

use itertools::{multiunzip, Itertools};
use proc_macro::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{punctuated::Punctuated, Data, Fields, GenericParam, Ident, Meta, Token};

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

// VmGenericConfig derive macro

#[proc_macro_derive(VmGenericConfig, attributes(system, extension))]
pub fn vm_generic_config_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = &ast.ident;

    let gen_name_with_uppercase_idents = |ident: &Ident| {
        let mut name = ident.to_string().chars().collect::<Vec<_>>();
        assert!(name[0].is_lowercase(), "Field name must not be capitalized");
        let res_lower = Ident::new(&name.iter().collect::<String>(), Span::call_site().into());
        name[0] = name[0].to_ascii_uppercase();
        let res_upper = Ident::new(&name.iter().collect::<String>(), Span::call_site().into());
        (res_lower, res_upper)
    };

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
            let (system_name, system_name_upper) =
                gen_name_with_uppercase_idents(&system.ident.clone().unwrap());

            let extensions = fields
                .iter()
                .filter(|f| f.attrs.iter().any(|attr| attr.path().is_ident("extension")))
                .cloned()
                .collect::<Vec<_>>();

            let mut executor_enum_fields = Vec::new();
            let mut periphery_enum_fields = Vec::new();
            let mut create_chip_complex = Vec::new();
            for &e in extensions.iter() {
                let (field_name, field_name_upper) =
                    gen_name_with_uppercase_idents(&e.ident.clone().unwrap());
                // TRACKING ISSUE:
                // We cannot just use <e.ty.to_token_stream() as VmExtension<F>>::Executor because of this: <https://github.com/rust-lang/rust/issues/85576>
                let mut executor_name = Ident::new(
                    &format!("{}Executor", e.ty.to_token_stream()),
                    Span::call_site().into(),
                );
                let mut periphery_name = Ident::new(
                    &format!("{}Periphery", e.ty.to_token_stream()),
                    Span::call_site().into(),
                );
                if let Some(attr) = e
                    .attrs
                    .iter()
                    .find(|attr| attr.path().is_ident("extension"))
                {
                    match attr.meta {
                        Meta::Path(_) => {}
                        Meta::NameValue(_) => {
                            return syn::Error::new(
                                name.span(),
                                "Only `#[extension]` or `#[extension(...)] formats are supported",
                            )
                            .to_compile_error()
                            .into()
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
                                        } else if nv.path.is_ident("periphery") {
                                            periphery_name = Ident::new(
                                                &nv.value.to_token_stream().to_string(),
                                                Span::call_site().into(),
                                            );
                                            Ok(())
                                        } else {
                                            Err("only executor and periphery keys are supported")
                                        }
                                    }
                                    _ => Err("only name = value format is supported"),
                                }
                                .expect("wrong attributes format");
                            }
                        }
                    }
                };
                executor_enum_fields.push(quote! {
                    #[any_enum]
                    #field_name_upper(#executor_name<F>),
                });
                periphery_enum_fields.push(quote! {
                    #[any_enum]
                    #field_name_upper(#periphery_name<F>),
                });
                create_chip_complex.push(quote! {
                    let complex: VmChipComplex<F, Self::Executor, Self::Periphery> = complex.extend(&self.#field_name)?;
                });
            }

            let executor_type = Ident::new(&format!("{}Executor", name), name.span());
            let periphery_type = Ident::new(&format!("{}Periphery", name), name.span());
            TokenStream::from(quote! {
                #[derive(ChipUsageGetter, Chip, InstructionExecutor, From, AnyEnum)]
                pub enum #executor_type<F: PrimeField32> {
                    #[any_enum]
                    #system_name_upper(SystemExecutor<F>),
                    #(#executor_enum_fields)*
                }

                #[derive(ChipUsageGetter, Chip, From, AnyEnum)]
                pub enum #periphery_type<F: PrimeField32> {
                    #[any_enum]
                    #system_name_upper(SystemPeriphery<F>),
                    #(#periphery_enum_fields)*
                }

                impl<F: PrimeField32> VmGenericConfig<F> for #name {
                    type Executor = #executor_type<F>;
                    type Periphery = #periphery_type<F>;

                    fn system(&self) -> &SystemConfig {
                        &self.#system_name
                    }

                    fn create_chip_complex(
                        &self,
                    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
                        let complex = self.#system_name.create_chip_complex()?;
                        #(#create_chip_complex)*
                        Ok(complex)
                    }
                }
            })
        }
        _ => syn::Error::new(name.span(), "Only structs are supported")
            .to_compile_error()
            .into(),
    }
}
