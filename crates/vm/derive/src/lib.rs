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
    let (_, ty_generics, _) = ast.generics.split_for_impl();
    let mut generics = ast.generics.clone();

    // Check if first generic is 'F'
    let needs_f = match generics.params.first() {
        Some(GenericParam::Type(type_param)) => type_param.ident != "F",
        Some(_) => true, // First param is lifetime or const, so we need F
        None => true,    // No generics at all, so we need F
    };
    if needs_f {
        // Create new F generic parameter
        let f_param: GenericParam = syn::parse_quote!(F);

        // Insert at the beginning
        generics.params.insert(0, f_param);
    }
    let (impl_generics, _, _) = generics.split_for_impl();

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
                        memory: &mut ::openvm_circuit::system::memory::MemoryController<F>,
                        streams: &mut ::openvm_circuit::arch::Streams<F>,
                        rng: &mut ::rand::rngs::StdRng,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        from_state: ::openvm_circuit::arch::ExecutionState<u32>,
                    ) -> ::openvm_circuit::arch::Result<::openvm_circuit::arch::ExecutionState<u32>> {
                        self.0.execute(memory, streams, rng, instruction, from_state)
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
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let (execute_arms, get_opcode_name_arms): (Vec<_>, Vec<_>) =
                multiunzip(variants.iter().map(|(variant_name, field)| {
                    let field_ty = &field.ty;
                    let execute_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InstructionExecutor<#first_ty_generic>>::execute(x, memory, streams, rng, instruction, from_state)
                    };
                    let get_opcode_name_arm = quote! {
                        #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InstructionExecutor<#first_ty_generic>>::get_opcode_name(x, opcode)
                    };

                    (execute_arm, get_opcode_name_arm)
                }));
            quote! {
                impl #impl_generics ::openvm_circuit::arch::InstructionExecutor<#first_ty_generic> for #name #ty_generics {
                    fn execute(
                        &mut self,
                        memory: &mut ::openvm_circuit::system::memory::MemoryController<#first_ty_generic>,
                        streams: &mut ::openvm_circuit::arch::Streams<F>,
                        rng: &mut ::rand::rngs::StdRng,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<#first_ty_generic>,
                        from_state: ::openvm_circuit::arch::ExecutionState<u32>,
                    ) -> ::openvm_circuit::arch::Result<::openvm_circuit::arch::ExecutionState<u32>> {
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

#[proc_macro_derive(TraceStep)]
pub fn trace_step_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let (_, ty_generics, _) = ast.generics.split_for_impl();
    let mut generics = ast.generics.clone();

    // Check if first generic is 'F'
    let needs_f = match generics.params.first() {
        Some(GenericParam::Type(type_param)) => type_param.ident != "F",
        Some(_) => true, // First param is lifetime or const, so we need F
        None => true,    // No generics at all, so we need F
    };
    if needs_f {
        // Create new F generic parameter
        let f_param: GenericParam =
            syn::parse_quote!(F: ::openvm_stark_backend::p3_field::PrimeField32);

        // Insert at the beginning
        generics.params.insert(0, f_param);
    }
    let need_ctx = if generics.params.len() >= 2 {
        match &generics.params[2] {
            GenericParam::Type(type_param) => type_param.ident != "CTX",
            _ => true,
        }
    } else {
        true
    };
    if need_ctx {
        // Create new F generic parameter
        let ctx_param: GenericParam = syn::parse_quote!(CTX);

        // Insert at the beginning
        generics.params.insert(0, ctx_param);
    }
    let (impl_generics, _, _) = generics.split_for_impl();

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
            quote! {
                impl #impl_generics ::openvm_circuit::arch::TraceStep<F, CTX> for #name #ty_generics {
                    type RecordLayout = <#inner_ty as ::openvm_circuit::arch::TraceStep<F, CTX>>::RecordLayout;
                    type RecordMut<'a> = <#inner_ty as ::openvm_circuit::arch::TraceStep<F, CTX>>::RecordMut<'a>;

                    fn execute<'buf, RA>(
                        &mut self,
                        state: ::openvm_circuit::arch::execution::VmStateMut<F, ::openvm_circuit::system::memory::online::TracingMemory<F>, CTX>,
                        instruction: &::openvm_instructions::instruction::Instruction<F>,
                        arena: &'buf mut RA,
                    ) -> ::openvm_circuit::arch::Result<()>
                    where
                        RA: ::openvm_circuit::arch::RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
                    {
                        self.0.execute(state, instruction, arena)
                    }

                    fn get_opcode_name(&self, opcode: usize) -> String {
                        ::openvm_circuit::arch::TraceStep::<F, CTX>::get_opcode_name(&self.0, opcode)
                    }
                }
            }
                .into()
        }
        _ => unimplemented!(),
    }
}

#[proc_macro_derive(TraceFiller)]
pub fn trace_filler_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let name = &ast.ident;
    let (_, ty_generics, _) = ast.generics.split_for_impl();
    let mut generics = ast.generics.clone();

    // Check if first generic is 'F'
    let needs_f = match generics.params.first() {
        Some(GenericParam::Type(type_param)) => type_param.ident != "F",
        Some(_) => true, // First param is lifetime or const, so we need F
        None => true,    // No generics at all, so we need F
    };
    if needs_f {
        // Create new F generic parameter
        let f_param: GenericParam =
            syn::parse_quote!(F: ::openvm_stark_backend::p3_field::PrimeField32);

        // Insert at the beginning
        generics.params.insert(0, f_param);
    }
    let need_ctx = if generics.params.len() >= 2 {
        match &generics.params[2] {
            GenericParam::Type(type_param) => type_param.ident != "CTX",
            _ => true,
        }
    } else {
        true
    };
    if need_ctx {
        // Create new F generic parameter
        let ctx_param: GenericParam = syn::parse_quote!(CTX);

        // Insert at the beginning
        generics.params.insert(0, ctx_param);
    }
    let (impl_generics, _, _) = generics.split_for_impl();

    match &ast.data {
        Data::Struct(inner) => {
            // Check if the struct has only one unnamed field
            match &inner.fields {
                Fields::Unnamed(fields) => {
                    if fields.unnamed.len() != 1 {
                        panic!("Only one unnamed field is supported");
                    }
                    fields.unnamed.first().unwrap().ty.clone()
                }
                _ => panic!("Only unnamed fields are supported"),
            };
            quote! {
                impl #impl_generics ::openvm_circuit::arch::TraceFiller<F, CTX> for #name #ty_generics {
                    fn fill_trace(
                        &self,
                        mem_helper: &::openvm_circuit::system::memory::MemoryAuxColsFactory<F>,
                        trace: &mut ::openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>,
                        rows_used: usize,
                    ) where
                        Self: Send + Sync,
                        F: Send + Sync + Clone,
                    {
                        ::openvm_circuit::arch::TraceFiller::<F, CTX>::fill_trace(&self.0, mem_helper, trace, rows_used);
                    }

                    fn fill_trace_row(&self, mem_helper: &::openvm_circuit::system::memory::MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
                        ::openvm_circuit::arch::TraceFiller::<F, CTX>::fill_trace_row(&self.0, mem_helper, row_slice);
                    }

                    fn fill_dummy_trace_row(&self, mem_helper: &::openvm_circuit::system::memory::MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
                        ::openvm_circuit::arch::TraceFiller::<F, CTX>::fill_dummy_trace_row(&self.0, mem_helper, row_slice);
                    }
                }
            }
                .into()
        }
        _ => unimplemented!(),
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
                    #[inline(always)]
                    fn pre_compute_size(&self) -> usize {
                        self.0.pre_compute_size()
                    }
                    #[inline(always)]
                    fn pre_compute_e1<Ctx>(
                        &self,
                        pc: u32,
                        inst: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> ::openvm_circuit::arch::execution::Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::E1ExecutionCtx, {
                        self.0.pre_compute_e1(pc, inst, data)
                    }

                    fn set_trace_height(&mut self, height: usize) {
                        self.0.set_trace_buffer_height(height);
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
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let pre_compute_size_arms = variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic>>::pre_compute_size(x)
                }
            }).collect::<Vec<_>>();
            let pre_compute_e1_arms = variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic>>::pre_compute_e1(x, pc, instruction, data)
                }
            }).collect::<Vec<_>>();
            let set_trace_height_arms = variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic>>::set_trace_height(x, height)
                }
            }).collect::<Vec<_>>();

            quote! {
                impl #impl_generics ::openvm_circuit::arch::InsExecutorE1<#first_ty_generic> for #name #ty_generics {
                    #[inline(always)]
                    fn pre_compute_size(&self) -> usize {
                        match self {
                            #(#pre_compute_size_arms,)*
                        }
                    }

                    #[inline(always)]
                    fn pre_compute_e1<Ctx>(
                        &self,
                        pc: u32,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> ::openvm_circuit::arch::execution::Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::E1ExecutionCtx, {
                        match self {
                            #(#pre_compute_e1_arms,)*
                        }
                    }

                    fn set_trace_height(
                        &mut self,
                        height: usize,
                    ) {
                        match self {
                            #(#set_trace_height_arms,)*
                        }
                    }
                }
            }
            .into()
        }
        Data::Union(_) => unimplemented!("Unions are not supported"),
    }
}

#[proc_macro_derive(InsExecutorE2)]
pub fn ins_executor_e2_executor_derive(input: TokenStream) -> TokenStream {
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
                .push(syn::parse_quote! { #inner_ty: ::openvm_circuit::arch::InsExecutorE2<F> });
            quote! {
                impl #impl_generics ::openvm_circuit::arch::InsExecutorE2<F> for #name #ty_generics #where_clause {
                    #[inline(always)]
                    fn e2_pre_compute_size(&self) -> usize {
                        self.0.e2_pre_compute_size()
                    }
                    #[inline(always)]
                    fn pre_compute_e2<Ctx>(
                        &self,
                        chip_idx: usize,
                        pc: u32,
                        inst: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> ::openvm_circuit::arch::execution::Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::E2ExecutionCtx, {
                        self.0.pre_compute_e2(chip_idx, pc, inst, data)
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
            // Use full path ::openvm_circuit... so it can be used either within or outside the vm
            // crate. Assume F is already generic of the field.
            let pre_compute_size_arms = variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE2<#first_ty_generic>>::e2_pre_compute_size(x)
                }
            }).collect::<Vec<_>>();
            let pre_compute_e2_arms = variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                quote! {
                    #name::#variant_name(x) => <#field_ty as ::openvm_circuit::arch::InsExecutorE2<#first_ty_generic>>::pre_compute_e2(x, chip_idx, pc, instruction, data)
                }
            }).collect::<Vec<_>>();

            quote! {
                impl #impl_generics ::openvm_circuit::arch::InsExecutorE2<#first_ty_generic> for #name #ty_generics {
                    #[inline(always)]
                    fn e2_pre_compute_size(&self) -> usize {
                        match self {
                            #(#pre_compute_size_arms,)*
                        }
                    }

                    #[inline(always)]
                    fn pre_compute_e2<Ctx>(
                        &self,
                        chip_idx: usize,
                        pc: u32,
                        instruction: &::openvm_circuit::arch::instructions::instruction::Instruction<F>,
                        data: &mut [u8],
                    ) -> ::openvm_circuit::arch::execution::Result<::openvm_circuit::arch::ExecuteFunc<F, Ctx>>
                    where
                        Ctx: ::openvm_circuit::arch::execution_mode::E2ExecutionCtx, {
                        match self {
                            #(#pre_compute_e2_arms,)*
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

// VmConfig derive macro
#[derive(Debug)]
enum Source {
    System(Ident),
    Config(Ident),
}

#[proc_macro_derive(VmConfig, attributes(system, config, extension))]
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

            let source = fields
                .iter()
                .filter_map(|f| {
                    if f.attrs.iter().any(|attr| attr.path().is_ident("system")) {
                        Some(Source::System(f.ident.clone().unwrap()))
                    } else if f.attrs.iter().any(|attr| attr.path().is_ident("config")) {
                        Some(Source::Config(f.ident.clone().unwrap()))
                    } else {
                        None
                    }
                })
                .exactly_one()
                .expect("Exactly one field must have #[system] or #[config] attribute");
            let (source_name, source_name_upper) = match &source {
                Source::System(ident) | Source::Config(ident) => {
                    gen_name_with_uppercase_idents(ident)
                }
            };

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
                    let complex: ::openvm_circuit::arch::VmChipComplex<F, Self::Executor, Self::Periphery> = complex.extend(&self.#field_name)?;
                });
            }

            let (source_executor_type, source_periphery_type) = match &source {
                Source::System(_) => (
                    quote! { ::openvm_circuit::arch::SystemExecutor },
                    quote! { ::openvm_circuit::arch::SystemPeriphery },
                ),
                Source::Config(field_ident) => {
                    let field_type = fields
                        .iter()
                        .find(|f| f.ident.as_ref() == Some(field_ident))
                        .map(|f| &f.ty)
                        .expect("Field not found");

                    let executor_type = format!("{}Executor", quote!(#field_type));
                    let periphery_type = format!("{}Periphery", quote!(#field_type));

                    let executor_ident = Ident::new(&executor_type, field_ident.span());
                    let periphery_ident = Ident::new(&periphery_type, field_ident.span());

                    (quote! { #executor_ident }, quote! { #periphery_ident })
                }
            };

            let executor_type = Ident::new(&format!("{}Executor", name), name.span());
            let periphery_type = Ident::new(&format!("{}Periphery", name), name.span());

            TokenStream::from(quote! {
                #[derive(::openvm_circuit::circuit_derive::ChipUsageGetter, ::openvm_circuit::circuit_derive::Chip, ::openvm_circuit::derive::InstructionExecutor, ::openvm_circuit::derive::InsExecutorE1, ::openvm_circuit::derive::InsExecutorE2, ::derive_more::derive::From, ::openvm_circuit::derive::AnyEnum)]
                pub enum #executor_type<F: PrimeField32> {
                    #[any_enum]
                    #source_name_upper(#source_executor_type<F>),
                    #(#executor_enum_fields)*
                }

                #[derive(::openvm_circuit::circuit_derive::ChipUsageGetter, ::openvm_circuit::circuit_derive::Chip, ::derive_more::derive::From, ::openvm_circuit::derive::AnyEnum)]
                pub enum #periphery_type<F: PrimeField32> {
                    #[any_enum]
                    #source_name_upper(#source_periphery_type<F>),
                    #(#periphery_enum_fields)*
                }

                impl<F: PrimeField32> ::openvm_circuit::arch::VmConfig<F> for #name {
                    type Executor = #executor_type<F>;
                    type Periphery = #periphery_type<F>;

                    fn system(&self) -> &::openvm_circuit::arch::SystemConfig {
                        ::openvm_circuit::arch::VmConfig::<F>::system(&self.#source_name)
                    }
                    fn system_mut(&mut self) -> &mut ::openvm_circuit::arch::SystemConfig {
                        ::openvm_circuit::arch::VmConfig::<F>::system_mut(&mut self.#source_name)
                    }

                    fn create_chip_complex(
                        &self,
                    ) -> Result<::openvm_circuit::arch::VmChipComplex<F, Self::Executor, Self::Periphery>, ::openvm_circuit::arch::VmInventoryError> {
                        let complex = self.#source_name.create_chip_complex()?;
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
