//! Copied from sp1-derive under MIT license
// We can use valida-derive directly if they ever publish it
extern crate alloc;
extern crate proc_macro;

use hints::create_new_struct_and_impl_hintable;
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, Data, DeriveInput, Fields, GenericParam, ItemStruct, Lit, Meta,
    MetaNameValue,
};

mod hints;

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
    let (_, ty_generics, _) = generics.split_for_impl();

    match &ast.data {
        Data::Struct(_) => unimplemented!("Structs are not supported yet"),
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

            let (air_arms, _arms): (Vec<_>, Vec<_>) = variants.iter().map(|(variant_name, field)| {
                let field_ty = &field.ty;
                let air_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as afs_stark_backend::Chip<SC>>::air(x)
                };
                let generate_air_proof_input_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as afs_stark_backend::Chip<SC>>::generate_air_proof_input(x)
                };
                let generate_air_proof_input_with_id_arm = quote! {
                    #name::#variant_name(x) => <#field_ty as afs_stark_backend::Chip<SC>>::generate_air_proof_input_with_id(x, air_id)
                };
                (air_arm, (generate_air_proof_input_arm, generate_air_proof_input_with_id_arm))
            }).unzip();
            let (generate_air_proof_input_arms, generate_air_proof_input_with_id_arms): (
                Vec<_>,
                Vec<_>,
            ) = _arms.into_iter().unzip();

            // Attach an extra generic SC: StarkGenericConfig to the impl_generics
            let generics = &ast.generics;
            let mut new_generics = generics.clone();
            new_generics
                .params
                .push(syn::parse_quote! { SC: afs_stark_backend::config::StarkGenericConfig });

            let (impl_generics, _, _) = new_generics.split_for_impl();

            let mut new_generics = generics.clone();
            let where_clause = new_generics.make_where_clause();
            where_clause.predicates.push(syn::parse_quote! { afs_stark_backend::config::Domain<SC>: afs_stark_backend::p3_commit::PolynomialSpace<Val = F>
            });

            quote! {
                impl #impl_generics afs_stark_backend::Chip<SC> for #name #ty_generics #where_clause {
                    fn air(&self) -> std::sync::Arc<dyn afs_stark_backend::rap::AnyRap<SC>> {
                        match self {
                            #(#air_arms,)*
                        }
                    }
                    fn generate_air_proof_input(&self) -> afs_stark_backend::prover::types::AirProofInput<SC> {
                        match self {
                            #(#generate_air_proof_input_arms,)*
                        }
                    }
                    fn generate_air_proof_input_with_id(&self, air_id: usize) -> (usize, afs_stark_backend::prover::types::AirProofInput<SC>) {
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

#[proc_macro_derive(DslVariable)]
pub fn derive_variable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident; // Struct name

    let gen = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => {
                let fields_init = fields.named.iter().map(|f| {
                    let fname = &f.ident;
                    let ftype = &f.ty;
                    let ftype_str = quote! { #ftype }.to_string();
                    if ftype_str.contains("Array") {
                        quote! {
                            #fname: Array::Dyn(builder.uninit(), builder.uninit()),
                        }
                    } else {
                        quote! {
                            #fname: <#ftype as Variable<C>>::uninit(builder),
                        }
                    }
                });

                let fields_assign = fields.named.iter().map(|f| {
                    let fname = &f.ident;
                    quote! {
                        self.#fname.assign(src.#fname.into(), builder);
                    }
                });

                let fields_assert_eq = fields.named.iter().map(|f| {
                    let fname = &f.ident;
                    let ftype = &f.ty;
                    quote! {
                        <#ftype as Variable<C>>::assert_eq(lhs.#fname, rhs.#fname, builder);
                    }
                });

                let fields_assert_ne = fields.named.iter().map(|f| {
                    let fname = &f.ident;
                    let ftype = &f.ty;
                    quote! {
                        <#ftype as Variable<C>>::assert_ne(lhs.#fname, rhs.#fname, builder);
                    }
                });

                let field_sizes = fields.named.iter().map(|f| {
                    let ftype = &f.ty;
                    quote! {
                        <#ftype as MemVariable<C>>::size_of()
                    }
                });

                let field_loads = fields.named.iter().map(|f| {
                    let fname = &f.ident;
                    let ftype = &f.ty;
                    quote! {
                        {
                            // let address = builder.eval(ptr + Usize::Const(offset));
                            self.#fname.load(ptr, index, builder);
                            index.offset += <#ftype as MemVariable<C>>::size_of();
                        }
                    }
                });

                let field_stores = fields.named.iter().map(|f| {
                    let fname = &f.ident;
                    let ftype = &f.ty;
                    quote! {
                        {
                            // let address = builder.eval(ptr + Usize::Const(offset));
                            self.#fname.store(ptr, index, builder);
                            index.offset += <#ftype as MemVariable<C>>::size_of();
                        }
                    }
                });

                quote! {
                    impl<C: Config> Variable<C> for #name<C> {
                        type Expression = Self;

                        fn uninit(builder: &mut Builder<C>) -> Self {
                            Self {
                                #(#fields_init)*
                            }
                        }

                        fn assign(&self, src: Self::Expression, builder: &mut Builder<C>) {
                            #(#fields_assign)*
                        }

                        fn assert_eq(
                            lhs: impl Into<Self::Expression>,
                            rhs: impl Into<Self::Expression>,
                            builder: &mut Builder<C>,
                        ) {
                            let lhs = lhs.into();
                            let rhs = rhs.into();
                            #(#fields_assert_eq)*
                        }

                        fn assert_ne(
                            lhs: impl Into<Self::Expression>,
                            rhs: impl Into<Self::Expression>,
                            builder: &mut Builder<C>,
                        ) {
                            let lhs = lhs.into();
                            let rhs = rhs.into();
                            #(#fields_assert_ne)*
                        }
                    }

                    impl<C: Config> MemVariable<C> for #name<C> {
                        fn size_of() -> usize {
                            let mut size = 0;
                            #(size += #field_sizes;)*
                            size
                        }

                        fn load(&self, ptr: Ptr<<C as Config>::N>,
                            index: MemIndex<<C as Config>::N>,
                            builder: &mut Builder<C>) {
                            let mut index = index;
                            #(#field_loads)*
                        }

                        fn store(&self, ptr: Ptr<<C as Config>::N>,
                                 index: MemIndex<<C as Config>::N>,
                                builder: &mut Builder<C>) {
                            let mut index = index;
                            #(#field_stores)*
                        }
                    }
                }
            }
            _ => unimplemented!(),
        },
        _ => unimplemented!(),
    };

    gen.into()
}

#[proc_macro_derive(Hintable)]
pub fn hintable_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as ItemStruct);

    let new_struct = create_new_struct_and_impl_hintable(&ast);
    match new_struct {
        Ok(new_struct) => new_struct.into(),
        Err(err) => err.into(),
    }
}

#[proc_macro_derive(UsizeOpcode, attributes(opcode_offset))]
pub fn usize_opcode_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let mut offset = None;
    for attr in ast.attrs {
        if let Ok(Meta::NameValue(MetaNameValue {
            path,
            lit: Lit::Int(lit_int),
            ..
        })) = attr.parse_meta()
        {
            if path.is_ident("opcode_offset") {
                offset = Some(lit_int.base10_parse::<usize>().unwrap());
            }
        }
    }
    let offset = offset.expect("opcode_offset attribute not found");

    let methods = quote! {
        impl UsizeOpcode for #name {
            fn default_offset() -> usize {
                #offset
            }

            fn from_usize(value: usize) -> Self {
                Self::from_repr(value.try_into().unwrap())
                    .unwrap_or_else(|| panic!("Failed to convert usize {} to opcode {}", value, stringify!(#name)))
            }

            fn as_usize(&self) -> usize {
                *self as usize
            }
        }
    };

    TokenStream::from(methods)
}
