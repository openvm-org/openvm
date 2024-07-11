//! Copied from sp1-derive under MIT license
// We can use valida-derive directly if they ever publish it
extern crate alloc;
extern crate proc_macro;

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
                debug_assert_eq!(self.len(), std::mem::size_of::<#name<u8 #(, #non_first_generics)*>>());
                let (prefix, shorts, _suffix) = unsafe { self.align_to::<#name #type_generics>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert_eq!(shorts.len(), 1);
                &shorts[0]
            }
        }

        impl #impl_generics core::borrow::BorrowMut<#name #type_generics> for [#type_generic] #where_clause {
            fn borrow_mut(&mut self) -> &mut #name #type_generics {
                debug_assert_eq!(self.len(), std::mem::size_of::<#name<u8 #(, #non_first_generics)*>>());
                let (prefix, shorts, _suffix) = unsafe { self.align_to_mut::<#name #type_generics>() };
                debug_assert!(prefix.is_empty(), "Alignment should match");
                debug_assert_eq!(shorts.len(), 1);
                &mut shorts[0]
            }
        }
    };

    TokenStream::from(methods)
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
