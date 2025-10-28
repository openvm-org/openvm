/*
 * The `ColsRef` procedural macro is used in constraint generation to create column structs that
 * have dynamic sizes.
 *
 * Note: this macro was originally created for use in the SHA-2 VM extension, where we reuse the
 * same constraint generation code for three different circuits (SHA-256, SHA-512, and SHA-384).
 * See the [SHA-2 VM extension](openvm/extensions/sha2/circuit/src/sha2_chip/air.rs) for an
 * example of how to use the `ColsRef` macro to reuse constraint generation code over multiple
 * circuits.
 *
 * This macro can also be used in other situations where we want to derive Borrow<T> for &[u8],
 * for some complicated struct T.
 */
mod utils;

use utils::*;

extern crate proc_macro;

use itertools::Itertools;
use quote::{format_ident, quote};
use syn::{parse_quote, DeriveInput};

pub fn cols_ref_impl(
    derive_input: DeriveInput,
    config: proc_macro2::Ident,
) -> proc_macro2::TokenStream {
    let DeriveInput {
        ident,
        generics,
        data,
        vis,
        ..
    } = derive_input;

    let generic_types = generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(type_param) = p {
                Some(type_param)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    if generic_types.len() != 1 {
        panic!("Struct must have exactly one generic type parameter");
    }

    let generic_type = generic_types[0];

    let const_generics = generics.const_params().map(|p| &p.ident).collect_vec();

    match data {
        syn::Data::Struct(data_struct) => {
            // Process the fields of the struct, transforming the types for use in ColsRef struct
            let const_field_infos: Vec<FieldInfo> = data_struct
                .fields
                .iter()
                .map(|f| get_const_cols_ref_fields(f, generic_type, &const_generics))
                .collect_vec();

            // The ColsRef struct is named by appending `Ref` to the struct name
            let const_cols_ref_name = syn::Ident::new(&format!("{}Ref", ident), ident.span());

            // the args to the `from` method will be different for the ColsRef and ColsRefMut
            // structs
            let from_args = quote! { slice: &'a [#generic_type] };

            // Package all the necessary information to generate the ColsRef struct
            let struct_info = StructInfo {
                name: const_cols_ref_name,
                vis: vis.clone(),
                generic_type: generic_type.clone(),
                field_infos: const_field_infos,
                fields: data_struct.fields.clone(),
                from_args,
                derive_clone: true,
            };

            // Generate the ColsRef struct
            let const_cols_ref_struct = make_struct(struct_info.clone(), &config);

            // Generate the `from_mut` method for the ColsRef struct
            let from_mut_impl = make_from_mut(struct_info, &config);

            // Process the fields of the struct, transforming the types for use in ColsRefMut struct
            let mut_field_infos: Vec<FieldInfo> = data_struct
                .fields
                .iter()
                .map(|f| get_mut_cols_ref_fields(f, generic_type, &const_generics))
                .collect_vec();

            // The ColsRefMut struct is named by appending `RefMut` to the struct name
            let mut_cols_ref_name = syn::Ident::new(&format!("{}RefMut", ident), ident.span());

            // the args to the `from` method will be different for the ColsRef and ColsRefMut
            // structs
            let from_args = quote! { slice: &'a mut [#generic_type] };

            // Package all the necessary information to generate the ColsRefMut struct
            let struct_info = StructInfo {
                name: mut_cols_ref_name,
                vis,
                generic_type: generic_type.clone(),
                field_infos: mut_field_infos,
                fields: data_struct.fields,
                from_args,
                derive_clone: false,
            };

            // Generate the ColsRefMut struct
            let mut_cols_ref_struct = make_struct(struct_info, &config);

            quote! {
                #const_cols_ref_struct
                #from_mut_impl
                #mut_cols_ref_struct
            }
        }
        _ => panic!("ColsRef can only be derived for structs"),
    }
}

#[derive(Debug, Clone)]
struct StructInfo {
    name: syn::Ident,
    vis: syn::Visibility,
    generic_type: syn::TypeParam,
    field_infos: Vec<FieldInfo>,
    fields: syn::Fields,
    from_args: proc_macro2::TokenStream,
    derive_clone: bool,
}

// Generate the ColsRef and ColsRefMut structs, depending on the value of `struct_info`
// This function is meant to reduce code duplication between the code needed to generate the two
// structs Notable differences between the two structs are:
//   - the types of the fields
//   - ColsRef derives Clone, but ColsRefMut cannot (since it stores mutable references)
//   - the `from` method parameter is a reference to a slice for ColsRef and a mutable reference to
//     a slice for ColsRefMut
fn make_struct(struct_info: StructInfo, config: &proc_macro2::Ident) -> proc_macro2::TokenStream {
    let StructInfo {
        name,
        vis,
        generic_type,
        field_infos,
        fields,
        from_args,
        derive_clone,
    } = struct_info;

    let field_types = field_infos.iter().map(|f| &f.ty).collect_vec();
    let length_exprs = field_infos.iter().map(|f| &f.length_expr).collect_vec();
    let prepare_subslices = field_infos
        .iter()
        .map(|f| &f.prepare_subslice)
        .collect_vec();
    let initializers = field_infos.iter().map(|f| &f.initializer).collect_vec();

    let idents = fields.iter().map(|f| &f.ident).collect_vec();

    let clone_impl = if derive_clone {
        quote! {
            #[derive(Clone)]
        }
    } else {
        quote! {}
    };

    quote! {
        #clone_impl
        #[derive(Debug)]
        #vis struct #name <'a, #generic_type> {
            #( pub #idents: #field_types ),*
        }

        impl<'a, #generic_type> #name<'a, #generic_type> {
            pub fn from<C: #config>(#from_args) -> Self {
                #( #prepare_subslices )*
                Self {
                    #( #idents: #initializers ),*
                }
            }

            // Returns number of cells in the struct (where each cell has type T).
            // This method should only be called if the struct has no primitive types (i.e. for columns structs).
            pub const fn width<C: #config>() -> usize {
                0 #( + #length_exprs )*
            }
        }
    }
}

// Generate the `from_mut` method for the ColsRef struct
fn make_from_mut(struct_info: StructInfo, config: &proc_macro2::Ident) -> proc_macro2::TokenStream {
    let StructInfo {
        name,
        vis: _,
        generic_type,
        field_infos: _,
        fields,
        from_args: _,
        derive_clone: _,
    } = struct_info;

    let from_mut_impl = fields
        .iter()
        .map(|f| {
            let ident = f.ident.clone().unwrap();

            let derives_aligned_borrow = f
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("aligned_borrow"));

            let is_array = matches!(f.ty, syn::Type::Array(_));

            if is_array {
                // calling view() on ArrayViewMut returns an ArrayView
                quote! {
                    other.#ident.view()
                }
            } else if derives_aligned_borrow {
                // implicitly converts a mutable reference to an immutable reference, so leave the
                // field value unchanged
                quote! {
                    other.#ident
                }
            } else if is_columns_struct(&f.ty) {
                // lifetime 'b is used in from_mut to allow more flexible lifetime of return value
                let cols_ref_type =
                    get_const_cols_ref_type(&f.ty, &generic_type, parse_quote! { 'b });
                // Recursively call `from_mut` on the ColsRef field
                quote! {
                    <#cols_ref_type>::from_mut::<C>(&other.#ident)
                }
            } else if is_generic_type(&f.ty, &generic_type) {
                // implicitly converts a mutable reference to an immutable reference, so leave the
                // field value unchanged
                quote! {
                    &other.#ident
                }
            } else {
                panic!("Unsupported field type (in make_from_mut): {:?}", f.ty);
            }
        })
        .collect_vec();

    let field_idents = fields
        .iter()
        .map(|f| f.ident.clone().unwrap())
        .collect_vec();

    let mut_struct_ident = format_ident!("{}Mut", name.to_string());
    let mut_struct_type: syn::Type = parse_quote! {
        #mut_struct_ident<'a, #generic_type>
    };

    parse_quote! {
        // lifetime 'b is used in from_mut to allow more flexible lifetime of return value
        impl<'b, #generic_type> #name<'b, #generic_type> {
            pub fn from_mut<'a, C: #config>(other: &'b #mut_struct_type) -> Self
            {
                Self {
                    #( #field_idents: #from_mut_impl ),*
                }
            }
        }
    }
}

// Information about a field that is used to generate the ColsRef and ColsRefMut structs
// See the `make_struct` function to see how this information is used
#[derive(Debug, Clone)]
struct FieldInfo {
    // type for struct definition
    ty: syn::Type,
    // an expr calculating the length of the field
    length_expr: proc_macro2::TokenStream,
    // prepare a subslice of the slice to be used in the 'from' method
    prepare_subslice: proc_macro2::TokenStream,
    // an expr used in the Self initializer in the 'from' method
    // may refer to the subslice declared in prepare_subslice
    initializer: proc_macro2::TokenStream,
}

// Prepare the fields for the const ColsRef struct
fn get_const_cols_ref_fields(
    f: &syn::Field,
    generic_type: &syn::TypeParam,
    const_generics: &[&syn::Ident],
) -> FieldInfo {
    let length_var = format_ident!("{}_length", f.ident.clone().unwrap());
    let slice_var = format_ident!("{}_slice", f.ident.clone().unwrap());

    let derives_aligned_borrow = f
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("aligned_borrow"));

    let is_array = matches!(f.ty, syn::Type::Array(_));

    if is_array {
        let ArrayInfo { dims, elem_type } = get_array_info(&f.ty, const_generics);
        debug_assert!(
            !dims.is_empty(),
            "Array field must have at least one dimension"
        );

        let ndarray_ident: syn::Ident = format_ident!("ArrayView{}", dims.len());
        let ndarray_type: syn::Type = parse_quote! {
            ndarray::#ndarray_ident<'a, #elem_type>
        };

        // dimensions of the array in terms of number of cells
        let dim_exprs = dims
            .iter()
            .map(|d| match d {
                // need to prepend C:: for const generic array dimensions
                Dimension::ConstGeneric(expr) => quote! { C::#expr },
                Dimension::Other(expr) => quote! { #expr },
            })
            .collect_vec();

        if derives_aligned_borrow {
            let length_expr = quote! {
                <#elem_type>::width() #(* #dim_exprs)*
            };

            FieldInfo {
                ty: parse_quote! {
                    #ndarray_type
                },
                length_expr: length_expr.clone(),
                prepare_subslice: quote! {
                    let (#slice_var, slice) = slice.split_at(#length_expr);
                    let #slice_var: &[#elem_type] = unsafe { &*(#slice_var as *const [T] as *const [#elem_type]) };
                    let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(#dim_exprs),* ) , #slice_var).unwrap();
                },
                initializer: quote! {
                    #slice_var
                },
            }
        } else if is_columns_struct(&elem_type) {
            panic!("Arrays of columns structs are currently not supported");
        } else if is_generic_type(&elem_type, generic_type) {
            let length_expr = quote! {
                1 #(* #dim_exprs)*
            };
            FieldInfo {
                ty: parse_quote! {
                    #ndarray_type
                },
                length_expr: length_expr.clone(),
                prepare_subslice: quote! {
                    let (#slice_var, slice) = slice.split_at(#length_expr);
                    let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(#dim_exprs),* ) , #slice_var).unwrap();
                },
                initializer: quote! {
                    #slice_var
                },
            }
        } else if is_primitive_type(&elem_type) {
            FieldInfo {
                ty: parse_quote! {
                    &'a #elem_type
                },
                // Columns structs won't ever have primitive types, but this macro can be used on
                // other structs as well, to make it easy to borrow a struct from &[u8].
                // We just set length = 0 knowing that calling the width() method is undefined if
                // the struct has a primitive type.
                length_expr: quote! {
                    0
                },
                prepare_subslice: quote! {
                    let (#slice_var, slice) = slice.split_at(std::mem::size_of::<#elem_type>() #(* #dim_exprs)*);
                    let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(#dim_exprs),* ) , #slice_var).unwrap();
                },
                initializer: quote! {
                    #slice_var
                },
            }
        } else {
            panic!("Unsupported field type (in get_const_cols_ref_fields): {:?}", f.ty);
        }
    } else if derives_aligned_borrow {
        // treat the field as a struct that derives AlignedBorrow (and doesn't depend on the config)
        let f_ty = &f.ty;
        FieldInfo {
            ty: parse_quote! {
                &'a #f_ty
            },
            length_expr: quote! {
                <#f_ty>::width()
            },
            prepare_subslice: quote! {
                let #length_var = <#f_ty>::width();
                let (#slice_var, slice) = slice.split_at(#length_var);
            },
            initializer: quote! {
                {
                    use core::borrow::Borrow;
                    #slice_var.borrow()
                }
            },
        }
    } else if is_columns_struct(&f.ty) {
        let const_cols_ref_type = get_const_cols_ref_type(&f.ty, generic_type, parse_quote! { 'a });
        FieldInfo {
            ty: parse_quote! {
                #const_cols_ref_type
            },
            length_expr: quote! {
                <#const_cols_ref_type>::width::<C>()
            },
            prepare_subslice: quote! {
                let #length_var = <#const_cols_ref_type>::width::<C>();
                let (#slice_var, slice) = slice.split_at(#length_var);
                let #slice_var = <#const_cols_ref_type>::from::<C>(#slice_var);
            },
            initializer: quote! {
            #slice_var
            },
        }
    } else if is_generic_type(&f.ty, generic_type) {
        FieldInfo {
            ty: parse_quote! {
                &'a #generic_type
            },
            length_expr: quote! {
                1
            },
            prepare_subslice: quote! {
                let #length_var = 1;
                let (#slice_var, slice) = slice.split_at(#length_var);
            },
            initializer: quote! {
                &#slice_var[0]
            },
        }
    } else {
        panic!("Unsupported field type (in get_mut_cols_ref_fields): {:?}", f.ty);
    }
}

// Prepare the fields for the mut ColsRef struct
fn get_mut_cols_ref_fields(
    f: &syn::Field,
    generic_type: &syn::TypeParam,
    const_generics: &[&syn::Ident],
) -> FieldInfo {
    let length_var = format_ident!("{}_length", f.ident.clone().unwrap());
    let slice_var = format_ident!("{}_slice", f.ident.clone().unwrap());

    let derives_aligned_borrow = f
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("aligned_borrow"));

    let is_array = matches!(f.ty, syn::Type::Array(_));

    if is_array {
        let ArrayInfo { dims, elem_type } = get_array_info(&f.ty, const_generics);
        debug_assert!(
            !dims.is_empty(),
            "Array field must have at least one dimension"
        );

        let ndarray_ident: syn::Ident = format_ident!("ArrayViewMut{}", dims.len());
        let ndarray_type: syn::Type = parse_quote! {
            ndarray::#ndarray_ident<'a, #elem_type>
        };

        // dimensions of the array in terms of number of cells
        let dim_exprs = dims
            .iter()
            .map(|d| match d {
                // need to prepend C:: for const generic array dimensions
                Dimension::ConstGeneric(expr) => quote! { C::#expr },
                Dimension::Other(expr) => quote! { #expr },
            })
            .collect_vec();

        if derives_aligned_borrow {
            let length_expr = quote! {
                <#elem_type>::width() #(* #dim_exprs)*
            };

            FieldInfo {
                ty: parse_quote! {
                    #ndarray_type
                },
                length_expr: length_expr.clone(),
                prepare_subslice: quote! {
                    let (#slice_var, slice) = slice.split_at_mut (#length_expr);
                    let #slice_var: &mut [#elem_type] = unsafe { &mut *(#slice_var as *mut [T] as *mut [#elem_type]) };
                    let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(#dim_exprs),* ) , #slice_var).unwrap();
                },
                initializer: quote! {
                    #slice_var
                },
            }
        } else if is_columns_struct(&elem_type) {
            panic!("Arrays of columns structs are currently not supported");
        } else if is_generic_type(&elem_type, generic_type) {
            let length_expr = quote! {
                1 #(* #dim_exprs)*
            };
            FieldInfo {
                ty: parse_quote! {
                    #ndarray_type
                },
                length_expr: length_expr.clone(),
                prepare_subslice: quote! {
                    let (#slice_var, slice) = slice.split_at_mut(#length_expr);
                    let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(#dim_exprs),* ) , #slice_var).unwrap();
                },
                initializer: quote! {
                    #slice_var
                },
            }
        } else if is_primitive_type(&elem_type) {
            FieldInfo {
                ty: parse_quote! {
                    &'a mut #elem_type
                },
                // Columns structs won't ever have primitive types, but this macro can be used on
                // other structs as well, to make it easy to borrow a struct from &[u8].
                // We just set length = 0 knowing that calling the width() method is undefined if
                // the struct has a primitive type.
                length_expr: quote! {
                    0
                },
                prepare_subslice: quote! {
                    let (#slice_var, slice) = slice.split_at_mut(std::mem::size_of::<#elem_type>() #(* #dim_exprs)*);
                    let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(#dim_exprs),* ) , #slice_var).unwrap();
                },
                initializer: quote! {
                    #slice_var
                },
            }
        } else {
            panic!("Unsupported field type (in get_mut_cols_ref_fields): {:?}", f.ty);
        }
    } else if derives_aligned_borrow {
        // treat the field as a struct that derives AlignedBorrow (and doesn't depend on the config)
        let f_ty = &f.ty;
        FieldInfo {
            ty: parse_quote! {
                &'a mut #f_ty
            },
            length_expr: quote! {
                <#f_ty>::width()
            },
            prepare_subslice: quote! {
                let #length_var = <#f_ty>::width();
                let (#slice_var, slice) = slice.split_at_mut(#length_var);
            },
            initializer: quote! {
                {
                    use core::borrow::BorrowMut;
                    #slice_var.borrow_mut()
                }
            },
        }
    } else if is_columns_struct(&f.ty) {
        let mut_cols_ref_type = get_mut_cols_ref_type(&f.ty, generic_type);
        FieldInfo {
            ty: parse_quote! {
                #mut_cols_ref_type
            },
            length_expr: quote! {
                <#mut_cols_ref_type>::width::<C>()
            },
            prepare_subslice: quote! {
                let #length_var = <#mut_cols_ref_type>::width::<C>();
                let (#slice_var, slice) = slice.split_at_mut(#length_var);
                let #slice_var = <#mut_cols_ref_type>::from::<C>(#slice_var);
            },
            initializer: quote! {
                #slice_var
            },
        }
    } else if is_generic_type(&f.ty, generic_type) {
        FieldInfo {
            ty: parse_quote! {
                &'a mut #generic_type
            },
            length_expr: quote! {
                1
            },
            prepare_subslice: quote! {
                let #length_var = 1;
                let (#slice_var, slice) = slice.split_at_mut(#length_var);
            },
            initializer: quote! {
                &mut #slice_var[0]
            },
        }
    } else {
        panic!("Unsupported field type (in get_mut_cols_ref_fields): {:?}", f.ty);
    }
}

// Helper functions

fn is_columns_struct(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        type_path
            .path
            .segments
            .iter()
            .last()
            .map(|s| s.ident.to_string().ends_with("Cols"))
            .unwrap_or(false)
    } else {
        false
    }
}

// If 'ty' is a struct that derives ColsRef, return the ColsRef struct type
// Otherwise, return None
fn get_const_cols_ref_type(
    ty: &syn::Type,
    generic_type: &syn::TypeParam,
    lifetime: syn::Lifetime,
) -> syn::TypePath {
    if !is_columns_struct(ty) {
        panic!("Expected a columns struct, got {:?}", ty);
    }

    if let syn::Type::Path(type_path) = ty {
        let s = type_path.path.segments.iter().last().unwrap();
        if s.ident.to_string().ends_with("Cols") {
            let const_cols_ref_ident = format_ident!("{}Ref", s.ident);
            let const_cols_ref_type = parse_quote! {
                #const_cols_ref_ident<#lifetime, #generic_type>
            };
            const_cols_ref_type
        } else {
            panic!("is_columns_struct returned true for type {:?} but the last segment is not a columns struct", ty);
        }
    } else {
        panic!(
            "is_columns_struct returned true but the type {:?} is not a path",
            ty
        );
    }
}

// If 'ty' is a struct that derives ColsRef, return the ColsRefMut struct type
// Otherwise, return None
fn get_mut_cols_ref_type(ty: &syn::Type, generic_type: &syn::TypeParam) -> syn::TypePath {
    if !is_columns_struct(ty) {
        panic!("Expected a columns struct, got {:?}", ty);
    }

    if let syn::Type::Path(type_path) = ty {
        let s = type_path.path.segments.iter().last().unwrap();
        if s.ident.to_string().ends_with("Cols") {
            let mut_cols_ref_ident = format_ident!("{}RefMut", s.ident);
            let mut_cols_ref_type = parse_quote! {
                #mut_cols_ref_ident<'a, #generic_type>
            };
            mut_cols_ref_type
        } else {
            panic!("is_columns_struct returned true for type {:?} but the last segment is not a columns struct", ty);
        }
    } else {
        panic!(
            "is_columns_struct returned true but the type {:?} is not a path",
            ty
        );
    }
}

fn is_generic_type(ty: &syn::Type, generic_type: &syn::TypeParam) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if type_path.path.segments.len() == 1 {
            type_path
                .path
                .segments
                .iter()
                .last()
                .map(|s| s.ident == generic_type.ident)
                .unwrap_or(false)
        } else {
            false
        }
    } else {
        false
    }
}
