#![feature(proc_macro_diagnostic)]

extern crate proc_macro;

use itertools::Itertools;
//use openvm_macros_common::MacroArgs;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, parse_quote, DeriveInput, Expr};

#[proc_macro_derive(ColsRef, attributes(dim))]
pub fn cols_ref(input: TokenStream) -> TokenStream {
    let derive_input: DeriveInput = parse_macro_input!(input as DeriveInput);

    let span = derive_input.ident.span().clone();
    let res = cols_ref_impl(derive_input);
    if res.is_err() {
        return syn::Error::new(span, res.err().unwrap().to_string())
            .to_compile_error()
            .into();
    } else {
        res.unwrap().into()
    }
}

fn cols_ref_impl(derive_input: DeriveInput) -> Result<proc_macro2::TokenStream, String> {
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
        return Err("Struct must have exactly one generic type parameter".to_string());
    }

    let generic_type = generic_types[0];

    match data {
        syn::Data::Struct(data_struct) => {
            let const_field_infos: Vec<FieldInfo> = data_struct
                .fields
                .iter()
                .map(|f| get_const_cols_ref_fields(f, generic_type))
                .collect::<Result<Vec<_>, String>>()
                .map_err(|e| format!("Failed to process fields. {}", e))?;

            let const_cols_ref_name = syn::Ident::new(&format!("{}Ref", ident), ident.span());
            let from_args = quote! { slice: &'a [#generic_type] };

            let struct_info = StructInfo {
                name: const_cols_ref_name,
                vis: vis.clone(),
                generic_type: generic_type.clone(),
                field_infos: const_field_infos,
                fields: data_struct.fields.clone(),
                from_args,
                derive_clone: true,
            };

            let const_cols_ref_struct = make_struct(struct_info.clone());

            let from_mut_impl = make_from_mut(struct_info)?;

            let mut_field_infos: Vec<FieldInfo> = data_struct
                .fields
                .iter()
                .map(|f| get_mut_cols_ref_fields(f, generic_type))
                .collect::<Result<Vec<_>, String>>()
                .map_err(|e| format!("Failed to process fields. {}", e))?;

            let mut_cols_ref_name = syn::Ident::new(&format!("{}RefMut", ident), ident.span());
            let from_args = quote! { slice: &'a mut [#generic_type] };

            let struct_info = StructInfo {
                name: mut_cols_ref_name,
                vis,
                generic_type: generic_type.clone(),
                field_infos: mut_field_infos,
                fields: data_struct.fields,
                from_args,
                derive_clone: false,
            };

            let mut_cols_ref_struct = make_struct(struct_info);

            Ok(quote! {
                #const_cols_ref_struct
                #from_mut_impl
                #mut_cols_ref_struct
            })
        }
        _ => Err("ColsRef can only be derived for structs".to_string()),
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

fn make_struct(struct_info: StructInfo) -> proc_macro2::TokenStream {
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
        #vis struct #name <'a, #generic_type> {
            #( pub #idents: #field_types ),*
        }

        impl<'a, #generic_type> #name<'a, #generic_type> {
            pub fn from<C: ShaConfig>(#from_args) -> Self {
                #( #prepare_subslices )*
                Self {
                    #( #idents: #initializers ),*
                }
            }

            pub fn len<C: ShaConfig>() -> usize {
                0 #( + #length_exprs )*
            }
        }
    }
}

fn make_from_mut(struct_info: StructInfo) -> Result<proc_macro2::TokenStream, String> {
    let StructInfo {
        name,
        vis,
        generic_type,
        field_infos,
        fields,
        from_args,
        derive_clone,
    } = struct_info;

    let fields = match fields {
        syn::Fields::Named(fields) => fields.named,
        _ => {
            return Err("Fields must be named".to_string());
        }
    };

    let from_mut_impl = fields
        .iter()
        .map(|f| {
            let ident = f.ident.clone().unwrap();
            match &f.ty {
                syn::Type::Path(type_path) => {
                    let first_ident = type_path.path.segments.first().unwrap().ident.to_string();
                    if first_ident.ends_with("Cols") {
                        // lifetime 'b is used in from_mut to allow more flexible lifetime of return value
                        let cols_ref_type =
                            get_const_cols_ref_type(&f.ty, &generic_type, parse_quote! { 'b });
                        Ok(quote! {
                            <#cols_ref_type>::from_mut::<C>(&other.#ident)
                        })
                    } else {
                        // Not a ColsRef type, so the type is T
                        Ok(quote! {
                            &other.#ident
                        })
                    }
                }
                syn::Type::Array(_) => {
                    // type is nested array of T
                    Ok(quote! {
                        other.#ident.view()
                    })
                }
                _ => Err(format!("Unsupported type: {:?}", f.ty)),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    let field_idents = fields
        .iter()
        .map(|f| f.ident.clone().unwrap())
        .collect_vec();

    let mut_struct_ident = format_ident!("{}Mut", name.to_string());
    let mut_struct_type: syn::Type = parse_quote! {
        #mut_struct_ident<'a, #generic_type>
    };

    Ok(parse_quote! {
        impl<'b, #generic_type> #name<'b, #generic_type> {
            pub fn from_mut<'a, C: ShaConfig>(other: &'b #mut_struct_type) -> Self
            {
                Self {
                    #( #field_idents: #from_mut_impl ),*
                }
            }
        }
    })
}

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
) -> Result<FieldInfo, String> {
    let length_var = format_ident!("{}_length", f.ident.clone().unwrap());
    let slice_var = format_ident!("{}_slice", f.ident.clone().unwrap());
    match get_const_cols_ref_type(&f.ty, generic_type, parse_quote! { 'a }) {
        Some(const_cols_ref_type) => Ok(FieldInfo {
            ty: parse_quote! {
                #const_cols_ref_type
            },
            length_expr: quote! {
                <#const_cols_ref_type>::len::<C>()
            },
            prepare_subslice: quote! {
                let #length_var = <#const_cols_ref_type>::len::<C>();
                let (#slice_var, slice) = slice.split_at(#length_var);
                let #slice_var = <#const_cols_ref_type>::from::<C>(#slice_var);
            },
            initializer: quote! {
                #slice_var
            },
        }),
        None => {
            // Not a ColsRef type, so assume it is T (the generic type) or a nested array of T
            let dims = get_dims(&f.ty).map_err(|e| {
                format!(
                    "Failed to parse the type of the field '{}'. {}",
                    f.ident.clone().unwrap(),
                    e
                )
            })?;

            if dims.is_empty() {
                // the field has type T
                Ok(FieldInfo {
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
                })
            } else {
                // nested array of T
                let ndarray_ident: syn::Ident = format_ident!("ArrayView{}", dims.len());
                let ndarray_type: syn::Type = parse_quote! {
                    ndarray::#ndarray_ident<'a, #generic_type>
                };
                Ok(FieldInfo {
                    ty: parse_quote! {
                        #ndarray_type
                    },
                    length_expr: quote! {
                        1 #(* C::#dims)*
                    },
                    prepare_subslice: quote! {
                        let #length_var = 1 #(* C::#dims)*;
                        let (#slice_var, slice) = slice.split_at(#length_var);
                        let #slice_var = ndarray::#ndarray_ident::from_shape( ( #(C::#dims),* ) , #slice_var).unwrap();
                    },
                    initializer: quote! {
                        #slice_var
                    },
                })
            }
        }
    }
}

// Prepare the fields for the mut ColsRef struct
fn get_mut_cols_ref_fields(
    f: &syn::Field,
    generic_type: &syn::TypeParam,
) -> Result<FieldInfo, String> {
    let length_var = format_ident!("{}_length", f.ident.clone().unwrap());
    let slice_var = format_ident!("{}_slice", f.ident.clone().unwrap());
    match get_mut_cols_ref_type(&f.ty, generic_type) {
        Some(mut_cols_ref_type) => Ok(FieldInfo {
            ty: parse_quote! {
                #mut_cols_ref_type
            },
            length_expr: quote! {
                <#mut_cols_ref_type>::len::<C>()
            },
            prepare_subslice: quote! {
                let #length_var = <#mut_cols_ref_type>::len::<C>();
                let (mut #slice_var, mut slice) = slice.split_at_mut(#length_var);
                let #slice_var = <#mut_cols_ref_type>::from::<C>(#slice_var);
            },
            initializer: quote! {
                #slice_var
            },
        }),
        None => {
            // Not a ColsRef type, so assume it is T (the generic type) or a nested array of T
            let dims = get_dims(&f.ty).map_err(|e| {
                format!(
                    "Failed to parse the type of the field '{}'. {}",
                    f.ident.clone().unwrap(),
                    e
                )
            })?;

            if dims.is_empty() {
                // the field has type T
                Ok(FieldInfo {
                    ty: parse_quote! {
                        &'a mut #generic_type
                    },
                    length_expr: quote! {
                        1
                    },
                    prepare_subslice: quote! {
                        let #length_var = 1;
                        let (mut #slice_var, mut slice) = slice.split_at_mut(#length_var);
                    },
                    initializer: quote! {
                        &mut #slice_var[0]
                    },
                })
            } else {
                // nested array of T
                let ndarray_ident: syn::Ident = format_ident!("ArrayViewMut{}", dims.len());
                let ndarray_type: syn::Type = parse_quote! {
                    ndarray::#ndarray_ident<'a, #generic_type>
                };
                Ok(FieldInfo {
                    ty: parse_quote! {
                        #ndarray_type
                    },
                    length_expr: quote! {
                        1 #(* C::#dims)*
                    },
                    prepare_subslice: quote! {
                        let #length_var = 1 #(* C::#dims)*;
                        let (mut #slice_var, mut slice) = slice.split_at_mut(#length_var);
                        let mut #slice_var = ndarray::#ndarray_ident::from_shape( ( #(C::#dims),* ) , #slice_var).unwrap();
                    },
                    initializer: quote! {
                        #slice_var
                    },
                })
            }
        }
    }
}

// If 'ty' is a struct that derives ColsRef, return the ColsRef struct type
// Otherwise, return None
fn get_const_cols_ref_type(
    ty: &syn::Type,
    generic_type: &syn::TypeParam,
    lifetime: syn::Lifetime,
) -> Option<syn::TypePath> {
    if let syn::Type::Path(type_path) = ty {
        type_path.path.segments.iter().last().and_then(|s| {
            if s.ident.to_string().ends_with("Cols") {
                let const_cols_ref_ident = format_ident!("{}Ref", s.ident);
                let const_cols_ref_type = parse_quote! {
                    #const_cols_ref_ident<#lifetime, #generic_type>
                };
                Some(const_cols_ref_type)
            } else {
                None
            }
        })
    } else {
        None
    }
}

// If 'ty' is a struct that derives ColsRef, return the ColsRefMut struct type
// Otherwise, return None
fn get_mut_cols_ref_type(ty: &syn::Type, generic_type: &syn::TypeParam) -> Option<syn::TypePath> {
    if let syn::Type::Path(type_path) = ty {
        type_path.path.segments.iter().last().and_then(|s| {
            if s.ident.to_string().ends_with("Cols") {
                let mut_cols_ref_ident = format_ident!("{}RefMut", s.ident);
                let mut_cols_ref_type = parse_quote! {
                    #mut_cols_ref_ident<'a, #generic_type>
                };
                Some(mut_cols_ref_type)
            } else {
                None
            }
        })
    } else {
        None
    }
}

fn get_dims(ty: &syn::Type) -> Result<Vec<Expr>, String> {
    get_dims_impl(ty).map(|dims| dims.into_iter().rev().collect())
}

fn get_dims_impl(ty: &syn::Type) -> Result<Vec<Expr>, String> {
    match ty {
        syn::Type::Array(array) => {
            let mut dims = get_dims(array.elem.as_ref())?;
            match array.len {
                syn::Expr::Path(_) => dims.push(array.len.clone()),
                _ => return Err("Array lengths must be const generic parameters.".to_string()),
            }
            Ok(dims)
        }
        syn::Type::Path(_) => Ok(Vec::new()),
        _ => Err(
            "Only generic types and (nested) arrays of generic types are supported.".to_string(),
        ),
    }
}
