use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::ItemStruct;

pub fn create_new_struct_and_impl_hintable(ast: &ItemStruct) -> Result<TokenStream, TokenStream> {
    let name = &ast.ident;
    let name_prefix = name.to_string();
    let name_var = format!("{}Var", name_prefix);
    let name_var_ident = Ident::new(&name_var, Span::call_site());

    let fields = ast.fields.clone();

    let field_types: Vec<_> = fields
        .iter()
        .map(|field| field.ty.to_token_stream())
        .collect();

    let field_names: Vec<TokenStream> = fields
        .iter()
        .map(|field| {
            if let Some(ref field) = field.ident {
                return field.to_token_stream();
            }
            quote! {
                compile_error!("Hintable macro only supports named fields");
            }
        })
        .collect();

    let (_, ty_generics, where_clause) = ast.generics.split_for_impl();

    let impl_generics = {
        let params = &ast.generics.params;
        quote! { < C: axvm_native_compiler::prelude::Config, #params >}
    };
    let input_struct_tokens: Vec<_> = field_names
        .iter()
        .zip(field_types.iter())
        .map(|(name, field_type)| {
            quote! {
                pub #name: <#field_type as axvm_recursion::hints::Hintable<C> >::HintVariable,
            }
        })
        .collect();

    let read_tokens: Vec<_> = field_names
        .iter()
        .zip(field_types.iter())
        .map(|(name, field_type)| {
            quote! {
                let #name = <#field_type as axvm_recursion::hints::Hintable<C>>::read(builder);
            }
        })
        .collect();

    let write_tokens: Vec<_> = field_names
        .iter()
        .map(|name| {
            quote! {
                stream.extend(axvm_recursion::hints::Hintable::<C>::write(&self.#name));
            }
        })
        .collect();

    Ok(quote! {
        #[derive(axvm_native_compiler_derive::DslVariable, Debug, Clone)]
        pub struct #name_var_ident #impl_generics  {
            #(#input_struct_tokens)*
        }

        impl #impl_generics axvm_recursion::hints::Hintable<C> for #name #ty_generics #where_clause {
            type HintVariable = #name_var_ident<C>;

            fn read(builder: &mut axvm_native_compiler::prelude::Builder<C>) -> Self::HintVariable {
                #(#read_tokens)*

                #name_var_ident {
                    #(#field_names,)*
                }
            }

            fn write(&self) -> Vec<Vec<<C as axvm_native_compiler::prelude::Config>::N>> {
                let mut stream = Vec::new();

                #(#write_tokens)*

                stream
            }
        }
    })
}
