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

    let existing_generics = ast.generics.clone();
    if !existing_generics.params.is_empty() {
        return Err(quote! {
            compile_error!("Hintable macro only supports structs with no generics for now");
        });
    }
    let input_struct_tokens: Vec<_> = field_names
        .iter()
        .zip(field_types.iter())
        .map(|(name, field_type)| {
            quote! {
                pub #name: <#field_type as Hintable<C> >::HintVariable,
            }
        })
        .collect();

    let read_tokens: Vec<_> = field_names
        .iter()
        .zip(field_types.iter())
        .map(|(name, field_type)| {
            quote! {
                let #name = <#field_type as Hintable<C>>::read(builder);
            }
        })
        .collect();

    let write_tokens: Vec<_> = field_names
        .iter()
        .map(|name| {
            quote! {
                stream.extend(Hintable::<C>::write(&self.#name));
            }
        })
        .collect();

    Ok(quote! {
        #[derive(DslVariable, Debug, Clone)]
        pub struct #name_var_ident <C: Config>  {
            #(#input_struct_tokens)*
        }

        impl<C: Config> Hintable<C> for #name {
            type HintVariable = #name_var_ident<C>;

            fn read(builder: &mut Builder<C>) -> Self::HintVariable {
                #(#read_tokens)*

                #name_var_ident {
                    #(#field_names,)*
                }
            }

            fn write(&self) -> Vec<Vec<<C as Config>::N>> {
                let mut stream = Vec::new();

                #(#write_tokens)*

                stream
            }
        }
    })
}
