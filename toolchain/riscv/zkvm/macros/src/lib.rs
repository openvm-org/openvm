extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, GenericArgument, Ident, Pat, PathArguments, Stmt, Type};

#[proc_macro_attribute]
pub fn axvm(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let stmt = parse_macro_input!(item as Stmt);

    // Variables to hold the extracted identifiers
    let mut var_name: Option<Ident> = None;
    let mut func_name: Option<Ident> = None;
    let mut generic_param: Option<Expr> = None;
    let mut arg1: Option<Ident> = None;
    let mut arg2: Option<Ident> = None;

    if let Stmt::Local(local) = stmt {
        // Extract the variable name from the pattern
        if let Pat::Ident(pat_ident) = &local.pat {
            var_name = Some(pat_ident.ident.clone());
        }

        // Extract the initialization expression
        if let Some((_eq_token, init_expr)) = &local.init {
            // Check if the expression is a function call
            if let Expr::Call(expr_call) = &**init_expr {
                // Extract the function being called
                if let Expr::Path(expr_path) = &*expr_call.func {
                    // Extract the function name and generic parameters
                    let path = &expr_path.path;
                    if let Some(segment) = path.segments.last() {
                        func_name = Some(segment.ident.clone());

                        // Extract generic parameters if any
                        if let PathArguments::AngleBracketed(angle_bracketed) = &segment.arguments {
                            let generic_arg = angle_bracketed.args.first().unwrap();
                            match generic_arg {
                                GenericArgument::Const(const_expr) => {
                                    generic_param = Some(const_expr.clone());
                                }
                                _ => panic!("Must provide constant generic"),
                            }
                        }
                    }
                }

                // Extract function arguments
                let args = &expr_call.args;
                let mut args_iter = args.iter();

                if let Some(Expr::Path(arg1_path)) = args_iter.next() {
                    if let Some(arg1_segment) = arg1_path.path.segments.last() {
                        arg1 = Some(arg1_segment.ident.clone());
                    }
                }

                if let Some(Expr::Path(arg2_path)) = args_iter.next() {
                    if let Some(arg2_segment) = arg2_path.path.segments.last() {
                        arg2 = Some(arg2_segment.ident.clone());
                    }
                }
            }
        }
    }

    // Now you have the identifiers: var_name, func_name, generic_param, arg1, arg2
    // You can use them as needed in your macro logic

    let output = quote! {
        // Debug output
        println!("var_name: {:?}", stringify!(#var_name));
        println!("func_name: {:?}", stringify!(#func_name));
        println!("generic_param: {:?}", stringify!(#generic_param));
        println!("arg1: {:?}", stringify!(#arg1));
        println!("arg2: {:?}", stringify!(#arg2));
    };

    // Convert the quoted tokens back to proc_macro::TokenStream
    output.into()
}
