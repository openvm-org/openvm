extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, GenericArgument, Ident, Pat, PathArguments, Stmt};

#[proc_macro_attribute]
pub fn axvm(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let stmt = parse_macro_input!(item as Stmt);

    // Variables to hold the extracted identifiers
    let mut rd_name: Option<Ident> = None;
    let mut func_name: Option<Ident> = None;
    let mut generic_param: Option<Expr> = None;
    let mut rs1_name: Option<Ident> = None;
    let mut rs2_name: Option<Ident> = None;

    if let Stmt::Local(local) = stmt {
        // Extract the variable name from the pattern
        if let Pat::Ident(pat_ident) = &local.pat {
            rd_name = Some(pat_ident.ident.clone());
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

                if let Some(Expr::Path(rs1_path)) = args_iter.next() {
                    if let Some(rs1_segment) = rs1_path.path.segments.last() {
                        rs1_name = Some(rs1_segment.ident.clone());
                    }
                }

                if let Some(Expr::Path(rs2_path)) = args_iter.next() {
                    if let Some(rs2_segment) = rs2_path.path.segments.last() {
                        rs2_name = Some(rs2_segment.ident.clone());
                    }
                }
            }
        }
    }

    let output = quote! {
        // #[cfg(target_os = "zkvm")]
        // unsafe {
        //     core::arch::asm!(
        //         ".insn r 0b0101011 0b000 0x00 {rd} {rs1} {rs2}",
        //         rd = in(reg) #rd_name,
        //         rs1 = in(reg) #rs1_name,
        //         rs2 = in(reg) #rs2_name,
        //     )
        // }
        #[cfg(not(target_os = "zkvm"))]
        {
            // Debug output
            println!("rd_name: {:?}", stringify!(#rd_name));
            println!("func_name: {:?}", stringify!(#func_name));
            println!("generic_param: {:?}", stringify!(#generic_param));
            println!("rs1_name: {:?}", stringify!(#rs1_name));
            println!("rs2_name: {:?}", stringify!(#rs2_name));
        }
    };

    // Convert the quoted tokens back to proc_macro::TokenStream
    output.into()
}
