extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprReference, GenericArgument, Ident, PathArguments, Stmt};

#[proc_macro]
pub fn axvm(item: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let stmt = parse_macro_input!(item as Stmt);

    // Variables to hold the extracted identifiers
    // The return path, not pointer
    let mut dest_name: Option<Ident> = None;
    let mut func_name: Option<Ident> = None;
    let mut generic_param: Option<Expr> = None;
    // Pointer to first argument
    let mut rs1_name: Option<ExprReference> = None;
    // Pointer to second argument
    let mut rs2_name: Option<ExprReference> = None;

    if let Stmt::Semi(expr, _) = stmt {
        if let Expr::Assign(expr_assign) = expr {
            // Parse the left-hand side (lhs)
            if let Expr::Path(rd_path) = *expr_assign.left {
                if let Some(segment) = rd_path.path.segments.last() {
                    dest_name = Some(segment.ident.clone());
                }
            }

            // Parse the right-hand side (rhs)
            if let Expr::Call(expr_call) = *expr_assign.right {
                if let Expr::Path(expr_path) = *expr_call.func {
                    let path = &expr_path.path;
                    if let Some(segment) = path.segments.last() {
                        func_name = Some(segment.ident.clone());

                        if let PathArguments::AngleBracketed(angle_bracketed) = &segment.arguments {
                            if let Some(GenericArgument::Const(expr)) = angle_bracketed.args.first()
                            {
                                generic_param = Some(expr.clone());
                            } else {
                                panic!("Must provide constant generic");
                            }
                        } else {
                            panic!("No generic arguments");
                        }
                    }
                }

                // Extract function arguments
                let args = &expr_call.args;
                let mut args_iter = args.iter();

                if let Some(Expr::Reference(rs1_ref)) = args_iter.next() {
                    rs1_name = Some(rs1_ref.clone());
                } else {
                    panic!("Must provide reference as first argument");
                }

                if let Some(Expr::Reference(rs2_ref)) = args_iter.next() {
                    rs2_name = Some(rs2_ref.clone());
                } else {
                    panic!("Must provide reference as second argument");
                }
            }
        } else {
            return syn::Error::new_spanned(expr, "Expected an assignment expression")
                .to_compile_error()
                .into();
        }
    } else {
        return syn::Error::new_spanned(stmt, "Expected a statement")
            .to_compile_error()
            .into();
    }

    let rd_name = quote! {
        &mut #dest_name
    };
    let output = quote! {
        #[cfg(target_os = "zkvm")]
        unsafe {
            core::arch::asm!(
                ".insn r 0b0101011, 0x00, 0x00, {rd}, {rs1}, {rs2}",
                rd = in(reg) #rd_name,
                rs1 = in(reg) #rs1_name,
                rs2 = in(reg) #rs2_name,
            )
        }
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
