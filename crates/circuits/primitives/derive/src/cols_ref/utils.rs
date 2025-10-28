use syn::{Expr, ExprBlock, ExprPath, Ident, Stmt, Type, TypePath};

pub fn is_primitive_type(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { path, .. }) if path.segments.len() == 1 => {
            matches!(
                path.segments[0].ident.to_string().as_str(),
                "u8" | "u16"
                    | "u32"
                    | "u64"
                    | "u128"
                    | "usize"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "i128"
                    | "isize"
                    | "f32"
                    | "f64"
                    | "bool"
                    | "char"
            )
        }
        _ => false,
    }
}

// Type of array dimension
pub enum Dimension {
    ConstGeneric(Expr),
    Other(Expr),
}

// Describes a nested array
pub struct ArrayInfo {
    pub dims: Vec<Dimension>,
    pub elem_type: Type,
}

pub fn get_array_info(ty: &Type, const_generics: &[&Ident]) -> ArrayInfo {
    let dims = get_dims(ty, const_generics);
    let elem_type = get_elem_type(ty);
    ArrayInfo { dims, elem_type }
}

fn get_elem_type(ty: &Type) -> Type {
    match ty {
        Type::Array(array) => get_elem_type(array.elem.as_ref()),
        Type::Path(_) => ty.clone(),
        _ => panic!("Unsupported type: {:?}", ty),
    }
}

// Get a vector of the dimensions of the array
// Each dimension is either a constant generic or a literal integer value
fn get_dims(ty: &Type, const_generics: &[&Ident]) -> Vec<Dimension> {
    get_dims_impl(ty, const_generics)
        .into_iter()
        .rev()
        .collect()
}

fn get_dims_impl(ty: &Type, const_generics: &[&Ident]) -> Vec<Dimension> {
    match ty {
        Type::Array(array) => {
            let mut dims = get_dims_impl(array.elem.as_ref(), const_generics);
            match &array.len {
                Expr::Block(syn::ExprBlock { block, .. }) => {
                    if block.stmts.len() != 1 {
                        panic!(
                            "Expected exactly one statement in block, got: {:?}",
                            block.stmts.len()
                        );
                    }
                    if let Stmt::Expr(Expr::Path(expr_path), ..) = &block.stmts[0] {
                        if let Some(len_ident) = expr_path.path.get_ident() {
                            if const_generics.contains(&len_ident) {
                                println!("Const generic new: {:?}", expr_path);
                                dims.push(Dimension::ConstGeneric(expr_path.clone().into()));
                            } else {
                                dims.push(Dimension::Other(expr_path.clone().into()));
                            }
                        }
                    }
                }
                Expr::Path(ExprPath { path, .. }) => {
                    let len_ident = path.get_ident();
                    if len_ident.is_some() && const_generics.contains(&len_ident.unwrap()) {
                        println!("Const generic old: {:?}", array.len);
                        dims.push(Dimension::ConstGeneric(array.len.clone()));
                    } else {
                        dims.push(Dimension::Other(array.len.clone()));
                    }
                }
                Expr::Lit(expr_lit) => dims.push(Dimension::Other(expr_lit.clone().into())),
                _ => panic!("Unsupported array length type: {:?}", array.len),
            }
            dims
        }
        Type::Path(_) => Vec::new(),
        _ => panic!("Unsupported field type (in get_dims_impl)"),
    }
}
