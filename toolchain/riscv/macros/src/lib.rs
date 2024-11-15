use syn::{
    parse::{Parse, ParseStream},
    Stmt,
};

pub struct Stmts {
    pub stmts: Vec<Stmt>,
}

impl Parse for Stmts {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut stmts = Vec::new();
        while !input.is_empty() {
            stmts.push(input.parse()?);
        }
        Ok(Stmts { stmts })
    }
}
