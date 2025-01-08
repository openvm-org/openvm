use proc_macro2::{Span, TokenStream};
use syn::{
    parse::{Parse, ParseStream},
    Ident, Token,
};

enum AsmArg {
    In(TokenStream),
    Out(TokenStream),
    InOut(TokenStream),
    ConstExpr(TokenStream),
    ConstLit(syn::LitStr),
}

struct CustomInsnR {
    pub rd: AsmArg,
    pub rs1: AsmArg,
    pub rs2: AsmArg,
    pub opcode: TokenStream,
    pub funct3: TokenStream,
    pub funct7: TokenStream,
}

impl Parse for CustomInsnR {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut rd = None;
        let mut rs1 = None;
        let mut rs2 = None;
        let mut opcode = None;
        let mut funct3 = None;
        let mut funct7 = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            let value = if key == "opcode" || key == "funct3" || key == "funct7" {
                let mut tokens = TokenStream::new();
                while !input.is_empty() && !input.peek(Token![,]) {
                    tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
                }
                match key.to_string().as_str() {
                    "opcode" => opcode = Some(tokens),
                    "funct3" => funct3 = Some(tokens),
                    "funct7" => funct7 = Some(tokens),
                    _ => unreachable!(),
                }
                None
            } else {
                let lookahead = input.lookahead1();
                Some(if lookahead.peek(kw::In) {
                    input.parse::<kw::In>()?;
                    let mut tokens = TokenStream::new();
                    while !input.is_empty() && !input.peek(Token![,]) {
                        tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
                    }
                    AsmArg::In(tokens)
                } else if lookahead.peek(kw::Out) {
                    input.parse::<kw::Out>()?;
                    let mut tokens = TokenStream::new();
                    while !input.is_empty() && !input.peek(Token![,]) {
                        tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
                    }
                    AsmArg::Out(tokens)
                } else if lookahead.peek(kw::InOut) {
                    input.parse::<kw::InOut>()?;
                    let mut tokens = TokenStream::new();
                    while !input.is_empty() && !input.peek(Token![,]) {
                        tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
                    }
                    AsmArg::InOut(tokens)
                } else if lookahead.peek(kw::Const) {
                    input.parse::<kw::Const>()?;
                    if input.peek(syn::LitStr) {
                        AsmArg::ConstLit(input.parse()?)
                    } else {
                        let mut tokens = TokenStream::new();
                        while !input.is_empty() && !input.peek(Token![,]) {
                            tokens.extend(TokenStream::from(
                                input.parse::<proc_macro2::TokenTree>()?,
                            ));
                        }
                        AsmArg::ConstExpr(tokens)
                    }
                } else {
                    return Err(lookahead.error());
                })
            };

            match key.to_string().as_str() {
                "rd" => rd = value,
                "rs1" => rs1 = value,
                "rs2" => rs2 = value,
                "opcode" | "funct3" | "funct7" => (),
                _ => return Err(syn::Error::new(key.span(), "unexpected field")),
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let opcode = opcode.ok_or_else(|| syn::Error::new(input.span(), "missing opcode field"))?;
        let funct3 = funct3.ok_or_else(|| syn::Error::new(input.span(), "missing funct3 field"))?;
        let funct7 = funct7.ok_or_else(|| syn::Error::new(input.span(), "missing funct7 field"))?;
        let rd = rd.ok_or_else(|| syn::Error::new(input.span(), "missing rd field"))?;
        let rs1 = rs1.ok_or_else(|| syn::Error::new(input.span(), "missing rs1 field"))?;
        let rs2 = rs2.ok_or_else(|| syn::Error::new(input.span(), "missing rs2 field"))?;

        Ok(CustomInsnR {
            rd,
            rs1,
            rs2,
            opcode,
            funct3,
            funct7,
        })
    }
}

mod kw {
    syn::custom_keyword!(In);
    syn::custom_keyword!(Out);
    syn::custom_keyword!(InOut);
    syn::custom_keyword!(Const);
}

#[proc_macro]
pub fn custom_insn_r(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let CustomInsnR {
        rd,
        rs1,
        rs2,
        opcode,
        funct3,
        funct7,
    } = syn::parse_macro_input!(input as CustomInsnR);

    let mut template = String::from(".insn r {opcode}, {funct3}, {funct7}");
    let mut args = vec![];

    // Helper function to handle register arguments
    fn handle_reg_arg(
        template: &mut String,
        args: &mut Vec<proc_macro2::TokenStream>,
        arg: &AsmArg,
        reg_name: &str,
    ) {
        let reg_ident = syn::Ident::new(reg_name, Span::call_site());
        match arg {
            AsmArg::ConstLit(lit) => {
                template.push_str(", ");
                template.push_str(&lit.value());
            }
            AsmArg::In(tokens) => {
                template.push_str(", {");
                template.push_str(reg_name);
                template.push('}');
                args.push(quote::quote! { #reg_ident = in(reg) #tokens });
            }
            AsmArg::Out(tokens) => {
                template.push_str(", {");
                template.push_str(reg_name);
                template.push('}');
                args.push(quote::quote! { #reg_ident = out(reg) #tokens });
            }
            AsmArg::InOut(tokens) => {
                template.push_str(", {");
                template.push_str(reg_name);
                template.push('}');
                args.push(quote::quote! { #reg_ident = inout(reg) #tokens });
            }
            AsmArg::ConstExpr(tokens) => {
                template.push_str(", {");
                template.push_str(reg_name);
                template.push('}');
                args.push(quote::quote! { #reg_ident = const #tokens });
            }
        }
    }

    // Build the template string and args based on which parameters are literals vs expressions
    handle_reg_arg(&mut template, &mut args, &rd, "rd");
    handle_reg_arg(&mut template, &mut args, &rs1, "rs1");
    handle_reg_arg(&mut template, &mut args, &rs2, "rs2");

    let expanded = quote::quote! {
        unsafe {
            core::arch::asm!(
                #template,
                opcode = const #opcode,
                funct3 = const #funct3,
                funct7 = const #funct7,
                #(#args),*
            )
        }
    };

    expanded.into()
}
