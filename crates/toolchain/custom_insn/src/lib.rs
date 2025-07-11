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

struct CustomInsnI {
    pub rd: AsmArg,
    pub rs1: AsmArg,
    pub imm: AsmArg,
    pub opcode: TokenStream,
    pub funct3: TokenStream,
}

/// Returns `(rd, rs1, opcode, funct3)`.
#[allow(clippy::type_complexity)]
fn parse_common_fields(
    input: ParseStream,
) -> syn::Result<(
    Option<AsmArg>,
    Option<AsmArg>,
    Option<TokenStream>,
    Option<TokenStream>,
)> {
    let mut rd = None;
    let mut rs1 = None;
    let mut opcode = None;
    let mut funct3 = None;

    while !input.is_empty() {
        let key: Ident = input.parse()?;
        input.parse::<Token![=]>()?;

        let value = if key == "opcode" || key == "funct3" {
            let mut tokens = TokenStream::new();
            while !input.is_empty() && !input.peek(Token![,]) {
                tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
            }
            match key.to_string().as_str() {
                "opcode" => opcode = Some(tokens),
                "funct3" => funct3 = Some(tokens),
                _ => unreachable!(),
            }
            None
        } else if key == "rd" || key == "rs1" {
            Some(parse_asm_arg(input)?)
        } else {
            while !input.is_empty() && !input.peek(Token![,]) {
                input.parse::<proc_macro2::TokenTree>()?;
            }
            None
        };

        match key.to_string().as_str() {
            "rd" => rd = value,
            "rs1" => rs1 = value,
            "opcode" | "funct3" => (),
            // Skip other fields instead of returning an error
            _ => {
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
        }

        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
    }

    Ok((rd, rs1, opcode, funct3))
}

// Helper function to parse AsmArg
fn parse_asm_arg(input: ParseStream) -> syn::Result<AsmArg> {
    let lookahead = input.lookahead1();
    if lookahead.peek(kw::In) {
        input.parse::<kw::In>()?;
        let mut tokens = TokenStream::new();
        while !input.is_empty() && !input.peek(Token![,]) {
            tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
        }
        Ok(AsmArg::In(tokens))
    } else if lookahead.peek(kw::Out) {
        // ... similar for Out
        input.parse::<kw::Out>()?;
        let mut tokens = TokenStream::new();
        while !input.is_empty() && !input.peek(Token![,]) {
            tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
        }
        Ok(AsmArg::Out(tokens))
    } else if lookahead.peek(kw::InOut) {
        // ... similar for InOut
        input.parse::<kw::InOut>()?;
        let mut tokens = TokenStream::new();
        while !input.is_empty() && !input.peek(Token![,]) {
            tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
        }
        Ok(AsmArg::InOut(tokens))
    } else if lookahead.peek(kw::Const) {
        input.parse::<kw::Const>()?;
        if input.peek(syn::LitStr) {
            Ok(AsmArg::ConstLit(input.parse()?))
        } else {
            let mut tokens = TokenStream::new();
            while !input.is_empty() && !input.peek(Token![,]) {
                tokens.extend(TokenStream::from(input.parse::<proc_macro2::TokenTree>()?));
            }
            Ok(AsmArg::ConstExpr(tokens))
        }
    } else {
        Err(lookahead.error())
    }
}

impl Parse for CustomInsnR {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let input_fork = input.fork();
        let (rd, rs1, opcode, funct3) = parse_common_fields(input)?;

        // Parse rs2 and funct7 from the forked input
        let mut rs2 = None;
        let mut funct7 = None;
        while !input_fork.is_empty() {
            let key: Ident = input_fork.parse()?;
            input_fork.parse::<Token![=]>()?;

            if key == "rs2" {
                rs2 = Some(parse_asm_arg(&input_fork)?);
            } else if key == "funct7" {
                let mut tokens = TokenStream::new();
                while !input_fork.is_empty() && !input_fork.peek(Token![,]) {
                    tokens.extend(TokenStream::from(
                        input_fork.parse::<proc_macro2::TokenTree>()?,
                    ));
                }
                funct7 = Some(tokens);
            } else {
                // Skip other fields
                while !input_fork.is_empty() && !input_fork.peek(Token![,]) {
                    input_fork.parse::<proc_macro2::TokenTree>()?;
                }
            }

            if !input_fork.is_empty() {
                input_fork.parse::<Token![,]>()?;
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

impl Parse for CustomInsnI {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let input_fork = input.fork();
        let (rd, rs1, opcode, funct3) = parse_common_fields(input)?;

        // Parse imm from the forked input
        let mut imm = None;
        while !input_fork.is_empty() {
            let key: Ident = input_fork.parse()?;
            input_fork.parse::<Token![=]>()?;

            if key == "imm" {
                let value = parse_asm_arg(&input_fork)?;
                match value {
                    AsmArg::ConstLit(lit) => imm = Some(AsmArg::ConstLit(lit)),
                    AsmArg::ConstExpr(expr) => imm = Some(AsmArg::ConstExpr(expr)),
                    _ => return Err(syn::Error::new(key.span(), "imm must be a Const")),
                }
            } else {
                // Skip other fields
                while !input_fork.is_empty() && !input_fork.peek(Token![,]) {
                    input_fork.parse::<proc_macro2::TokenTree>()?;
                }
            }

            if !input_fork.is_empty() {
                input_fork.parse::<Token![,]>()?;
            }
        }

        let opcode = opcode.ok_or_else(|| syn::Error::new(input.span(), "missing opcode field"))?;
        let funct3 = funct3.ok_or_else(|| syn::Error::new(input.span(), "missing funct3 field"))?;
        let rd = rd.ok_or_else(|| syn::Error::new(input.span(), "missing rd field"))?;
        let rs1 = rs1.ok_or_else(|| syn::Error::new(input.span(), "missing rs1 field"))?;
        let imm = imm.ok_or_else(|| syn::Error::new(input.span(), "missing imm field"))?;

        Ok(CustomInsnI {
            rd,
            rs1,
            imm,
            opcode,
            funct3,
        })
    }
}

// Helper function for handling register arguments in both proc macros
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

mod kw {
    syn::custom_keyword!(In);
    syn::custom_keyword!(Out);
    syn::custom_keyword!(InOut);
    syn::custom_keyword!(Const);
}

/// Custom RISC-V instruction macro for the zkVM.
///
/// This macro is used to define custom R-type RISC-V instructions for the zkVM.
/// Usage:
/// ```rust
/// custom_insn_r!(
///     opcode = OPCODE,
///     funct3 = FUNCT3,
///     funct7 = FUNCT7,
///     rd = InOut x0,
///     rs1 = In rs1,
///     rs2 = In rs2
/// );
/// ```
/// Here, `opcode`, `funct3`, and `funct7` are the opcode, funct3, and funct7 fields of the RISC-V
/// instruction. `rd`, `rs1`, and `rs2` are the destination register, source register 1, and source
/// register 2 respectively. The `In`, `Out`, `InOut`, and `Const` keywords are required to specify
/// the type of the register arguments. They translate to `in(reg)`, `out(reg)`, `inout(reg)`, and
/// `const` respectively, and mean
/// - "read the value from this variable" before execution (`In`),
/// - "write the value to this variable" after execution (`Out`),
/// - "read the value from this variable, then write it back to the same variable" after execution
///   (`InOut`), and
/// - "use this constant value" (`Const`).
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
    handle_reg_arg(&mut template, &mut args, &rd, "rd");
    handle_reg_arg(&mut template, &mut args, &rs1, "rs1");
    handle_reg_arg(&mut template, &mut args, &rs2, "rs2");

    let expanded = quote::quote! {
        #[cfg(target_os = "zkvm")]
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

/// Custom RISC-V instruction macro for the zkVM.
///
/// This macro is used to define custom I-type RISC-V instructions for the zkVM.
/// Usage:
/// ```rust
/// custom_insn_i!(
///     opcode = OPCODE,
///     funct3 = FUNCT3,
///     rd = InOut x0,
///     rs1 = In rs1,
///     imm = Const 123
/// );
/// ```
/// Here, `opcode`, `funct3` are the opcode and funct3 fields of the RISC-V instruction.
/// `rd`, `rs1`, and `imm` are the destination register, source register 1, and immediate value
/// respectively. The `In`, `Out`, `InOut`, and `Const` keywords are required to specify the type of
/// the register arguments. They translate to `in(reg)`, `out(reg)`, `inout(reg)`, and `const`
/// respectively, and mean
/// - "read the value from this variable" before execution (`In`),
/// - "write the value to this variable" after execution (`Out`),
/// - "read the value from this variable, then write it back to the same variable" after execution
///   (`InOut`), and
/// - "use this constant value" (`Const`).
///
/// The `imm` argument is required to be a constant value.
#[proc_macro]
pub fn custom_insn_i(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let CustomInsnI {
        rd,
        rs1,
        imm,
        opcode,
        funct3,
    } = syn::parse_macro_input!(input as CustomInsnI);

    let mut template = String::from(".insn i {opcode}, {funct3}");
    let mut args = vec![];

    // Helper function to handle register arguments
    handle_reg_arg(&mut template, &mut args, &rd, "rd");
    handle_reg_arg(&mut template, &mut args, &rs1, "rs1");
    handle_reg_arg(&mut template, &mut args, &imm, "imm");

    let expanded = quote::quote! {
        #[cfg(target_os = "zkvm")]
        unsafe {
            core::arch::asm!(
                #template,
                opcode = const #opcode,
                funct3 = const #funct3,
                #(#args),*
            )
        }
    };

    expanded.into()
}
