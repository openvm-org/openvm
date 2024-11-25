use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Expr, Ident, Token,
};

pub struct MacroArgs {
    pub items: Vec<Item>,
}

pub struct Item {
    pub name: Ident,
    pub params: Punctuated<Param, Token![,]>,
}

pub struct Param {
    pub name: Ident,
    pub eq_token: Token![=],
    pub value: Expr,
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(MacroArgs {
            items: input
                .parse_terminated(Item::parse, Token![,])?
                .into_iter()
                .collect(),
        })
    }
}

impl Parse for Item {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        let content;
        syn::braced!(content in input);
        let params = content.parse_terminated(Param::parse, Token![,])?;
        Ok(Item { name, params })
    }
}

impl Parse for Param {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        let eq_token: Token![=] = input.parse()?;
        let value: Expr = input.parse()?;
        Ok(Param {
            name,
            eq_token,
            value,
        })
    }
}

pub fn string_to_bytes(s: &str) -> Vec<u8> {
    if s.starts_with("0x") {
        return s
            .chars()
            .skip(2)
            .filter(|c| !c.is_whitespace())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .chunks(2)
            .map(|ch| u8::from_str_radix(&ch.iter().rev().collect::<String>(), 16).unwrap())
            .collect();
    }
    let mut digits = s
        .chars()
        .map(|c| c.to_digit(10).expect("Invalid numeric literal"))
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    while !digits.is_empty() {
        let mut rem = 0u32;
        let mut new_digits = Vec::new();
        for &d in digits.iter() {
            rem = rem * 10 + d;
            new_digits.push(rem / 256);
            rem %= 256;
        }
        digits = new_digits.into_iter().skip_while(|&d| d == 0).collect();
        bytes.push(rem as u8);
    }
    bytes
}
