use std::collections::VecDeque;

use proc_macro2::{Spacing, Span, TokenTree};

use crate::folder1::file2_tree::ExpressionContainer;
use crate::parser::error::ParserError;
use crate::parser::metadata::ParserMetadata;

const EQ: &str = "==";
const PLUS: &str = "+";
const MINUS: &str = "-";

// `tokens` should be nonempty
pub fn get_operator(tokens: &mut VecDeque<TokenTree>) -> Result<(String, ParserMetadata), ParserError> {
    let mut source_text = String::new();
    loop {
        let next = tokens.pop_front().unwrap();
        match next {
            TokenTree::Punct(punct) => {
                source_text.push(punct.as_char());
                match punct.spacing() {
                    Spacing::Alone => return Ok((source_text.clone(), ParserMetadata { source_text: Some(source_text) })),
                    Spacing::Joint => {}
                }
            }
            _ => {
                return Err(ParserError::ExpectedOperator(ParserMetadata::new(next.span())));
            }
        }
    }
}

pub fn parse_expression_high(tokens: &mut VecDeque<TokenTree>) -> Result<ExpressionContainer, ParserError> {
    let left = parse_expression_1(tokens)?;
    if tokens.is_empty() {
        return Ok(left)
    }
    Span::
    let (operator, metadata) = get_operator(tokens)?;
    if operator != EQ {
        return Err(ParserError::ExpectedEq(metadata));
    }
    
    todo!()
}

pub fn parse_expression_1(tokens: &mut VecDeque<TokenTree>) -> Result<ExpressionContainer, ParserError> {
    todo!()
}