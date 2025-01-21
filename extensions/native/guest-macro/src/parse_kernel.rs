use proc_macro2::{TokenStream, TokenTree};

#[derive(Debug)]
pub struct ParsedKernelArgument {
    pub name: String,
    pub rust_type: String,
    pub edsl_type: String,
}

#[derive(Debug)]
pub struct ParsedKernel {
    pub function_name: String,
    pub arguments: Vec<ParsedKernelArgument>,
    pub file_path: String,
    pub rust_return_type: String,
    pub edsl_return_type: String,
}

pub fn parse_raw_kernel(source: TokenStream) -> ParsedKernel {
    let token_trees = source.into_iter().collect::<Vec<_>>();

    let function_name = match token_trees[1].clone() {
        TokenTree::Ident(ident) => ident.to_string(),
        _ => panic!("First token must be the function name"),
    };

    let arguments = match token_trees[2].clone() {
        TokenTree::Group(group) => {
            assert_eq!(group.delimiter(), proc_macro2::Delimiter::Parenthesis);

            let as_string = group.stream().to_string();
            let argument_strings = as_string
                .split(',')
                .map(|argument| argument.trim())
                .collect::<Vec<_>>();

            argument_strings
                .into_iter()
                .map(|argument_string| {
                    let colon_index = argument_string.find(':').unwrap();
                    let bar_index = argument_string.find('|').unwrap();
                    let name = argument_string[..colon_index].trim().to_string();
                    let rust_type = argument_string[colon_index + 1..bar_index]
                        .trim()
                        .to_string();
                    let edsl_type = argument_string[bar_index + 1..].trim().to_string();
                    ParsedKernelArgument {
                        name,
                        rust_type,
                        edsl_type,
                    }
                })
                .collect::<Vec<_>>()
        }
        _ => panic!("Second token must be the list of arguments"),
    };

    let return_type_token_trees = token_trees[5..token_trees.len() - 1].to_vec();
    let return_type_stream = TokenStream::from_iter(return_type_token_trees);
    let return_type_string = return_type_stream.to_string();
    let bar_index = return_type_string.find('|').unwrap();
    let rust_return_type = return_type_string[..bar_index].trim().to_string();
    let edsl_return_type = return_type_string[bar_index + 1..].trim().to_string();

    let file_path = match token_trees[token_trees.len() - 1].clone() {
        TokenTree::Group(group) => group.stream().to_string(),
        _ => panic!("Last token must be the function body"),
    };

    ParsedKernel {
        function_name,
        arguments,
        file_path,
        rust_return_type,
        edsl_return_type,
    }
}
