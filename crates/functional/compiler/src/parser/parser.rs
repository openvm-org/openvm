use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

use crate::{
    core::{
        containers::ExpressionContainer,
        ir::{
            AlgebraicTypeDeclaration, AlgebraicTypeVariant, Argument, ArgumentBehavior,
            ArithmeticOperator, Body, Branch, BranchComponent, Expression, Function, FunctionCall,
            Match, Material, Program, Statement, StatementVariant, Type,
        },
    },
    parser::metadata::ParserMetadata,
};

#[derive(Parser)]
#[grammar = "parser/language.pest"]
pub struct LanguageParser;

pub fn parse_program_source(input: &str) -> Result<Program, Box<pest::error::Error<Rule>>> {
    let mut parse_result = LanguageParser::parse(Rule::program, input)?;
    Ok(parse_program(parse_result.next().unwrap()))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum VariableUsageTag {
    Def,
    Let,
    Fix,
    Set,
    Rep,
}

pub fn parse_expression(pair: Pair<Rule>) -> ExpressionContainer {
    parse_expression_helper(pair, None)
}
fn parse_expression_helper(pair: Pair<Rule>, tag: Option<VariableUsageTag>) -> ExpressionContainer {
    let parser_metadata = ParserMetadata::new(&pair);
    let expression = match pair.as_rule() {
        Rule::expr => {
            let mut inner = pair.into_inner();
            let left = parse_expression_helper(inner.next().unwrap(), tag);
            if let Some(_) = inner.next() {
                let right = parse_expression_helper(inner.next().unwrap(), tag);
                Expression::Eq { left, right }
            } else {
                return left;
            }
        }
        Rule::expr1 => {
            let mut inner = pair.into_inner();
            let left = parse_expression_helper(inner.next().unwrap(), tag);
            if let Some(operator) = inner.next() {
                let right = parse_expression_helper(inner.next().unwrap(), tag);
                match operator.as_rule() {
                    Rule::plus => Expression::Arithmetic {
                        operator: ArithmeticOperator::Plus,
                        left,
                        right,
                    },
                    Rule::minus => Expression::Arithmetic {
                        operator: ArithmeticOperator::Minus,
                        left,
                        right,
                    },
                    Rule::concat => Expression::ConstArrayConcatenation { left, right },
                    _ => unreachable!(),
                }
            } else {
                return left;
            }
        }
        Rule::expr2 => {
            let mut inner = pair.into_inner();
            let left = parse_expression_helper(inner.next().unwrap(), tag);
            if let Some(operator) = inner.next() {
                let right = parse_expression_helper(inner.next().unwrap(), tag);
                match operator.as_rule() {
                    Rule::times => Expression::Arithmetic {
                        operator: ArithmeticOperator::Times,
                        left,
                        right,
                    },
                    Rule::div => Expression::Arithmetic {
                        operator: ArithmeticOperator::Div,
                        left,
                        right,
                    },
                    _ => unreachable!(),
                }
            } else {
                return left;
            }
        }
        Rule::expr3 => {
            let mut inner = pair.into_inner();
            if inner.len() == 1 {
                return parse_expression_helper(inner.next().unwrap(), tag);
            } else {
                inner.next();
                let right = parse_expression_helper(inner.next().unwrap(), tag);
                Expression::Arithmetic {
                    operator: ArithmeticOperator::Minus,
                    left: ExpressionContainer::synthetic(Expression::Constant { value: 0 }),
                    right,
                }
            }
        }
        Rule::expr4 => {
            let mut inner = pair.into_inner();
            let left = parse_expression_helper(inner.next().unwrap(), tag);
            if inner.len() == 0 {
                return left;
            }
            let base_parser_metadata = left.parser_metadata;
            let mut left = *left.expression;
            for right in inner {
                let array = ExpressionContainer::new(left, base_parser_metadata.clone());
                left = match right.as_rule() {
                    Rule::const_index => {
                        let index = parse_literal(right.into_inner().next().unwrap()) as usize;
                        Expression::ConstArrayAccess { array, index }
                    }
                    Rule::const_slice => {
                        let mut right_inner = right.into_inner();
                        let from = parse_literal(right_inner.next().unwrap()) as usize;
                        let to = parse_literal(right_inner.next().unwrap()) as usize;
                        Expression::ConstArraySlice { array, from, to }
                    }
                    _ => unreachable!(),
                };
            }
            left
        }
        Rule::expr5 => {
            let mut inner = pair.into_inner();
            if inner.len() == 1 {
                return parse_expression_helper(inner.next().unwrap(), tag);
            } else {
                let operator = inner.next().unwrap();
                let right = parse_expression_helper(inner.next().unwrap(), tag);
                match operator.as_rule() {
                    Rule::readable => Expression::ReadableViewOfPrefix {
                        appendable_prefix: right,
                    },
                    Rule::into_array => Expression::PrefixIntoArray {
                        appendable_prefix: right,
                    },
                    _ => unreachable!(),
                }
            }
        }
        Rule::expr6 => {
            let inside = pair.into_inner().next().unwrap();
            return parse_expression_helper(inside, tag);
        }
        Rule::expr_def => {
            assert_eq!(tag, None);
            let inside = pair.into_inner().next().unwrap();
            return parse_expression_helper(inside, Some(VariableUsageTag::Def));
        }
        Rule::expr_let => {
            assert_eq!(tag, None);
            let inside = pair.into_inner().next().unwrap();
            return parse_expression_helper(inside, Some(VariableUsageTag::Let));
        }
        Rule::expr_fix => {
            assert_eq!(tag, None);
            let inside = pair.into_inner().next().unwrap();
            return parse_expression_helper(inside, Some(VariableUsageTag::Fix));
        }
        Rule::expr_set => {
            assert_eq!(tag, None);
            let inside = pair.into_inner().next().unwrap();
            return parse_expression_helper(inside, Some(VariableUsageTag::Set));
        }
        Rule::expr_rep => {
            assert_eq!(tag, None);
            let inside = pair.into_inner().next().unwrap();
            return parse_expression_helper(inside, Some(VariableUsageTag::Rep));
        }
        Rule::expr_low => {
            let inside = pair.into_inner().next().unwrap();
            match inside.as_rule() {
                Rule::var_identifier => {
                    let name = inside.as_str();
                    let (declares, defines, represents) = match tag {
                        Some(VariableUsageTag::Def) => (true, true, true),
                        Some(VariableUsageTag::Let) => (true, true, false),
                        Some(VariableUsageTag::Fix) => (false, true, true),
                        Some(VariableUsageTag::Set) => (false, true, false),
                        Some(VariableUsageTag::Rep) => (false, false, true),
                        None => (false, false, false),
                    };
                    Expression::Variable {
                        name: name.to_string(),
                        declares,
                        defines,
                        represents,
                    }
                }
                Rule::number_literal => {
                    let value = parse_literal(inside);
                    Expression::Constant { value }
                }
                Rule::algebraic_expr => {
                    let mut inner = inside.into_inner();
                    let constructor = inner.next().unwrap().as_str().to_string();
                    let fields = if let Some(fields) = inner.next() {
                        parse_expr_series_helper(fields, tag)
                    } else {
                        vec![]
                    };
                    Expression::Algebraic {
                        constructor,
                        fields,
                    }
                }
                Rule::const_array => {
                    let elements =
                        parse_expr_series_helper(inside.into_inner().next().unwrap(), tag);
                    Expression::ConstArray { elements }
                }
                Rule::empty_const_array => {
                    let elem_type = parse_wrapped_type(inside.into_inner().next().unwrap());
                    Expression::EmptyConstArray { elem_type }
                }
                Rule::unmaterialized_expr => {
                    return parse_expression_helper(inside.into_inner().next().unwrap(), tag);
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    };
    ExpressionContainer::new(expression, parser_metadata)
}

fn parse_literal(pair: Pair<Rule>) -> isize {
    pair.as_str().parse().unwrap()
}

fn parse_expr_series(pair: Pair<Rule>) -> Vec<ExpressionContainer> {
    parse_expr_series_helper(pair, None)
}
fn parse_expr_series_helper(
    pair: Pair<Rule>,
    tag: Option<VariableUsageTag>,
) -> Vec<ExpressionContainer> {
    let mut exprs = vec![];
    for pair in pair.into_inner() {
        exprs.push(parse_expression_helper(pair, tag));
    }
    exprs
}

fn parse_wrapped_type(pair: Pair<Rule>) -> Type {
    parse_type(pair.into_inner().next().unwrap())
}

fn parse_type(pair: Pair<Rule>) -> Type {
    match pair.as_rule() {
        Rule::field_type => Type::Field,
        Rule::reference_type => {
            Type::Reference(Box::new(parse_type(pair.into_inner().next().unwrap())))
        }
        Rule::readable_prefix_type => {
            let mut inner = pair.into_inner();
            let length = parse_expression(inner.next().unwrap());
            let elem_type = parse_type(inner.next().unwrap());
            Type::ReadablePrefix(Box::new(elem_type), Box::new(length))
        }
        Rule::appendable_prefix_type => {
            let mut inner = pair.into_inner();
            let length = parse_expression(inner.next().unwrap());
            let elem_type = parse_type(inner.next().unwrap());
            Type::AppendablePrefix(Box::new(elem_type), Box::new(length))
        }
        Rule::const_array_type => {
            let mut inner = pair.into_inner();
            let elem_type = parse_type(inner.next().unwrap());
            let len = parse_literal(inner.next().unwrap()) as usize;
            Type::ConstArray(Box::new(elem_type), len)
        }
        Rule::named_type => Type::NamedType(pair.as_str().to_string()),
        Rule::unmaterialized_type => {
            Type::Unmaterialized(Box::new(parse_type(pair.into_inner().next().unwrap())))
        }
        _ => unreachable!(),
    }
}

pub fn parse_body_elem(pair: Pair<Rule>, material: Material, body: &mut Body) {
    let parser_metadata = ParserMetadata::new(&pair);
    match pair.as_rule() {
        Rule::statement_materialized | Rule::statement_unmaterialized => {
            let material = if pair.as_rule() == Rule::statement_materialized {
                Material::Materialized
            } else {
                Material::Dematerialized
            } & material;
            let inside = pair.into_inner().next().unwrap();
            body.statements.push(Statement {
                variant: parse_statement_variant(inside),
                material,
                parser_metadata,
            });
        }
        Rule::function_call_materialized | Rule::function_call_unmaterialized => {
            let material = if pair.as_rule() == Rule::function_call_materialized {
                Material::Materialized
            } else {
                Material::Dematerialized
            } & material;
            let inside = pair.into_inner().next().unwrap();
            body.function_calls
                .push(parse_function_call(inside, material));
        }
        Rule::match_materialized | Rule::match_unmaterialized => {
            let material = if pair.as_rule() == Rule::match_materialized {
                Material::Materialized
            } else {
                Material::Dematerialized
            } & material;
            let inside = pair.into_inner().next().unwrap();
            body.matches.push(parse_match(inside, material));
        }
        _ => unreachable!(),
    }
}

fn parse_statement_variant(pair: Pair<Rule>) -> StatementVariant {
    match pair.as_rule() {
        Rule::statement_inside => {
            let inside = pair.into_inner().next().unwrap();
            match inside.as_rule() {
                Rule::alloc_declaration => {
                    let mut inner = inside.into_inner();
                    let tipo = parse_wrapped_type(inner.next().unwrap());
                    let name = inner.next().unwrap().as_str().to_string();
                    StatementVariant::VariableDeclaration {
                        name,
                        tipo,
                        represents: true,
                    }
                }
                Rule::unalloc_declaration => {
                    let mut inner = inside.into_inner();
                    let tipo = parse_wrapped_type(inner.next().unwrap());
                    let name = inner.next().unwrap().as_str().to_string();
                    StatementVariant::VariableDeclaration {
                        name,
                        tipo,
                        represents: false,
                    }
                }
                Rule::equality => {
                    let mut inner = inside.into_inner();
                    let left = parse_expression(inner.next().unwrap());
                    let right = parse_expression(inner.next().unwrap());
                    StatementVariant::Equality { left, right }
                }
                Rule::reference => {
                    let mut inner = inside.into_inner();
                    let reference = parse_expression(inner.next().unwrap());
                    let data = parse_expression(inner.next().unwrap());
                    StatementVariant::Reference { reference, data }
                }
                Rule::dereference => {
                    let mut inner = inside.into_inner();
                    let data = parse_expression(inner.next().unwrap());
                    let reference = parse_expression(inner.next().unwrap());
                    StatementVariant::Reference { reference, data }
                }
                Rule::empty_prefix => {
                    let mut inner = inside.into_inner();
                    let prefix = parse_expression(inner.next().unwrap());
                    let elem_type = parse_wrapped_type(inner.next().unwrap());
                    StatementVariant::EmptyPrefix { prefix, elem_type }
                }
                Rule::prefix_append => {
                    let mut inner = inside.into_inner();
                    let new_prefix = parse_expression(inner.next().unwrap());
                    let old_prefix = parse_expression(inner.next().unwrap());
                    let elem = parse_expression(inner.next().unwrap());
                    StatementVariant::PrefixAppend {
                        new_prefix,
                        old_prefix,
                        elem,
                    }
                }
                Rule::prefix_access => {
                    let mut inner = inside.into_inner();
                    let elem = parse_expression(inner.next().unwrap());
                    let prefix = parse_expression(inner.next().unwrap());
                    let index = parse_expression(inner.next().unwrap());
                    StatementVariant::ArrayAccess {
                        elem,
                        array: prefix,
                        index,
                    }
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
}

fn parse_function_call(pair: Pair<Rule>, material: Material) -> FunctionCall {
    let parser_metadata = ParserMetadata::new(&pair);
    match pair.as_rule() {
        Rule::function_call_inside => {
            let mut inner = pair.into_inner();
            let function_name = inner.next().unwrap().as_str().to_string();
            let arguments = parse_expr_series(inner.next().unwrap());
            FunctionCall {
                function: function_name,
                arguments,
                material,
                parser_metadata,
            }
        }
        _ => unreachable!(),
    }
}

fn parse_match(pair: Pair<Rule>, material: Material) -> Match {
    let parser_metadata = ParserMetadata::new(&pair);
    match pair.as_rule() {
        Rule::match_inside => {
            let mut inner = pair.into_inner();
            let match_argument = inner.next().unwrap();
            let (value, check_material) = match match_argument.as_rule() {
                Rule::match_argument_materialized => {
                    println!("match_argument_materialized");
                    (
                        parse_expression(match_argument.into_inner().next().unwrap()),
                        Material::Materialized,
                    )
                }
                Rule::match_argument_unmaterialized => {
                    println!("match_argument_unmaterialized");
                    (
                        parse_expression(match_argument.into_inner().next().unwrap()),
                        Material::Dematerialized,
                    )
                }
                _ => unreachable!(),
            };
            let mut branches = vec![];
            for branch in inner {
                let branch_parser_metadata = ParserMetadata::new(&branch);
                let mut branch_inner = branch.into_inner();
                let constructor = branch_inner.next().unwrap().as_str().to_string();
                let mut components = vec![];
                while branch_inner.len() > 1 {
                    let component = branch_inner.next().unwrap();
                    match component.as_rule() {
                        Rule::component => {
                            let mut component_inner = component.into_inner();
                            let represents = if component_inner.len() == 1 {
                                true
                            } else {
                                component_inner.next().unwrap().as_rule() == Rule::alloc
                            };
                            let name = component_inner.next().unwrap().as_str().to_string();
                            components.push(BranchComponent { name, represents });
                        }
                        _ => unreachable!(),
                    }
                }
                let body = parse_body(branch_inner.next().unwrap(), material);
                branches.push(Branch {
                    constructor,
                    components,
                    body,
                    parser_metadata: branch_parser_metadata,
                });
            }
            Match {
                value,
                check_material: check_material & material,
                branches,
                parser_metadata,
            }
        }
        _ => unreachable!(),
    }
}

fn parse_body(pair: Pair<Rule>, material: Material) -> Body {
    match pair.as_rule() {
        Rule::body => {
            let inside = pair.into_inner().next().unwrap();
            let material = if inside.as_rule() == Rule::body_materialized {
                Material::Materialized
            } else {
                Material::Dematerialized
            } & material;

            let mut body = Body {
                statements: vec![],
                function_calls: vec![],
                matches: vec![],
            };
            for elem in inside.into_inner() {
                parse_body_elem(elem, material, &mut body);
            }
            body
        }
        _ => unreachable!(),
    }
}

fn parse_function(pair: Pair<Rule>) -> Function {
    let parser_metadata = ParserMetadata::new(&pair);
    let mut inner = pair.into_inner();
    let inline = inner.next().unwrap().as_rule() == Rule::inline_function_keyword;
    let function_name = inner.next().unwrap().as_str().to_string();
    let mut arguments = vec![];
    while inner.len() > 1 {
        let argument = inner.next().unwrap();
        let argument_parser_metadata = ParserMetadata::new(&argument);
        let mut argument_inner = argument.into_inner();
        let behavior = if argument_inner.next().unwrap().as_rule() == Rule::arg_in {
            ArgumentBehavior::In
        } else {
            ArgumentBehavior::Out
        };
        let represents = if argument_inner.len() == 2 {
            match behavior {
                ArgumentBehavior::In => true,
                ArgumentBehavior::Out => false,
            }
        } else {
            argument_inner.next().unwrap().as_rule() == Rule::alloc
        };
        let tipo = parse_wrapped_type(argument_inner.next().unwrap());
        let name = argument_inner.next().unwrap().as_str().to_string();
        arguments.push(Argument {
            behavior,
            tipo,
            name,
            represents,
            parser_metadata: argument_parser_metadata,
        });
    }
    let body = parse_body(inner.next().unwrap(), Material::Materialized);
    Function {
        name: function_name,
        arguments,
        body,
        inline,
        parser_metadata,
    }
}

fn parse_type_definition(pair: Pair<Rule>) -> AlgebraicTypeDeclaration {
    let pair = pair.into_inner().next().unwrap();
    let parser_metadata = ParserMetadata::new(&pair);
    match pair.as_rule() {
        Rule::type_definition_single => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let mut components = vec![];
            for component in inner {
                components.push(parse_type(component));
            }
            AlgebraicTypeDeclaration {
                name: name.clone(),
                variants: vec![AlgebraicTypeVariant { name, components }],
                parser_metadata,
            }
        }
        Rule::type_definition_enum => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let mut variants = vec![];
            for variant in inner {
                let mut inner = variant.into_inner();
                let name = inner.next().unwrap().as_str().to_string();
                let mut components = vec![];
                for component in inner {
                    components.push(parse_type(component));
                }
                variants.push(AlgebraicTypeVariant { name, components });
            }
            AlgebraicTypeDeclaration {
                name: name.clone(),
                variants,
                parser_metadata,
            }
        }
        _ => unreachable!(),
    }
}

fn parse_program(pair: Pair<Rule>) -> Program {
    let mut program = Program {
        algebraic_types: vec![],
        functions: vec![],
    };
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::type_definition => program.algebraic_types.push(parse_type_definition(pair)),
            Rule::function_definition => program.functions.push(parse_function(pair)),
            Rule::EOI => {}
            _ => unreachable!(),
        }
    }
    program
}
