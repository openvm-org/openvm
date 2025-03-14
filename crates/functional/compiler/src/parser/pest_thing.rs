use pest::iterators::Pair;
use pest_derive::Parser;
use pest::Parser;
use crate::folder1::file2_tree::ExpressionContainer;
use crate::folder1::ir::{ArithmeticOperator, Expression, Program, Type};

/*#[derive(Parser)]
#[grammar = "parser/language.pest"]
pub struct LanguageParser;*/
//#[grammar = "parser/language.pest"]
//pub struct LanguageParser;
#[allow(non_upper_case_globals)]
const _PEST_GRAMMAR_LanguageParser: [&'static str; 1usize] = [
    "number_literal = @{ \"-\"? ~ \'0\'..\'9\'+ }\nvar_identifier = @{ (ASCII_ALPHA_LOWER | \"_\") ~ (ASCII_ALPHANUMERIC | \"_\")* }\ntype_identifier = @{ ASCII_ALPHA_UPPER ~ (ASCII_ALPHANUMERIC | \"_\")* }\n\nfield_type = { \"F\" }\nreference_type = { \"&\" ~ tipo }\narray_type = { \"&\" ~ \"[\" ~ tipo ~ \"]\" }\nunder_construction_array_type = { \"#\" ~ \"[\" ~ tipo ~ \"]\" }\nconst_array_type = { \"[\" ~ tipo ~ \";\" ~ number_literal ~ \"]\" }\nnamed_type = { type_identifier }\nunmaterialized_type = { \"{\" ~ tipo ~ \"}\" }\ntipo = { field_type | reference_type | array_type | under_construction_array_type | const_array_type | named_type }\n\nwrapped_type = { \"<\" ~ tipo ~ \">\" }\n\nlet_ident = { \"let\" ~ var_identifier }\ndef_ident = { \"def\" ~ var_identifier }\nunmaterialized_expr = { \"{\" ~ expr ~ \"}\" }\nalgebraic_expr = { type_identifier ~ (\"(\" ~ expr_series ~ \")\")? }\nconst_array = { \"[\" ~ expr_series ~ \"]\" }\nempty_const_array = { \"[\" ~ wrapped_type ~ \"]\" }\n\nconst_index = { \"[\" ~ number_literal ~ \"]\" }\nconst_slice = { \"[\" ~ number_literal ~ \"..\" ~ number_literal ~ \"]\" }\n\neq = { \"==\" }\nplus = { \"+\" }\nminus = { \"-\" }\ntimes = { \"*\" }\ndiv = { \"/\" }\nconcat = { \"++\" }\n\nexpr = { expr1 ~ (eq ~ expr)? }\nexpr1 = { expr2 ~ ((plus | minus | concat) ~ expr1)? }\nexpr2 = { expr_low ~ ((times | div) ~ expr3)? }\nexpr3 = { expr_low ~ (const_index | const_slice)* }\nexpr_low = { let_ident | def_ident | unmaterialized_expr | algebraic_expr | const_array | empty_const_array }\n\nvar_identifier_series = { (var_identifier ~ \",\")* ~ var_identifier? }\nexpr_series = { (expr ~ \",\")* ~ expr? }\n\ndeclaration = { \"declare\" ~ wrapped_type ~ var_identifier }\nequality = { expr ~ \"=\" ~ expr }\nreference = { expr ~ \"->\" ~ expr }\ndereference = { expr ~ \"<-\" ~ expr }\nempty_under_construction_array = { expr ~ \"->\" ~ wrapped_type ~ \"|\" }\nunder_construction_array_prepend = { expr ~ \"->\" ~ expr ~ \"|\" ~ expr }\nfinalize_array = { expr ~ \"->\" ~ \"|\" ~ expr }\narray_index = { expr ~ \"<-\" ~ expr ~ \"!!\" ~ expr }\n\nstatement_inside = { declaration | equality | reference | dereference | empty_under_construction_array | under_construction_array_prepend | finalize_array | array_index }\nstatement_materialized = { statement_inside ~ \";\" }\nstatement_unmaterialized = { \"{\" ~ statement_inside ~ \"}\" ~ \";\" }\n\nfunction_call_inside = { var_identifier ~ \"(\" ~ expr_series ~ \")\" }\nfunction_call_materialized = { function_call_inside ~ \";\" }\nfunction_call_unmaterialized = { \"{\" ~ function_call_inside ~ \"}\" ~ \";\" }\n\nmatch_arm = { type_identifier ~ (\"(\" ~ var_identifier_series ~ \")\")? ~ \"=>\" ~ body }\nmatch_argument_materialized = { \"match\" ~ expr }\nmatch_argument_unmaterialized = { \"{\" ~ \"match\" ~ expr ~ \"}\" }\nmatch_inside = { (match_argument_materialized | match_argument_unmaterialized) ~ \"(\" ~ match_arm* ~ \")\" }\nmatch_materialized = { match_inside }\nmatch_unmaterialized = { \"{\" ~ match_inside ~ \"}\" }\n\nbody_elem = { statement_materialized | statement_unmaterialized | function_call_materialized | function_call_unmaterialized | match_materialized | match_unmaterialized }\nbody_inside = { (body_elem | (\"{\" ~ body_elem ~ \"}\"))* }\nbody_materialized = { (\"(\" ~ body_inside ~ \")\") }\nbody_unmaterialized = { \"{\" ~ body_inside ~ \"}\" }\nbody = { body_materialized | body_unmaterialized }\n\narg_in = { \"in\" }\narg_out = { \"out\" }\nargument = { (arg_in | arg_out) ~ wrapped_type ~ expr }\nfunction_definition = { (\"fn\" ~ var_identifier ~ \"(\" ~ (argument ~ \",\")* ~ argument? ~ \")\" ~ body) }\n\ntype_definition_single = { \"struct\" ~ type_identifier ~ \"(\" ~ (tipo ~ \",\") ~ tipo? ~ \")\" }\ntype_definition_variant = { type_identifier ~ \"(\" ~ (tipo ~ \",\") ~ tipo? ~ \")\" }\ntype_definition_enum = { \"enum\" ~ type_identifier ~ \"(\" ~ (type_definition_variant ~ \",\") ~ type_definition_variant? ~ \")\" }\ntype_definition = { type_definition_single | type_definition_enum }\n\nprogram = { (type_definition | function_definition)* }",
];
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
pub enum Rule {
    number_literal,
    var_identifier,
    type_identifier,
    field_type,
    reference_type,
    array_type,
    under_construction_array_type,
    const_array_type,
    named_type,
    unmaterialized_type,
    tipo,
    wrapped_type,
    let_ident,
    def_ident,
    unmaterialized_expr,
    algebraic_expr,
    const_array,
    empty_const_array,
    const_index,
    const_slice,
    eq,
    plus,
    minus,
    times,
    div,
    concat,
    expr,
    expr1,
    expr2,
    expr3,
    expr_low,
    var_identifier_series,
    expr_series,
    declaration,
    equality,
    reference,
    dereference,
    empty_under_construction_array,
    under_construction_array_prepend,
    finalize_array,
    array_index,
    statement_inside,
    statement_materialized,
    statement_unmaterialized,
    function_call_inside,
    function_call_materialized,
    function_call_unmaterialized,
    match_arm,
    match_argument_materialized,
    match_argument_unmaterialized,
    match_inside,
    match_materialized,
    match_unmaterialized,
    body_elem,
    body_inside,
    body_materialized,
    body_unmaterialized,
    body,
    arg_in,
    arg_out,
    argument,
    function_definition,
    type_definition_single,
    type_definition_variant,
    type_definition_enum,
    type_definition,
    program,
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::clone::Clone for Rule {
    #[inline]
    fn clone(&self) -> Rule {
        *self
    }
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::marker::Copy for Rule {}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::fmt::Debug for Rule {
    #[inline]
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::write_str(
            f,
            match self {
                Rule::number_literal => "number_literal",
                Rule::var_identifier => "var_identifier",
                Rule::type_identifier => "type_identifier",
                Rule::field_type => "field_type",
                Rule::reference_type => "reference_type",
                Rule::array_type => "array_type",
                Rule::under_construction_array_type => {
                    "under_construction_array_type"
                }
                Rule::const_array_type => "const_array_type",
                Rule::named_type => "named_type",
                Rule::unmaterialized_type => "unmaterialized_type",
                Rule::tipo => "tipo",
                Rule::wrapped_type => "wrapped_type",
                Rule::let_ident => "let_ident",
                Rule::def_ident => "def_ident",
                Rule::unmaterialized_expr => "unmaterialized_expr",
                Rule::algebraic_expr => "algebraic_expr",
                Rule::const_array => "const_array",
                Rule::empty_const_array => "empty_const_array",
                Rule::const_index => "const_index",
                Rule::const_slice => "const_slice",
                Rule::eq => "eq",
                Rule::plus => "plus",
                Rule::minus => "minus",
                Rule::times => "times",
                Rule::div => "div",
                Rule::concat => "concat",
                Rule::expr => "expr",
                Rule::expr1 => "expr1",
                Rule::expr2 => "expr2",
                Rule::expr3 => "expr3",
                Rule::expr_low => "expr_low",
                Rule::var_identifier_series => "var_identifier_series",
                Rule::expr_series => "expr_series",
                Rule::declaration => "declaration",
                Rule::equality => "equality",
                Rule::reference => "reference",
                Rule::dereference => "dereference",
                Rule::empty_under_construction_array => {
                    "empty_under_construction_array"
                }
                Rule::under_construction_array_prepend => {
                    "under_construction_array_prepend"
                }
                Rule::finalize_array => "finalize_array",
                Rule::array_index => "array_index",
                Rule::statement_inside => "statement_inside",
                Rule::statement_materialized => "statement_materialized",
                Rule::statement_unmaterialized => "statement_unmaterialized",
                Rule::function_call_inside => "function_call_inside",
                Rule::function_call_materialized => "function_call_materialized",
                Rule::function_call_unmaterialized => "function_call_unmaterialized",
                Rule::match_arm => "match_arm",
                Rule::match_argument_materialized => "match_argument_materialized",
                Rule::match_argument_unmaterialized => {
                    "match_argument_unmaterialized"
                }
                Rule::match_inside => "match_inside",
                Rule::match_materialized => "match_materialized",
                Rule::match_unmaterialized => "match_unmaterialized",
                Rule::body_elem => "body_elem",
                Rule::body_inside => "body_inside",
                Rule::body_materialized => "body_materialized",
                Rule::body_unmaterialized => "body_unmaterialized",
                Rule::body => "body",
                Rule::arg_in => "arg_in",
                Rule::arg_out => "arg_out",
                Rule::argument => "argument",
                Rule::function_definition => "function_definition",
                Rule::type_definition_single => "type_definition_single",
                Rule::type_definition_variant => "type_definition_variant",
                Rule::type_definition_enum => "type_definition_enum",
                Rule::type_definition => "type_definition",
                Rule::program => "program",
            },
        )
    }
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::cmp::Eq for Rule {
    #[inline]
    #[doc(hidden)]
    #[coverage(off)]
    fn assert_receiver_is_total_eq(&self) -> () {}
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::hash::Hash for Rule {
    #[inline]
    fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
        let __self_discr = ::core::intrinsics::discriminant_value(self);
        ::core::hash::Hash::hash(&__self_discr, state)
    }
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::cmp::Ord for Rule {
    #[inline]
    fn cmp(&self, other: &Rule) -> ::core::cmp::Ordering {
        let __self_discr = ::core::intrinsics::discriminant_value(self);
        let __arg1_discr = ::core::intrinsics::discriminant_value(other);
        ::core::cmp::Ord::cmp(&__self_discr, &__arg1_discr)
    }
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::marker::StructuralPartialEq for Rule {}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::cmp::PartialEq for Rule {
    #[inline]
    fn eq(&self, other: &Rule) -> bool {
        let __self_discr = ::core::intrinsics::discriminant_value(self);
        let __arg1_discr = ::core::intrinsics::discriminant_value(other);
        __self_discr == __arg1_discr
    }
}
#[automatically_derived]
#[allow(dead_code, non_camel_case_types, clippy::upper_case_acronyms)]
impl ::core::cmp::PartialOrd for Rule {
    #[inline]
    fn partial_cmp(
        &self,
        other: &Rule,
    ) -> ::core::option::Option<::core::cmp::Ordering> {
        let __self_discr = ::core::intrinsics::discriminant_value(self);
        let __arg1_discr = ::core::intrinsics::discriminant_value(other);
        ::core::cmp::PartialOrd::partial_cmp(&__self_discr, &__arg1_discr)
    }
}
impl Rule {
    pub fn all_rules() -> &'static [Rule] {
        &[
            Rule::number_literal,
            Rule::var_identifier,
            Rule::type_identifier,
            Rule::field_type,
            Rule::reference_type,
            Rule::array_type,
            Rule::under_construction_array_type,
            Rule::const_array_type,
            Rule::named_type,
            Rule::unmaterialized_type,
            Rule::tipo,
            Rule::wrapped_type,
            Rule::let_ident,
            Rule::def_ident,
            Rule::unmaterialized_expr,
            Rule::algebraic_expr,
            Rule::const_array,
            Rule::empty_const_array,
            Rule::const_index,
            Rule::const_slice,
            Rule::eq,
            Rule::plus,
            Rule::minus,
            Rule::times,
            Rule::div,
            Rule::concat,
            Rule::expr,
            Rule::expr1,
            Rule::expr2,
            Rule::expr3,
            Rule::expr_low,
            Rule::var_identifier_series,
            Rule::expr_series,
            Rule::declaration,
            Rule::equality,
            Rule::reference,
            Rule::dereference,
            Rule::empty_under_construction_array,
            Rule::under_construction_array_prepend,
            Rule::finalize_array,
            Rule::array_index,
            Rule::statement_inside,
            Rule::statement_materialized,
            Rule::statement_unmaterialized,
            Rule::function_call_inside,
            Rule::function_call_materialized,
            Rule::function_call_unmaterialized,
            Rule::match_arm,
            Rule::match_argument_materialized,
            Rule::match_argument_unmaterialized,
            Rule::match_inside,
            Rule::match_materialized,
            Rule::match_unmaterialized,
            Rule::body_elem,
            Rule::body_inside,
            Rule::body_materialized,
            Rule::body_unmaterialized,
            Rule::body,
            Rule::arg_in,
            Rule::arg_out,
            Rule::argument,
            Rule::function_definition,
            Rule::type_definition_single,
            Rule::type_definition_variant,
            Rule::type_definition_enum,
            Rule::type_definition,
            Rule::program,
        ]
    }
}
#[allow(clippy::all)]
impl ::pest::Parser<Rule> for LanguageParser {
    fn parse<'i>(
        rule: Rule,
        input: &'i str,
    ) -> ::std::result::Result<
        ::pest::iterators::Pairs<'i, Rule>,
        ::pest::error::Error<Rule>,
    > {
        mod rules {
            #![allow(clippy::upper_case_acronyms)]
            pub mod hidden {
                use super::super::Rule;
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn skip(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    Ok(state)
                }
            }
            pub mod visible {
                use super::super::Rule;
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn number_literal(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::number_literal,
                            |state| {
                                state
                                    .atomic(
                                        ::pest::Atomicity::Atomic,
                                        |state| {
                                            state
                                                .sequence(|state| {
                                                    state
                                                        .optional(|state| { state.match_string("-") })
                                                        .and_then(|state| { state.match_range('0'..'9') })
                                                        .and_then(|state| {
                                                            state.repeat(|state| { state.match_range('0'..'9') })
                                                        })
                                                })
                                        },
                                    )
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn var_identifier(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::var_identifier,
                            |state| {
                                state
                                    .atomic(
                                        ::pest::Atomicity::Atomic,
                                        |state| {
                                            state
                                                .sequence(|state| {
                                                    self::ASCII_ALPHA_LOWER(state)
                                                        .or_else(|state| { state.match_string("_") })
                                                        .and_then(|state| {
                                                            state
                                                                .repeat(|state| {
                                                                    self::ASCII_ALPHANUMERIC(state)
                                                                        .or_else(|state| { state.match_string("_") })
                                                                })
                                                        })
                                                })
                                        },
                                    )
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn type_identifier(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::type_identifier,
                            |state| {
                                state
                                    .atomic(
                                        ::pest::Atomicity::Atomic,
                                        |state| {
                                            state
                                                .sequence(|state| {
                                                    self::ASCII_ALPHA_UPPER(state)
                                                        .and_then(|state| {
                                                            state
                                                                .repeat(|state| {
                                                                    self::ASCII_ALPHANUMERIC(state)
                                                                        .or_else(|state| { state.match_string("_") })
                                                                })
                                                        })
                                                })
                                        },
                                    )
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn field_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::field_type, |state| { state.match_string("F") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn reference_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::reference_type,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("&")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn array_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::array_type,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("&")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("[") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn under_construction_array_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::under_construction_array_type,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("#")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("[") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn const_array_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::const_array_type,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("[")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(";") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::number_literal(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn named_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::named_type,
                            |state| { self::type_identifier(state) },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn unmaterialized_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::unmaterialized_type,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn tipo(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::tipo,
                            |state| {
                                self::field_type(state)
                                    .or_else(|state| { self::reference_type(state) })
                                    .or_else(|state| { self::array_type(state) })
                                    .or_else(|state| {
                                        self::under_construction_array_type(state)
                                    })
                                    .or_else(|state| { self::const_array_type(state) })
                                    .or_else(|state| { self::named_type(state) })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn wrapped_type(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::wrapped_type,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("<")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(">") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn let_ident(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::let_ident,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("let")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::var_identifier(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn def_ident(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::def_ident,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("def")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::var_identifier(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn unmaterialized_expr(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::unmaterialized_expr,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn algebraic_expr(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::algebraic_expr,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::type_identifier(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                state
                                                                    .match_string("(")
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { self::expr_series(state) })
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { state.match_string(")") })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn const_array(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::const_array,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("[")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr_series(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn empty_const_array(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::empty_const_array,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("[")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::wrapped_type(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn const_index(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::const_index,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("[")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::number_literal(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn const_slice(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::const_slice,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("[")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::number_literal(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("..") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::number_literal(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("]") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn eq(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::eq, |state| { state.match_string("==") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn plus(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::plus, |state| { state.match_string("+") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn minus(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::minus, |state| { state.match_string("-") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn times(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::times, |state| { state.match_string("*") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn div(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::div, |state| { state.match_string("/") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn concat(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::concat, |state| { state.match_string("++") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn expr(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::expr,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr1(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                self::eq(state)
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { self::expr(state) })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn expr1(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::expr1,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr2(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                self::plus(state)
                                                                    .or_else(|state| { self::minus(state) })
                                                                    .or_else(|state| { self::concat(state) })
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { self::expr1(state) })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn expr2(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::expr2,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr_low(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                self::times(state)
                                                                    .or_else(|state| { self::div(state) })
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { self::expr3(state) })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn expr3(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::expr3,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr_low(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .sequence(|state| {
                                                        state
                                                            .optional(|state| {
                                                                self::const_index(state)
                                                                    .or_else(|state| { self::const_slice(state) })
                                                                    .and_then(|state| {
                                                                        state
                                                                            .repeat(|state| {
                                                                                state
                                                                                    .sequence(|state| {
                                                                                        super::hidden::skip(state)
                                                                                            .and_then(|state| {
                                                                                                self::const_index(state)
                                                                                                    .or_else(|state| { self::const_slice(state) })
                                                                                            })
                                                                                    })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn expr_low(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::expr_low,
                            |state| {
                                self::let_ident(state)
                                    .or_else(|state| { self::def_ident(state) })
                                    .or_else(|state| { self::unmaterialized_expr(state) })
                                    .or_else(|state| { self::algebraic_expr(state) })
                                    .or_else(|state| { self::const_array(state) })
                                    .or_else(|state| { self::empty_const_array(state) })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn var_identifier_series(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::var_identifier_series,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .sequence(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                self::var_identifier(state)
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { state.match_string(",") })
                                                            })
                                                            .and_then(|state| {
                                                                state
                                                                    .repeat(|state| {
                                                                        state
                                                                            .sequence(|state| {
                                                                                super::hidden::skip(state)
                                                                                    .and_then(|state| {
                                                                                        state
                                                                                            .sequence(|state| {
                                                                                                self::var_identifier(state)
                                                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                                                    .and_then(|state| { state.match_string(",") })
                                                                                            })
                                                                                    })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state.optional(|state| { self::var_identifier(state) })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn expr_series(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::expr_series,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .sequence(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                self::expr(state)
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { state.match_string(",") })
                                                            })
                                                            .and_then(|state| {
                                                                state
                                                                    .repeat(|state| {
                                                                        state
                                                                            .sequence(|state| {
                                                                                super::hidden::skip(state)
                                                                                    .and_then(|state| {
                                                                                        state
                                                                                            .sequence(|state| {
                                                                                                self::expr(state)
                                                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                                                    .and_then(|state| { state.match_string(",") })
                                                                                            })
                                                                                    })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state.optional(|state| { self::expr(state) })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn declaration(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::declaration,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("declare")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::wrapped_type(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::var_identifier(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn equality(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::equality,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("=") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn reference(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::reference,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("->") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn dereference(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::dereference,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("<-") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn empty_under_construction_array(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::empty_under_construction_array,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("->") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::wrapped_type(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("|") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn under_construction_array_prepend(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::under_construction_array_prepend,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("->") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("|") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn finalize_array(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::finalize_array,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("->") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("|") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn array_index(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::array_index,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::expr(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("<-") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("!!") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn statement_inside(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::statement_inside,
                            |state| {
                                self::declaration(state)
                                    .or_else(|state| { self::equality(state) })
                                    .or_else(|state| { self::reference(state) })
                                    .or_else(|state| { self::dereference(state) })
                                    .or_else(|state| {
                                        self::empty_under_construction_array(state)
                                    })
                                    .or_else(|state| {
                                        self::under_construction_array_prepend(state)
                                    })
                                    .or_else(|state| { self::finalize_array(state) })
                                    .or_else(|state| { self::array_index(state) })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn statement_materialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::statement_materialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::statement_inside(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(";") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn statement_unmaterialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::statement_unmaterialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::statement_inside(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(";") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn function_call_inside(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::function_call_inside,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::var_identifier(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("(") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr_series(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn function_call_materialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::function_call_materialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::function_call_inside(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(";") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn function_call_unmaterialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::function_call_unmaterialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::function_call_inside(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(";") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn match_arm(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::match_arm,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::type_identifier(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .optional(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                state
                                                                    .match_string("(")
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { self::var_identifier_series(state) })
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { state.match_string(")") })
                                                            })
                                                    })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("=>") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::body(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn match_argument_materialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::match_argument_materialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("match")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn match_argument_unmaterialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::match_argument_unmaterialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("match") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn match_inside(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::match_inside,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::match_argument_materialized(state)
                                            .or_else(|state| {
                                                self::match_argument_unmaterialized(state)
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("(") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .sequence(|state| {
                                                        state
                                                            .optional(|state| {
                                                                self::match_arm(state)
                                                                    .and_then(|state| {
                                                                        state
                                                                            .repeat(|state| {
                                                                                state
                                                                                    .sequence(|state| {
                                                                                        super::hidden::skip(state)
                                                                                            .and_then(|state| { self::match_arm(state) })
                                                                                    })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn match_materialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::match_materialized,
                            |state| { self::match_inside(state) },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn match_unmaterialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::match_unmaterialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::match_inside(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn body_elem(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::body_elem,
                            |state| {
                                self::statement_materialized(state)
                                    .or_else(|state| { self::statement_unmaterialized(state) })
                                    .or_else(|state| {
                                        self::function_call_materialized(state)
                                    })
                                    .or_else(|state| {
                                        self::function_call_unmaterialized(state)
                                    })
                                    .or_else(|state| { self::match_materialized(state) })
                                    .or_else(|state| { self::match_unmaterialized(state) })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn body_inside(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::body_inside,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .optional(|state| {
                                                self::body_elem(state)
                                                    .or_else(|state| {
                                                        state
                                                            .sequence(|state| {
                                                                state
                                                                    .match_string("{")
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { self::body_elem(state) })
                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                    .and_then(|state| { state.match_string("}") })
                                                            })
                                                    })
                                                    .and_then(|state| {
                                                        state
                                                            .repeat(|state| {
                                                                state
                                                                    .sequence(|state| {
                                                                        super::hidden::skip(state)
                                                                            .and_then(|state| {
                                                                                self::body_elem(state)
                                                                                    .or_else(|state| {
                                                                                        state
                                                                                            .sequence(|state| {
                                                                                                state
                                                                                                    .match_string("{")
                                                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                                                    .and_then(|state| { self::body_elem(state) })
                                                                                                    .and_then(|state| { super::hidden::skip(state) })
                                                                                                    .and_then(|state| { state.match_string("}") })
                                                                                            })
                                                                                    })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn body_materialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::body_materialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("(")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::body_inside(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn body_unmaterialized(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::body_unmaterialized,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("{")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::body_inside(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("}") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn body(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::body,
                            |state| {
                                self::body_materialized(state)
                                    .or_else(|state| { self::body_unmaterialized(state) })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn arg_in(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::arg_in, |state| { state.match_string("in") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn arg_out(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.rule(Rule::arg_out, |state| { state.match_string("out") })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn argument(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::argument,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::arg_in(state)
                                            .or_else(|state| { self::arg_out(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::wrapped_type(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::expr(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn function_definition(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::function_definition,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("fn")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::var_identifier(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("(") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .sequence(|state| {
                                                        state
                                                            .optional(|state| {
                                                                state
                                                                    .sequence(|state| {
                                                                        self::argument(state)
                                                                            .and_then(|state| { super::hidden::skip(state) })
                                                                            .and_then(|state| { state.match_string(",") })
                                                                    })
                                                                    .and_then(|state| {
                                                                        state
                                                                            .repeat(|state| {
                                                                                state
                                                                                    .sequence(|state| {
                                                                                        super::hidden::skip(state)
                                                                                            .and_then(|state| {
                                                                                                state
                                                                                                    .sequence(|state| {
                                                                                                        self::argument(state)
                                                                                                            .and_then(|state| { super::hidden::skip(state) })
                                                                                                            .and_then(|state| { state.match_string(",") })
                                                                                                    })
                                                                                            })
                                                                                    })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state.optional(|state| { self::argument(state) })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::body(state) })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn type_definition_single(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::type_definition_single,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("struct")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::type_identifier(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("(") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(",") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state.optional(|state| { self::tipo(state) })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn type_definition_variant(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::type_definition_variant,
                            |state| {
                                state
                                    .sequence(|state| {
                                        self::type_identifier(state)
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("(") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::tipo(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(",") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state.optional(|state| { self::tipo(state) })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn type_definition_enum(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::type_definition_enum,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .match_string("enum")
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::type_identifier(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string("(") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { self::type_definition_variant(state) })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(",") })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| {
                                                state
                                                    .optional(|state| { self::type_definition_variant(state) })
                                            })
                                            .and_then(|state| { super::hidden::skip(state) })
                                            .and_then(|state| { state.match_string(")") })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn type_definition(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::type_definition,
                            |state| {
                                self::type_definition_single(state)
                                    .or_else(|state| { self::type_definition_enum(state) })
                            },
                        )
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn program(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .rule(
                            Rule::program,
                            |state| {
                                state
                                    .sequence(|state| {
                                        state
                                            .optional(|state| {
                                                self::type_definition(state)
                                                    .or_else(|state| { self::function_definition(state) })
                                                    .and_then(|state| {
                                                        state
                                                            .repeat(|state| {
                                                                state
                                                                    .sequence(|state| {
                                                                        super::hidden::skip(state)
                                                                            .and_then(|state| {
                                                                                self::type_definition(state)
                                                                                    .or_else(|state| { self::function_definition(state) })
                                                                            })
                                                                    })
                                                            })
                                                    })
                                            })
                                    })
                            },
                        )
                }
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn ASCII_ALPHA_LOWER(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.match_range('a'..'z')
                }
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn ASCII_ALPHA_UPPER(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state.match_range('A'..'Z')
                }
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn ASCII_ALPHANUMERIC(
                    state: ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                ) -> ::pest::ParseResult<
                    ::std::boxed::Box<::pest::ParserState<'_, Rule>>,
                > {
                    state
                        .match_range('a'..'z')
                        .or_else(|state| state.match_range('A'..'Z'))
                        .or_else(|state| state.match_range('0'..'9'))
                }
            }
            pub use self::visible::*;
        }
        ::pest::state(
            input,
            |state| {
                match rule {
                    Rule::number_literal => rules::number_literal(state),
                    Rule::var_identifier => rules::var_identifier(state),
                    Rule::type_identifier => rules::type_identifier(state),
                    Rule::field_type => rules::field_type(state),
                    Rule::reference_type => rules::reference_type(state),
                    Rule::array_type => rules::array_type(state),
                    Rule::under_construction_array_type => {
                        rules::under_construction_array_type(state)
                    }
                    Rule::const_array_type => rules::const_array_type(state),
                    Rule::named_type => rules::named_type(state),
                    Rule::unmaterialized_type => rules::unmaterialized_type(state),
                    Rule::tipo => rules::tipo(state),
                    Rule::wrapped_type => rules::wrapped_type(state),
                    Rule::let_ident => rules::let_ident(state),
                    Rule::def_ident => rules::def_ident(state),
                    Rule::unmaterialized_expr => rules::unmaterialized_expr(state),
                    Rule::algebraic_expr => rules::algebraic_expr(state),
                    Rule::const_array => rules::const_array(state),
                    Rule::empty_const_array => rules::empty_const_array(state),
                    Rule::const_index => rules::const_index(state),
                    Rule::const_slice => rules::const_slice(state),
                    Rule::eq => rules::eq(state),
                    Rule::plus => rules::plus(state),
                    Rule::minus => rules::minus(state),
                    Rule::times => rules::times(state),
                    Rule::div => rules::div(state),
                    Rule::concat => rules::concat(state),
                    Rule::expr => rules::expr(state),
                    Rule::expr1 => rules::expr1(state),
                    Rule::expr2 => rules::expr2(state),
                    Rule::expr3 => rules::expr3(state),
                    Rule::expr_low => rules::expr_low(state),
                    Rule::var_identifier_series => {
                        rules::var_identifier_series(state)
                    }
                    Rule::expr_series => rules::expr_series(state),
                    Rule::declaration => rules::declaration(state),
                    Rule::equality => rules::equality(state),
                    Rule::reference => rules::reference(state),
                    Rule::dereference => rules::dereference(state),
                    Rule::empty_under_construction_array => {
                        rules::empty_under_construction_array(state)
                    }
                    Rule::under_construction_array_prepend => {
                        rules::under_construction_array_prepend(state)
                    }
                    Rule::finalize_array => rules::finalize_array(state),
                    Rule::array_index => rules::array_index(state),
                    Rule::statement_inside => rules::statement_inside(state),
                    Rule::statement_materialized => {
                        rules::statement_materialized(state)
                    }
                    Rule::statement_unmaterialized => {
                        rules::statement_unmaterialized(state)
                    }
                    Rule::function_call_inside => rules::function_call_inside(state),
                    Rule::function_call_materialized => {
                        rules::function_call_materialized(state)
                    }
                    Rule::function_call_unmaterialized => {
                        rules::function_call_unmaterialized(state)
                    }
                    Rule::match_arm => rules::match_arm(state),
                    Rule::match_argument_materialized => {
                        rules::match_argument_materialized(state)
                    }
                    Rule::match_argument_unmaterialized => {
                        rules::match_argument_unmaterialized(state)
                    }
                    Rule::match_inside => rules::match_inside(state),
                    Rule::match_materialized => rules::match_materialized(state),
                    Rule::match_unmaterialized => rules::match_unmaterialized(state),
                    Rule::body_elem => rules::body_elem(state),
                    Rule::body_inside => rules::body_inside(state),
                    Rule::body_materialized => rules::body_materialized(state),
                    Rule::body_unmaterialized => rules::body_unmaterialized(state),
                    Rule::body => rules::body(state),
                    Rule::arg_in => rules::arg_in(state),
                    Rule::arg_out => rules::arg_out(state),
                    Rule::argument => rules::argument(state),
                    Rule::function_definition => rules::function_definition(state),
                    Rule::type_definition_single => {
                        rules::type_definition_single(state)
                    }
                    Rule::type_definition_variant => {
                        rules::type_definition_variant(state)
                    }
                    Rule::type_definition_enum => rules::type_definition_enum(state),
                    Rule::type_definition => rules::type_definition(state),
                    Rule::program => rules::program(state),
                }
            },
        )
    }
}



/*pub fn parse_program(input: &str) -> Result<Program, pest::error::Error<Rule>> {
    let parse_result = LanguageParser::parse(Rule::program, input)?;
    parse_result.into

    todo!()
}*/

pub fn parse_expression(pair: Pair<Rule>) -> ExpressionContainer {
    match pair.as_rule() {
        Rule::expr => {
            let mut inner = pair.into_inner();
            let left = parse_expression(inner.next().unwrap());
            if let Some(_) = inner.next() {
                let right = parse_expression(inner.next().unwrap());
                ExpressionContainer::new(Expression::Eq { left, right })
            } else {
                left
            }
        }
        Rule::expr1 => {
            let mut inner = pair.into_inner();
            let left = parse_expression(inner.next().unwrap());
            if let Some(operator) = inner.next() {
                let right = parse_expression(inner.next().unwrap());
                match operator.as_rule() {
                    Rule::plus => ExpressionContainer::new(Expression::Arithmetic {
                        operator: ArithmeticOperator::Plus,
                        left,
                        right,
                    }),
                    Rule::minus => ExpressionContainer::new(Expression::Arithmetic {
                        operator: ArithmeticOperator::Minus,
                        left,
                        right,
                    }),
                    Rule::concat => ExpressionContainer::new(Expression::ConstArrayConcatenation {
                        left,
                        right,
                    }),
                    _ => unreachable!()
                }
            } else {
                left
            }
        }
        Rule::expr2 => {
            let mut inner = pair.into_inner();
            let left = parse_expression(inner.next().unwrap());
            if let Some(operator) = inner.next() {
                let right = parse_expression(inner.next().unwrap());
                match operator.as_rule() {
                    Rule::times => ExpressionContainer::new(Expression::Arithmetic {
                        operator: ArithmeticOperator::Times,
                        left,
                        right,
                    }),
                    Rule::div => ExpressionContainer::new(Expression::Arithmetic {
                        operator: ArithmeticOperator::Div,
                        left,
                        right,
                    }),
                    _ => unreachable!()
                }
            } else {
                left
            }
        }
        Rule::expr_low => {
            let inside = pair.into_inner().next().unwrap();
            match inside.as_rule() {
                Rule::var_identifier => {
                    let name = inside.as_str();
                    ExpressionContainer::new(Expression::Variable {
                        name: name.to_string(),
                    })
                }
                Rule::number_literal => {
                    let value = parse_literal(inside);
                    ExpressionContainer::new(Expression::Constant {
                        value,
                    })
                }
                Rule::let_ident => {
                    let mut inner = inside.into_inner();
                    let name = inner.next().unwrap().as_str();
                    ExpressionContainer::new(Expression::Let {
                        name: name.to_string(),
                    })
                }
                Rule::def_ident => {
                    let mut inner = inside.into_inner();
                    let name = inner.next().unwrap().as_str();
                    ExpressionContainer::new(Expression::Define {
                        name: name.to_string(),
                    })
                }
                Rule::algebraic_expr => {
                    let mut inner = inside.into_inner();
                    let constructor = inner.next().unwrap().as_str().to_string();
                    let fields = if let Some(fields) = inner.next() {
                        parse_expr_series(fields)
                    } else {
                        vec![]
                    };
                    ExpressionContainer::new(Expression::Algebraic {
                        constructor,
                        fields,
                    })
                }
                Rule::const_array => {
                    let elements = parse_expr_series(inside.into_inner().next().unwrap());
                    ExpressionContainer::new(Expression::ConstArray { elements })
                }
                Rule::empty_const_array => {
                    let elem_type = parse_wrapped_type(inside.into_inner().next().unwrap());
                    ExpressionContainer::new(Expression::EmptyConstArray { elem_type })
                }
                Rule::unmaterialized_expr => {
                    parse_expression(inside.into_inner().next().unwrap())
                }
                _ => unreachable!()
            }
        }
        _ => unreachable!()
    }
}

fn parse_literal(pair: Pair<Rule>) -> isize {
    pair.as_str().parse().unwrap()
}

fn parse_expr_series(pair: Pair<Rule>) -> Vec<ExpressionContainer> {
    let mut exprs = vec![];
    for pair in pair.into_inner() {
        exprs.push(parse_expression(pair));
    }
    exprs
}

fn parse_wrapped_type(pair: Pair<Rule>) -> Type {
    parse_type(pair.into_inner().next().unwrap())
}

fn parse_type(pair: Pair<Rule>) -> Type {
    todo!()
}
pub fn parse_statement(pair: Pair<Rule>) -> Statement {
    match pair.as_rule() {
        Rule::statement_materialized | Rule::statement_unmaterialized => {
            let inside = pair.into_inner().next().unwrap();
            parse_statement_inside(inside)
        }
        _ => unreachable!()
    }
}

fn parse_statement_inside(pair: Pair<Rule>) -> Statement {
    match pair.as_rule() {
        Rule::statement_inside => {
            let inside = pair.into_inner().next().unwrap();
            match inside.as_rule() {
                Rule::declaration => {
                    let mut inner = inside.into_inner();
                    let type_ = parse_wrapped_type(inner.next().unwrap());
                    let name = inner.next().unwrap().as_str().to_string();
                    Statement::Declaration { name, type_ }
                }
                Rule::equality => {
                    let mut inner = inside.into_inner();
                    let left = parse_expression(inner.next().unwrap());
                    let right = parse_expression(inner.next().unwrap());
                    Statement::Equality { left, right }
                }
                Rule::reference => {
                    let mut inner = inside.into_inner();
                    let left = parse_expression(inner.next().unwrap());
                    let right = parse_expression(inner.next().unwrap());
                    Statement::Reference { from: left, to: right }
                }
                Rule::dereference => {
                    let mut inner = inside.into_inner();
                    let left = parse_expression(inner.next().unwrap());
                    let right = parse_expression(inner.next().unwrap());
                    Statement::Dereference { from: left, to: right }
                }
                Rule::empty_under_construction_array => {
                    let mut inner = inside.into_inner();
                    let array = parse_expression(inner.next().unwrap());
                    let elem_type = parse_wrapped_type(inner.next().unwrap());
                    Statement::EmptyUnderConstructionArray { array, elem_type }
                }
                Rule::under_construction_array_prepend => {
                    let mut inner = inside.into_inner();
                    let array = parse_expression(inner.next().unwrap());
                    let element = parse_expression(inner.next().unwrap());
                    let new_array = parse_expression(inner.next().unwrap());
                    Statement::UnderConstructionArrayPrepend { array, element, new_array }
                }
                Rule::finalize_array => {
                    let mut inner = inside.into_inner();
                    let array = parse_expression(inner.next().unwrap());
                    let final_array = parse_expression(inner.next().unwrap());
                    Statement::FinalizeArray { array, final_array }
                }
                Rule::array_index => {
                    let mut inner = inside.into_inner();
                    let array = parse_expression(inner.next().unwrap());
                    let index = parse_expression(inner.next().unwrap());
                    let element = parse_expression(inner.next().unwrap());
                    Statement::ArrayIndex { array, index, element }
                }
                _ => unreachable!()
            }
        }
        _ => unreachable!()
    }
}