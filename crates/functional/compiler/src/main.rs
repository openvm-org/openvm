use crate::{
    folder1::{
        file2_tree::ExpressionContainer,
        ir::{
            AlgebraicTypeDeclaration, AlgebraicTypeVariant, Argument, ArgumentBehavior,
            ArithmeticOperator, Body, Branch, Expression, Function, FunctionCall, Match, Material,
            Program, Statement, StatementVariant, Type,
        },
        stage1::stage1,
    },
    parser::metadata::ParserMetadata,
    transpiled_fibonacci::{isize_to_field_elem, TLFunction_fibonacci, Tracker},
};

pub mod air;
pub mod execution;
pub mod folder1;
pub mod parser;
pub mod transpiled_fibonacci;

fn main() {
    println!("Hello, world!");
    // compile_and_transpile_fibonacci();
    test_fibonacci();

    let x = true;
    let y = false;
    let z = x | y;

    //let mut x = Box::new(Some(vec![]));
    //x.as_mut().as_mut().unwrap().push(1);
}

fn test_fibonacci() {
    let mut tracker = Tracker::default();
    let mut fibonacci = TLFunction_fibonacci::default();
    fibonacci.n = isize_to_field_elem(12);
    println!("calculating {}th fibonacci number", fibonacci.n);
    fibonacci.stage_0(&mut tracker);
    assert_eq!(fibonacci.a, isize_to_field_elem(144));
    println!("success: {}", fibonacci.a)
}

fn compile_and_transpile_fibonacci() {
    let program = Program {
        algebraic_types: vec![AlgebraicTypeDeclaration {
            name: "Bool".to_string(),
            variants: vec![
                AlgebraicTypeVariant {
                    name: "True".to_string(),
                    components: vec![],
                },
                AlgebraicTypeVariant {
                    name: "False".to_string(),
                    components: vec![],
                },
            ],
            parser_metadata: Default::default(),
        }],
        functions: vec![Function {
            parser_metadata: Default::default(),
            name: "fibonacci".to_string(),
            arguments: vec![
                Argument {
                    behavior: ArgumentBehavior::In,
                    tipo: Type::Field,
                    name: "n".to_string(),
                    represents: true,
                    parser_metadata: Default::default(),
                },
                Argument {
                    behavior: ArgumentBehavior::Out,
                    tipo: Type::Field,
                    name: "a".to_string(),
                    represents: false,
                    parser_metadata: Default::default(),
                },
                Argument {
                    behavior: ArgumentBehavior::Out,
                    tipo: Type::Field,
                    name: "b".to_string(),
                    represents: false,
                    parser_metadata: Default::default(),
                },
            ],
            body: Body {
                statements: vec![],
                function_calls: vec![],
                matches: vec![Match {
                    parser_metadata: Default::default(),
                    value: ExpressionContainer::new(Expression::Eq {
                        left: ExpressionContainer::new(Expression::Variable {
                            name: "n".to_string(),
                            declares: false,
                            defines: false,
                            represents: false,
                        }),
                        right: ExpressionContainer::new(Expression::Constant { value: 0 }),
                    }),
                    check_material: Material::Dematerialized,
                    branches: vec![
                        Branch {
                            parser_metadata: Default::default(),
                            constructor: "True".to_string(),
                            components: vec![],
                            body: Body {
                                statements: vec![
                                    Statement {
                                        parser_metadata: Default::default(),
                                        material: Material::Materialized,
                                        variant: StatementVariant::Equality {
                                            left: ExpressionContainer::new(Expression::Variable {
                                                name: "n".to_string(),
                                                declares: false,
                                                defines: false,
                                                represents: false,
                                            }),
                                            right: ExpressionContainer::new(Expression::Constant {
                                                value: 0,
                                            }),
                                        },
                                    },
                                    Statement {
                                        material: Material::Materialized,
                                        parser_metadata: Default::default(),
                                        variant: StatementVariant::Equality {
                                            left: ExpressionContainer::new(Expression::Variable {
                                                name: "a".to_string(),
                                                declares: false,
                                                defines: true,
                                                represents: true,
                                            }),
                                            right: ExpressionContainer::new(Expression::Constant {
                                                value: 0,
                                            }),
                                        },
                                    },
                                    Statement {
                                        material: Material::Materialized,
                                        parser_metadata: Default::default(),
                                        variant: StatementVariant::Equality {
                                            left: ExpressionContainer::new(Expression::Variable {
                                                name: "b".to_string(),
                                                declares: false,
                                                defines: true,
                                                represents: true,
                                            }),
                                            right: ExpressionContainer::new(Expression::Constant {
                                                value: 1,
                                            }),
                                        },
                                    },
                                ],
                                matches: vec![],
                                function_calls: vec![],
                            },
                        },
                        Branch {
                            parser_metadata: Default::default(),
                            constructor: "False".to_string(),
                            components: vec![],
                            body: Body {
                                function_calls: vec![FunctionCall {
                                    material: Material::Materialized,
                                    parser_metadata: Default::default(),
                                    function: "fibonacci".to_string(),
                                    arguments: vec![
                                        ExpressionContainer::new(Expression::Arithmetic {
                                            operator: ArithmeticOperator::Minus,
                                            left: ExpressionContainer::new(Expression::Variable {
                                                name: "n".to_string(),
                                                declares: false,
                                                defines: false,
                                                represents: false,
                                            }),
                                            right: ExpressionContainer::new(Expression::Constant {
                                                value: 1,
                                            }),
                                        }),
                                        ExpressionContainer::new(Expression::Variable {
                                            name: "x".to_string(),
                                            declares: true,
                                            defines: true,
                                            represents: true,
                                        }),
                                        ExpressionContainer::new(Expression::Variable {
                                            name: "y".to_string(),
                                            declares: true,
                                            defines: true,
                                            represents: true,
                                        }),
                                    ],
                                }],
                                matches: vec![],
                                statements: vec![
                                    Statement {
                                        material: Material::Materialized,
                                        parser_metadata: Default::default(),
                                        variant: StatementVariant::Equality {
                                            left: ExpressionContainer::new(Expression::Variable {
                                                name: "a".to_string(),
                                                declares: false,
                                                defines: true,
                                                represents: true,
                                            }),
                                            right: ExpressionContainer::new(Expression::Variable {
                                                name: "y".to_string(),
                                                declares: false,
                                                defines: false,
                                                represents: false,
                                            }),
                                        },
                                    },
                                    Statement {
                                        material: Material::Materialized,
                                        parser_metadata: Default::default(),
                                        variant: StatementVariant::Equality {
                                            left: ExpressionContainer::new(Expression::Variable {
                                                name: "b".to_string(),
                                                declares: false,
                                                defines: true,
                                                represents: true,
                                            }),
                                            right: ExpressionContainer::new(
                                                Expression::Arithmetic {
                                                    operator: ArithmeticOperator::Plus,
                                                    left: ExpressionContainer::new(
                                                        Expression::Variable {
                                                            name: "x".to_string(),
                                                            declares: false,
                                                            defines: false,
                                                            represents: false,
                                                        },
                                                    ),
                                                    right: ExpressionContainer::new(
                                                        Expression::Variable {
                                                            name: "y".to_string(),
                                                            declares: false,
                                                            defines: false,
                                                            represents: false,
                                                        },
                                                    ),
                                                },
                                            ),
                                        },
                                    },
                                ],
                            },
                        },
                    ],
                }],
            },
            inline: false,
        }],
    };

    let stage2_program = stage1(program).unwrap();
    //println!("{:?}", stage2_program);
    let transpiled = stage2_program.transpile();
    println!("{}", transpiled);
}
