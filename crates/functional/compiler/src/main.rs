use crate::folder1::{
    file2_tree::ExpressionContainer,
    ir::{
        AlgebraicTypeDeclaration, AlgebraicTypeVariant, Argument, ArgumentBehavior,
        ArithmeticOperator, Body, Branch, Expression, Function, FunctionCall, Match, Material,
        Program, Statement, Type,
    },
    stage1::stage1,
};

pub mod air;
pub mod execution;
pub mod folder1;
// mod transpiled_fibonacci;
// pub mod parser;

fn main() {
    println!("Hello, world!");

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
        }],
        functions: vec![Function {
            name: "fibonacci".to_string(),
            arguments: vec![
                Argument {
                    behavior: ArgumentBehavior::In,
                    tipo: Type::Field,
                    name: "n".to_string(),
                    represents: true,
                },
                Argument {
                    behavior: ArgumentBehavior::Out,
                    tipo: Type::Field,
                    name: "a".to_string(),
                    represents: false,
                },
                Argument {
                    behavior: ArgumentBehavior::Out,
                    tipo: Type::Field,
                    name: "b".to_string(),
                    represents: false,
                },
            ],
            body: Body {
                statements: vec![],
                function_calls: vec![],
                matches: vec![Match {
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
                            constructor: "True".to_string(),
                            components: vec![],
                            body: Body {
                                statements: vec![
                                    (
                                        Material::Materialized,
                                        Statement::Equality {
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
                                    ),
                                    (
                                        Material::Materialized,
                                        Statement::Equality {
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
                                    ),
                                    (
                                        Material::Materialized,
                                        Statement::Equality {
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
                                    ),
                                ],
                                matches: vec![],
                                function_calls: vec![],
                            },
                        },
                        Branch {
                            constructor: "False".to_string(),
                            components: vec![],
                            body: Body {
                                function_calls: vec![(
                                    Material::Materialized,
                                    FunctionCall {
                                        function: "fibonacci".to_string(),
                                        arguments: vec![
                                            ExpressionContainer::new(Expression::Variable {
                                                name: "n".to_string(),
                                                declares: false,
                                                defines: false,
                                                represents: false,
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
                                    },
                                )],
                                matches: vec![],
                                statements: vec![
                                    (
                                        Material::Materialized,
                                        Statement::Equality {
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
                                    ),
                                    (
                                        Material::Materialized,
                                        Statement::Equality {
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
                                    ),
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
