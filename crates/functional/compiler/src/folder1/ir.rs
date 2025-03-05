use std::{ops::BitAndAssign, sync::Arc};

use super::file2_tree::ExpressionContainer;
use crate::folder1::error::CompilationError;

#[derive(Clone, Debug)]
pub enum Type {
    Field,
    NamedType(String),
    Reference(Arc<Type>),
    Array(Arc<Type>),
    UnderConstructionArray(Arc<Type>),
    Unmaterialized(Arc<Type>),
    ConstArray(Arc<Type>, usize),
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Type::Field, Type::Field) => true,
            (Type::NamedType(left), Type::NamedType(right)) => left == right,
            (Type::Reference(left), Type::Reference(right)) => left.eq(right),
            (Type::Array(left), Type::Array(right)) => left.eq(right),
            (Type::UnderConstructionArray(left), Type::UnderConstructionArray(right)) => {
                left.eq(right)
            }
            (Type::Unmaterialized(left), Type::Unmaterialized(right)) => {
                left.eq_unmaterialized(right)
            }
            (Type::ConstArray(left, len), Type::ConstArray(right, len2)) => {
                left.eq(right) && *len == *len2
            }
            _ => false,
        }
    }
}

impl Type {
    pub fn eq_unmaterialized(&self, other: &Type) -> bool {
        if let Type::Unmaterialized(left) = self {
            left.eq_unmaterialized(other)
        } else if let Type::Unmaterialized(right) = other {
            self.eq_unmaterialized(right)
        } else {
            match (self, other) {
                (Type::Field, Type::Field) => true,
                (Type::NamedType(left), Type::NamedType(right)) => left == right,
                (Type::Reference(left), Type::Reference(right)) => left.eq_unmaterialized(right),
                (Type::Array(left), Type::Array(right)) => left.eq_unmaterialized(right),
                (Type::UnderConstructionArray(left), Type::UnderConstructionArray(right)) => {
                    left.eq_unmaterialized(right)
                }
                (Type::ConstArray(left, len), Type::ConstArray(right, len2)) => {
                    left.eq_unmaterialized(right) && *len == *len2
                }
                _ => false,
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum ArithmeticOperator {
    Plus,
    Minus,
    Times,
}

#[derive(Clone)]
pub enum Expression {
    Constant {
        value: isize,
    },
    Variable {
        name: String,
    }, // includes arguments and components from match
    Let {
        name: String,
    },
    Define {
        name: String,
    },
    Algebraic {
        constructor: String,
        fields: Vec<ExpressionContainer>,
    },
    Arithmetic {
        operator: ArithmeticOperator,
        left: ExpressionContainer,
        right: ExpressionContainer,
    },
    Dematerialized {
        value: ExpressionContainer,
    },
    EqUnmaterialized {
        left: ExpressionContainer,
        right: ExpressionContainer,
    },
    EmptyConstArray {
        elem_type: Type,
    },
    ConstArray {
        elements: Vec<ExpressionContainer>,
    },
    ConstArrayConcatenation {
        left: ExpressionContainer,
        right: ExpressionContainer,
    },
    ConstArrayAccess {
        array: ExpressionContainer,
        index: usize,
    },
    ConstArraySlice {
        array: ExpressionContainer,
        from: usize,
        to: usize,
    },
} // components of an algebraic value can be extracted using match, don’t need to have an expression for them

#[derive(Clone)]
pub enum Statement {
    VariableDeclaration {
        name: String,
        tipo: Type,
    },
    Equality {
        left: ExpressionContainer,
        right: ExpressionContainer,
    },
    Reference {
        reference: ExpressionContainer,
        data: ExpressionContainer,
    },
    Dereference {
        data: ExpressionContainer,
        reference: ExpressionContainer,
    },
    EmptyUnderConstructionArray {
        array: ExpressionContainer,
        elem_type: Type,
    },
    UnderConstructionArrayPrepend {
        new_array: ExpressionContainer,
        elem: ExpressionContainer,
        old_array: ExpressionContainer,
    },
    ArrayFinalization {
        finalized: ExpressionContainer,
        under_construction: ExpressionContainer,
    },
    ArrayAccess {
        elem: ExpressionContainer,
        array: ExpressionContainer,
        index: ExpressionContainer,
    },
}

#[derive(Clone)]
pub struct FunctionCall {
    pub function: String,
    pub arguments: Vec<ExpressionContainer>,
}

#[derive(Clone)]
pub struct Branch {
    pub constructor: String,
    pub components: Vec<String>,
    pub body: Body,
}

#[derive(Clone)]
pub struct Match {
    pub value: ExpressionContainer,
    pub check_material: Material,
    pub branches: Vec<Branch>,
}

#[derive(Clone)]
pub struct Body {
    //bool equals whether statement is materialized or not
    pub statements: Vec<(Material, Statement)>,
    pub matches: Vec<Match>,
    pub function_calls: Vec<(Material, FunctionCall)>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ArgumentBehavior {
    In,
    Out,
}

#[derive(Clone)]
pub struct Argument {
    pub(crate) behavior: ArgumentBehavior,
    pub(crate) tipo: Type,
    pub(crate) value: Expression,
}

#[derive(Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Argument>,
    pub body: Body,
    pub inline: bool,
}

#[derive(Clone)]
pub struct AlgebraicTypeVariant {
    pub name: String,
    pub components: Vec<Type>,
}

#[derive(Clone)]
pub struct AlgebraicTypeDeclaration {
    pub name: String,
    pub variants: Vec<AlgebraicTypeVariant>,
}

pub struct Program {
    pub functions: Vec<Function>,
    pub algebraic_types: Vec<AlgebraicTypeDeclaration>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Material {
    Materialized,
    Dematerialized,
}

impl Material {
    pub fn same_type(&self, type1: &Type, type2: &Type) -> bool {
        match self {
            Material::Materialized => type1.eq(type2),
            Material::Dematerialized => type1.eq_unmaterialized(type2),
        }
    }

    pub fn assert_type(&self, type1: &Type, type2: &Type) -> Result<(), CompilationError> {
        if !self.same_type(type1, type2) {
            Err(CompilationError::UnexpectedType(
                type1.clone(),
                type2.clone(),
            ))
        } else {
            Ok(())
        }
    }

    pub fn wrap(&self, tipo: &Type) -> Type {
        match self {
            Material::Materialized => tipo.clone(),
            Material::Dematerialized => Type::Unmaterialized(Arc::new(tipo.clone())),
        }
    }
}

impl BitAndAssign for Material {
    fn bitand_assign(&mut self, other: Self) {
        if other == Material::Dematerialized {
            *self = Material::Dematerialized;
        }
    }
}
