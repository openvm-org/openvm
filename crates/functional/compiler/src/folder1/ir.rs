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

pub const BOOLEAN_TYPE_NAME: &str = "Bool";
pub const TRUE_CONSTRUCTOR_NAME: &str = "True";
pub const FALSE_CONSTRUCTOR_NAME: &str = "False";
impl Type {
    pub fn boolean() -> Self {
        Self::NamedType(BOOLEAN_TYPE_NAME.to_string())
    }
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithmeticOperator {
    Plus,
    Minus,
    Times,
    Div,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BooleanOperator {
    And,
    Or,
    Xor,
}

#[derive(Clone, Debug)]
pub enum Expression {
    Constant {
        value: isize,
    },
    Variable {
        name: String,
        declares: bool,
        defines: bool,
        represents: bool,
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
    Eq {
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
    ConstArrayRepeated {
        element: ExpressionContainer,
        length: usize,
    },
    BooleanNot {
        value: ExpressionContainer,
    },
    BooleanBinary {
        left: ExpressionContainer,
        right: ExpressionContainer,
        operator: BooleanOperator,
    },
    Ternary {
        condition: ExpressionContainer,
        true_value: ExpressionContainer,
        false_value: ExpressionContainer,
    },
} // components of an algebraic value can be extracted using match, don’t need to have an expression for them

#[derive(Clone, Debug)]
pub enum Statement {
    VariableDeclaration {
        name: String,
        tipo: Type,
        represents: bool,
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

#[derive(Clone, Debug)]
pub struct FunctionCall {
    pub function: String,
    pub arguments: Vec<ExpressionContainer>,
}

#[derive(Clone, Debug)]
pub struct BranchComponent {
    pub name: String,
    pub represents: bool,
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub constructor: String,
    pub components: Vec<BranchComponent>,
    pub body: Body,
}

#[derive(Clone, Debug)]
pub struct Match {
    pub value: ExpressionContainer,
    pub check_material: Material,
    pub branches: Vec<Branch>,
}

#[derive(Clone, Debug)]
pub struct Body {
    //bool equals whether statement is materialized or not
    pub statements: Vec<(Material, Statement)>,
    pub matches: Vec<Match>,
    pub function_calls: Vec<(Material, FunctionCall)>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ArgumentBehavior {
    In,
    Out,
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub(crate) behavior: ArgumentBehavior,
    pub(crate) tipo: Type,
    pub(crate) name: String,
    pub(crate) represents: bool,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Argument>,
    pub body: Body,
    pub inline: bool,
}

#[derive(Clone, Debug)]
pub struct AlgebraicTypeVariant {
    pub name: String,
    pub components: Vec<Type>,
}

#[derive(Clone, Debug)]
pub struct AlgebraicTypeDeclaration {
    pub name: String,
    pub variants: Vec<AlgebraicTypeVariant>,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub algebraic_types: Vec<AlgebraicTypeDeclaration>,
    pub functions: Vec<Function>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
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
