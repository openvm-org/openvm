use datafusion::logical_expr::Operator;

use super::AfsExpr;

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    /// The left operand of the binary expression.
    pub left: Box<AfsExpr>,
    /// The comparison operator
    pub op: AfsOperator,
    /// The side right operand of the binary expression.
    pub right: Box<AfsExpr>,
}

/// Operators supported by AFS
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum AfsOperator {
    /// Expressions are equal
    Eq,
    /// Left side is smaller than right side
    Lt,
    /// Left side is smaller or equal to right side
    LtEq,
    /// Left side is greater than right side
    Gt,
    /// Left side is greater or equal to right side
    GtEq,
}

impl AfsOperator {
    pub fn from(op: Operator) -> Self {
        match op {
            Operator::Eq => AfsOperator::Eq,
            Operator::Lt => AfsOperator::Lt,
            Operator::LtEq => AfsOperator::LtEq,
            Operator::Gt => AfsOperator::Gt,
            Operator::GtEq => AfsOperator::GtEq,
            _ => panic!("Unsupported operator: {}", op),
        }
    }
}
