use afs_page::single_page_index_scan::page_index_scan_input::Comp;
use datafusion::logical_expr::Operator;

use super::AxdbExpr;

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    /// The left operand of the binary expression.
    pub left: Box<AxdbExpr>,
    /// The comparison operator
    pub op: Comp,
    /// The side right operand of the binary expression.
    pub right: Box<AxdbExpr>,
}

impl BinaryExpr {
    pub fn op_to_comp(op: Operator) -> Comp {
        match op {
            Operator::Eq => Comp::Eq,
            Operator::Lt => Comp::Lt,
            Operator::LtEq => Comp::Lte,
            Operator::Gt => Comp::Gt,
            Operator::GtEq => Comp::Gte,
            _ => panic!("Unsupported operator: {}", op),
        }
    }
}
