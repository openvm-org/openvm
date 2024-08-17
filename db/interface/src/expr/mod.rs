use datafusion::{logical_expr::Expr, scalar::ScalarValue};

use self::expr::BinaryExpr;
use crate::committed_page::column::Column;

pub mod expr;

#[derive(Debug, Clone)]
pub enum AxiomDbExpr {
    Column(Column),
    Literal(u32),
    BinaryExpr(BinaryExpr),
}

impl AxiomDbExpr {
    pub fn from(expr: &Expr) -> Self {
        match expr {
            Expr::Column(column) => AxiomDbExpr::Column(Column {
                page_id: column.flat_name(),
                name: column.name().to_string(),
            }),
            Expr::Literal(literal) => {
                if let ScalarValue::UInt32(Some(value)) = literal {
                    AxiomDbExpr::Literal(*value)
                } else {
                    panic!("Expected a UInt32 literal")
                }
            }
            Expr::BinaryExpr(binary_expr) => {
                let left = Self::from(&binary_expr.left);
                let right = Self::from(&binary_expr.right);
                AxiomDbExpr::BinaryExpr(BinaryExpr {
                    left: Box::new(left),
                    op: BinaryExpr::op_to_comp(binary_expr.op),
                    right: Box::new(right),
                })
            }
            _ => panic!("Unsupported expression: {:?}", expr),
        }
    }
}
