use datafusion::{logical_expr::Expr, scalar::ScalarValue};

use self::expressions::BinaryExpr;
use crate::committed_page::column::Column;

pub mod expressions;

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
            Expr::Literal(literal) => match literal {
                ScalarValue::UInt32(Some(value)) => AxiomDbExpr::Literal(*value as u16 as u32),
                ScalarValue::Utf8(Some(value)) => {
                    let parsed_value = value.parse::<u16>().expect("Expected a valid u16 string");
                    AxiomDbExpr::Literal(parsed_value as u32)
                }
                // Handles CSV files where the numeric values are interpreted as Int64 by default
                ScalarValue::Int64(Some(value)) => AxiomDbExpr::Literal(*value as u16 as u32),
                _ => panic!("Unsupported literal type: {:?}", literal),
            },
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
