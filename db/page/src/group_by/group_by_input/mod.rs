use std::str::FromStr;

use air::GroupByAir;
use serde::{Deserialize, Serialize};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GroupByOperation {
    Sum,
    Product,
}

impl FromStr for GroupByOperation {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s_upper = s.to_uppercase();
        match s_upper.as_str() {
            "SUM" => Ok(GroupByOperation::Sum),
            "PRODUCT" => Ok(GroupByOperation::Product),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid operand",
            )),
        }
    }
}
