use afs_chips::{
    group_by::group_by_input::GroupByOperation, single_page_index_scan::page_index_scan_input::Comp,
};
use color_eyre::eyre::{eyre, Result};
use serde_derive::{Deserialize, Serialize};
use std::str::FromStr;

use crate::{afs_interface::utils::string_to_table_id, table::types::TableId};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Operand {
    Index,
}

impl FromStr for Operand {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s_upper = s.to_uppercase();
        match s_upper.as_str() {
            "INDEX" => Ok(Operand::Index),
            _ => Err(eyre!("Invalid operand: {}", s)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ReadOp {
    pub index: String,
}

impl ReadOp {
    pub fn parse(args: Vec<String>) -> Result<Self> {
        if args.len() != 1 {
            return Err(eyre!("Invalid number of arguments for READ op"));
        }
        let index = args[0].clone();
        Ok(Self { index })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct WriteOp {
    pub index: String,
    pub data: String,
}

impl WriteOp {
    pub fn parse(args: Vec<String>) -> Result<Self> {
        if args.len() != 2 {
            return Err(eyre!("Invalid number of arguments for WRITE op"));
        }
        let index = args[0].clone();
        let data = args[1].clone();
        Ok(Self { index, data })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct InsertOp {
    pub index: String,
    pub data: String,
}

impl InsertOp {
    pub fn parse(args: Vec<String>) -> Result<Self> {
        if args.len() != 2 {
            return Err(eyre!("Invalid number of arguments for INSERT op"));
        }
        let index = args[0].clone();
        let data = args[1].clone();
        Ok(Self { index, data })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct FilterOp {
    pub table_id: TableId,
    pub operand: Operand,
    pub predicate: Comp,
    pub value: String,
}

impl FilterOp {
    pub fn parse(args: Vec<String>) -> Result<Self> {
        if args.len() != 4 {
            return Err(eyre!("Invalid number of arguments for predicate filter op"));
        }
        let table_id = string_to_table_id(args[0].clone());
        let operand = Operand::from_str(&args[1])?;
        let predicate = match args[2].as_str() {
            "=" => Comp::Eq,
            "<" => Comp::Lt,
            "<=" => Comp::Lte,
            ">" => Comp::Gt,
            ">=" => Comp::Gte,
            _ => return Err(eyre!("Invalid predicate: {}", args[1])),
        };
        let value = args[3].clone();
        Ok(Self {
            table_id,
            operand,
            predicate,
            value,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct GroupByOp {
    pub table_id: TableId,
    pub group_by_cols: Vec<usize>,
    pub agg_col: usize,
    pub op: GroupByOperation,
}

impl GroupByOp {
    pub fn parse(args: Vec<String>) -> Result<Self> {
        if args.len() < 4 {
            return Err(eyre!("GROUP BY op requires at least 4 arguments"));
        }
        let table_id = string_to_table_id(args[0].clone());
        let num_args = args.len();
        let group_by_cols = args[1..num_args - 2]
            .iter()
            .map(|s| s.parse::<usize>().unwrap())
            .collect();
        let agg_col = args[num_args - 2].parse::<usize>()?;
        let op = GroupByOperation::from_str(&args[num_args - 1])?;
        Ok(Self {
            table_id,
            group_by_cols,
            agg_col,
            op,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct InnerJoinOp {
    pub table_id_left: TableId,
    pub table_id_right: TableId,
    pub fkey_start: usize,
    pub fkey_end: usize,
}

impl InnerJoinOp {
    pub fn parse(args: Vec<String>) -> Result<Self> {
        if args.len() != 4 {
            return Err(eyre!("Invalid number of arguments for INNER JOIN op"));
        }
        let table_id_left = string_to_table_id(args[0].clone());
        let table_id_right = string_to_table_id(args[1].clone());
        let fkey_start = args[2].parse::<usize>()?;
        let fkey_end = args[3].parse::<usize>()?;
        Ok(Self {
            table_id_left,
            table_id_right,
            fkey_start,
            fkey_end,
        })
    }
}
