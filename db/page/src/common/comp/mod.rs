pub mod air;
pub mod columns;

use derive_more::Display;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Display, Clone, Deserialize, Serialize, PartialEq)]
pub enum Comp {
    #[default]
    Lt,
    Lte,
    Eq,
    Gte,
    Gt,
}
